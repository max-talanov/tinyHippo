#!/usr/bin/env python3
"""
bidirectional_replay_watson2025_scaled.py
==========================================
Bidirectional sequence replay — Watson et al. 2025 two-layer CA3 circuit.
Bio-plausible scaling for MareNostrum5 (or any MPI+OpenMP HPC cluster).

SCALE OPTIONS
=============
  --scale 1pct    (default)  Test/debug.  ~7.7k  neurons.  1 node,  <5 min.
  --scale 12pct              HPC dev.    ~93k   neurons. 16 nodes, ~25 min.
  --scale 100pct             Full rat.  ~781k   neurons. 256 nodes, ~3-5 h.

Reference neuron counts (Andersen et al. 2007, "The Hippocampus Book"):
  CA3 pyramidal total : ~330,000  (SUP 80% = 264k, DEEP 20% = 66k)
  CA3 interneurons    : ~33,000   (INT_SUP 75%, INT_DEEP 25%)
  CA1 pyramidal       : ~460,000
  CA1 basket          : ~14,000
  CA1 OLM             : ~9,000

CONNECTIVITY STRATEGY
======================
v2 KEY CHANGE: All large-population connections use NEST-native C++ rules
(fully MPI-parallel) instead of the Python bernoulli_connect() loop:

  fixed_indegree     — E<->I, Schaffer collaterals, CA1 local.
                       Each post neuron gets exactly K inputs from pre.
  pairwise_bernoulli — group-to-group sequence chain (small groups,
                       variable p; also handles p near 0 for D->S).

This eliminates the dominant serial bottleneck and makes the build phase
scale with MPI ranks rather than running in a single Python process.

Target in-degrees (scale-invariant, biologically motivated):
  Sequence chain (group-level, per neuron):
    fwd SUP: 20   bwd SUP: 5   local SUP: 15
    SUP->DEEP: 20  local DEEP: 8  fwd DEEP: 5  D->S: 1 (~0.18% Watson)
  Full-population (per post neuron):
    CA3 SUP->INT_SUP: 50    INT_SUP->SUP: 150
    CA3 DEEP->INT_DEEP: 20  INT_DEEP->DEEP: 80
    Cross-layer inh: 10     INT->INT: 30
    Schaffer SUP->CA1 PYR: 3000   DEEP->CA1 PYR: 1000
    Schaffer SUP->basket: 500     DEEP->basket: 200
    CA1: EE=5  EI=10  IE=50  OE=20

Usage
-----
  python bidirectional_replay_watson2025_scaled.py              # 1pct
  python bidirectional_replay_watson2025_scaled.py --scale 12pct
  python bidirectional_replay_watson2025_scaled.py --scale 100pct --no-figures
  mpirun -n 64 python bidirectional_replay_watson2025_scaled.py --scale 12pct

Requirements:  NEST >= 3.x, numpy, matplotlib, scipy (optional), tiny.py
"""

import argparse
import sys
import os
import time
import warnings
from dataclasses import dataclass
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

try:
    import h5py as _h5py_module
    _HDF5_AVAILABLE = True
except ImportError:
    _HDF5_AVAILABLE = False

try:
    from mpi4py import MPI as _MPI
    _MPI_AVAILABLE = True
except ImportError:
    _MPI_AVAILABLE = False

import nest


# ============================================================================
# MPI helpers
# ============================================================================

def _mpi_rank() -> int:
    """Return this process's MPI rank (0 if not running under MPI).

    Prefers mpi4py over nest.GetKernelStatus() because NEST may be built
    without MPI support, in which case every srun task reports rank=0 and
    all tasks race to write the same HDF5 file.
    """
    if _MPI_AVAILABLE:
        return _MPI.COMM_WORLD.Get_rank()
    ks = nest.GetKernelStatus()
    return int(ks.get("rank", ks.get("process_id", 0)))


def _mpi_size() -> int:
    """Return total number of MPI ranks (1 if not running under MPI).

    Prefers mpi4py for the same reason as _mpi_rank().
    """
    if _MPI_AVAILABLE:
        return _MPI.COMM_WORLD.Get_size()
    ks = nest.GetKernelStatus()
    return int(ks.get("total_num_processes",
               ks.get("num_processes",
               ks.get("mpi_num_processes", 1))))


def _gather_spikes(local_t: np.ndarray, local_s: np.ndarray):
    """
    Gather spike arrays from all MPI ranks to rank 0, returned sorted by time.

    On rank 0  : returns (all_times, all_senders) merged and time-sorted.
    On rank > 0: returns (empty, empty) — caller must not use the result.

    Falls back gracefully when mpi4py is unavailable (single-rank case).
    """
    if _mpi_size() == 1:
        # Single rank: nothing to gather
        order = np.argsort(local_t, kind="stable")
        return local_t[order], local_s[order]

    if not _MPI_AVAILABLE:
        # Multi-rank MPI run but mpi4py not importable — warn once from rank 0
        if _mpi_rank() == 0:
            import warnings
            warnings.warn(
                "mpi4py is not installed.  HDF5 will contain only rank-0 spikes.\n"
                "Install mpi4py (e.g. pip install mpi4py) for complete data.",
                RuntimeWarning, stacklevel=3,
            )
        # Return local data on rank 0, empty on others — partial but not corrupt
        if _mpi_rank() == 0:
            order = np.argsort(local_t, kind="stable")
            return local_t[order], local_s[order]
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)

    comm = _MPI.COMM_WORLD
    comm.Barrier()                         # ensure all ranks finished Simulate()

    all_t = comm.gather(local_t, root=0)
    all_s = comm.gather(local_s, root=0)

    if comm.Get_rank() == 0:
        t_merged = np.concatenate(all_t).astype(np.float32)
        s_merged = np.concatenate(all_s).astype(np.int32)
        order    = np.argsort(t_merged, kind="stable")
        return t_merged[order], s_merged[order]

    return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)
from tiny import (
    safe_set_seeds,
    maybe_make_theta_generators,
    make_swr_event_generators,
)


# ============================================================================
# GRADUAL SCALE  (any integer percentage 1–100)
# ============================================================================

# Reference neuron counts at 100% (Andersen et al. 2007)
_REF_100PCT = dict(
    N_ca3_sup      = 264_000,
    N_ca3_deep     =  66_000,
    N_ca3_int_sup  =  24_000,
    N_ca3_int_deep =   8_000,
    N_ca1_pyr      = 460_000,
    N_ca1_basket   =  14_000,
    N_ca1_olm      =   9_000,
)

# Reference counts for cortical modules (kept separate so gradual-scale
# hippocampal calculations are unaffected when --ec-lii is not requested).
_REF_CORTEX = dict(
    N_ec_lii = 100_000,   # EC layer II/III stellate cells (direct CA1 recipient)
)

# Reference counts for the dentate gyrus (Phase 6.2, --dg). Kept separate for
# the same reason as _REF_CORTEX: hippocampal (CA3/CA1) gradual-scale maths is
# untouched when --dg is not requested. Andersen et al. 2007, "The Hippocampus
# Book", rat:
#   granule cells    ~1,200,000  (the input/pattern-separation stage)
#   mossy cells         ~30,000  (hilar; split into low/high threshold per
#                                 Kassab & Alexandre 2018 — see the confirmed
#                                 f-I calibration, nest_dg_ca3_fi_calibration.py)
#   DG basket/HIPP      ~10,000  (feedback inhibition -> sparse coding)
_REF_DG = dict(
    N_dg_gc       = 1_200_000,
    N_dg_mc_low   =    15_000,   # 50% of ~30k mossy cells
    N_dg_mc_high  =    15_000,   # 50%
    N_dg_basket   =    10_000,
)


def _round_to_multiple(n: float, m: int) -> int:
    """Round n to the nearest multiple of m, minimum m."""
    return max(m, int(round(n / m)) * m)


def build_scale_config(pct: int) -> dict:
    """
    Build a scale configuration for any integer percentage 1–100.

    n_seq_groups scales as max(10, round(10 * sqrt(pct))):
      1% → 10 groups,  4% → 20,  25% → 50,  100% → 100
    This keeps groups-per-population in a biologically sensible range
    (~130–2640 CA3 SUP neurons per group) regardless of scale.

    All neuron counts are rounded to the nearest multiple of n_seq_groups
    so that N % n_seq_groups == 0 is always satisfied.

    Suggested CPUs (single-node OpenMP):
      total_N <   20k → 8   threads
      total_N <  100k → 16  threads
      total_N <  300k → 28  threads
      total_N >= 300k → 50  threads  (full node)
    """
    pct = int(pct)
    if not 1 <= pct <= 100:
        raise ValueError(f"--scale must be an integer 1–100, got {pct}")

    f = pct / 100.0  # linear scaling factor

    # n_seq_groups: sub-linear so small runs still have meaningful sequences
    n_groups = max(10, round(10 * pct ** 0.5))

    cfg = {"label": f"{pct}% scale", "n_seq_groups": n_groups}
    for key, ref in _REF_100PCT.items():
        cfg[key] = _round_to_multiple(ref * f, n_groups)

    total_N = sum(cfg[k] for k in _REF_100PCT)
    if   total_N <  20_000:  cfg["n_threads_default"] = 8
    elif total_N < 100_000:  cfg["n_threads_default"] = 16
    elif total_N < 300_000:  cfg["n_threads_default"] = 28
    else:                    cfg["n_threads_default"] = 50

    return cfg


# ============================================================================
# TARGET IN-DEGREES  (biologically motivated; scale-invariant)
# p = min(1.0, K / N_pre)   for pairwise_bernoulli
# K = min(K, N_pre)          for fixed_indegree
# ============================================================================

TARGET_INDEGREE = {
    # Sequence chain (group-level, per post neuron)
    "seq_fwd"              :    20,
    "seq_bwd"              :     5,
    "sup_local"            :    15,
    "sup_to_deep"          :    20,
    "deep_local"           :     8,
    "deep_fwd"             :     5,
    "deep_to_sup"          :     1,   # D->S near-absent (Watson ~0.18%)
    # CA3 E<->I (full population)
    "ca3_EI_sup"           :    50,
    "ca3_EI_deep"          :    20,
    "ca3_IE_sup"           :   150,
    "ca3_IE_deep"          :    80,
    "ca3_IE_cross"         :    10,
    "ca3_II"               :    30,
    # Schaffer collaterals (Ishizuka et al. 1990)
    "schaffer_sup_pyr"     : 3_000,
    "schaffer_deep_pyr"    : 1_000,
    "schaffer_sup_basket"  :   500,
    "schaffer_deep_basket" :   200,
    # CA1 local
    "ca1_EE"               :     5,
    "ca1_EI"               :    10,
    "ca1_IE"               :    50,
    "ca1_OE"               :    20,
    # Cortical projections (phase 1+)
    "ca1_ec_lii"           :   500,   # CA1 PYR -> EC LII (Naber et al. 2001)
    # Dentate gyrus (Phase 6.2, --dg). Scale-invariant in-degrees.
    "dg_gc_basket"         :    50,   # GC -> DG basket  (E->I, recruits inhibition)
    "dg_basket_gc"         :   140,   # DG basket -> GC  (I->E, feedback -> sparse)
    "dg_gc_mc"             :    30,   # GC -> mossy cell (mossy collaterals in hilus)
    "dg_mc_gc"             :     8,   # mossy cell -> GC (associational back-projection)
    "dg_mc_basket"         :    20,   # mossy cell -> DG basket (drives feedback inh)
    # Mossy fibre DG -> CA3: the "detonator" synapse. Each CA3 pyramidal cell
    # receives very FEW mossy-fibre inputs (~15-50 in rat; Acsady et al. 1998,
    # Henze et al. 2002) but each is disproportionately powerful. Low in-degree,
    # high weight -- the opposite of the dense Schaffer collaterals.
    "dg_mf_ca3_sup"        :    15,   # GC -> CA3 SUP  (primary MF target)
    "dg_mf_ca3_deep"       :     8,   # GC -> CA3 DEEP (weaker; SUP is main target)
    # Perforant path EC LII -> DG granule cells. This is the projection that
    # CLOSES the EC->DG->CA3->CA1->EC loop; without it DG is driven by a Poisson
    # stand-in and the "loop" is open at its entry point. Rat granule cells
    # receive ~4,000 PP synapses each (Andersen 2007); the scale-invariant
    # in-degree here is the usual reduced-model surrogate.
    # NOTE this closes a POSITIVE-FEEDBACK loop, so its gain must stay well
    # below unity. At K=100 w=1.2 the loop ran away (CA3 7.8 -> 73 Hz, granule
    # cells 100% active, pattern separation destroyed). See w_ec_dg/pp_residual.
    "ec_dg_pp"             :    50,   # EC LII -> GC
}


def K(key, n_pre):
    """Indegree clamped to pre-population size (for fixed_indegree)."""
    return min(TARGET_INDEGREE[key], int(n_pre))


def p(key, n_pre):
    """Probability achieving target indegree from n_pre (for pairwise_bernoulli)."""
    return min(1.0, TARGET_INDEGREE[key] / max(n_pre, 1))


# ============================================================================
# NEST helpers
# ============================================================================

def _to_nc(x):
    if isinstance(x, nest.NodeCollection):
        return x
    if isinstance(x, (int, np.integer)):
        return nest.NodeCollection([int(x)])
    return nest.NodeCollection([int(i) for i in x])


_NEST_MAX_LCID = 134_217_726          # 2^27 - 2: hard NEST per-VP per-synapse limit
_conn_count_per_vp: dict = {}         # cumulative static_synapse connections per VP


def _preflight_conn_check(pre_size: int, post_size: int, indegree: int, label: str = "") -> None:
    """Raise an informative error before NEST does if adding this fixed_indegree
    connection would push the cumulative static_synapse count past NEST's limit."""
    ks  = nest.GetKernelStatus()
    thr = ks.get("local_num_threads", ks.get("num_threads", ks.get("threads", 1)))
    mpi = ks.get("total_num_processes",
                  ks.get("num_processes", ks.get("mpi_num_processes", 1)))
    n_vp = max(thr * mpi, 1)

    new_total   = post_size * indegree
    new_per_vp  = new_total / n_vp
    prev        = _conn_count_per_vp.get("static_synapse", 0.0)
    after       = prev + new_per_vp

    tag = f" [{label}]" if label else ""
    print(f"  [conn-check{tag}] +{new_per_vp:,.0f}/VP  cumul={after:,.0f}/VP  "
          f"limit={_NEST_MAX_LCID:,}  VPs={n_vp}  ({after/_NEST_MAX_LCID*100:.1f}%)")

    if after > _NEST_MAX_LCID:
        raise RuntimeError(
            f"\n[Too many connections — would exceed NEST limit BEFORE calling Connect]\n"
            f"  Connection{tag}: {pre_size} pre × {post_size} post, indegree={indegree}\n"
            f"  This call adds ~{new_per_vp:,.0f} synapses/VP to static_synapse.\n"
            f"  Cumulative after this call: {after:,.0f}/VP  (limit {_NEST_MAX_LCID:,})\n"
            f"  Current VPs: {mpi} MPI × {thr} threads = {n_vp}\n\n"
            f"  Fixes:\n"
            f"   1) Increase MPI ranks: add --ntasks-per-node=2 (doubles VPs per node)\n"
            f"   2) Verify NEST actually accepted local_num_threads={thr} (check VP printout above)\n"
            f"   3) Reduce INDEGREES['schaffer_sup_pyr'/'schaffer_deep_pyr'] until sum < "
            f"{int(_NEST_MAX_LCID * n_vp / post_size):,}\n"
        )
    _conn_count_per_vp["static_synapse"] = after


def jittered_delay(base, jitter, lo=0.5):
    """Per-synapse delay: scalar when jitter<=0, else uniform(base-j, base+j).

    Axonal conduction delays are heterogeneous in reality, and the homogeneity
    here is load-bearing: with one delay per projection every downstream neuron
    integrates its inputs with ZERO temporal differentiation, so the timing
    structure that carries pattern identity in CA3 averages out at each hop
    (measured cell-level separation CA3 +0.143 -> CA1 +0.063 -> EC ~0).
    Heterogeneous delays make different post neurons sensitive to different
    input timings, which is the substrate a polychronous group needs
    (Izhikevich 2006).

    Applied only to FEEDFORWARD readout projections. The CA3 sequence chain
    keeps its scalar d_seq: that delay is tuned against the scaffold step and
    randomising it would break replay itself rather than test transmission.
    """
    if jitter is None or jitter <= 0:
        return float(base)
    return nest.random.uniform(min=max(lo, base - jitter), max=base + jitter)


# ============================================================================
# Global heterogeneity — every cell and every synapse is an individual
# ============================================================================
# Real neurons are not copies of one another. This model treated them as copies:
# within a population every cell shared identical a/b/c/d/I_e (only V_m varied,
# and that is a transient -- all cells relax to the same rest), and every
# synapse of a projection carried an identical weight and an identical delay.
#
# That homogeneity is the root of the synchrony ceiling in §13: K identical
# inputs arriving at one instant sum to exactly K*w, so the gain of any
# projection is capped by the postsynaptic threshold gap divided by K, rather
# than by anything biological. Spreading delays turns that instantaneous sum
# into a temporal integral, and spreading weights and cell excitability means
# the same input recruits a reproducible SUBSET rather than all-or-none.
_HET = {"w_cv": 0.0, "delay_cv": 0.0, "neuron_cv": 0.0, "wcomp": 1.0}


def set_heterogeneity(w_cv=0.0, delay_cv=0.0, neuron_cv=0.0, wcomp=1.0):
    """Set the model-wide defaults consulted by fixed_connect / het_params.

    wcomp scales every weight. Heterogeneity LOWERS effective gain: spreading
    arrival times reduces coincident summation, and because firing is a
    threshold nonlinearity, spreading input around a fixed mean moves the
    population mean rate DOWN rather than leaving it unchanged. The model's
    operating point was tuned with synchronous volleys, so switching it on
    without compensation drops CA1 PYR ~4x at cv 0.15 (4.05 -> 0.97 Hz) while
    interneurons barely move. --delay-jitter-wcomp already does this for the
    five jittered projections; this is the model-wide equivalent.
    """
    _HET["w_cv"] = float(w_cv or 0.0)
    _HET["delay_cv"] = float(delay_cv or 0.0)
    _HET["neuron_cv"] = float(neuron_cv or 0.0)
    _HET["wcomp"] = float(wcomp if wcomp else 1.0)
    if any(v for k, v in _HET.items() if k != "wcomp") or _HET["wcomp"] != 1.0:
        print(f"  [het] model-wide heterogeneity: weights cv={_HET['w_cv']}, "
              f"delays cv={_HET['delay_cv']}, neuron params cv={_HET['neuron_cv']}, "
              f"weight compensation x{_HET['wcomp']}")


def wcomp_w(weight):
    """Apply the excitatory compensation to an externally-driven weight.

    fixed_connect handles projection weights, but the Poisson drives -- the
    tonic CA3 excitation, the per-population background, the sharp-wave/ripple
    shaping -- are wired with a direct nest.Connect and so were never
    compensated. That is why CA3 SUP stayed pinned at ~4.5 Hz across wcomp
    1.5/2.0/3.0 while CA1, which is driven through fixed_connect Schaffer
    collaterals, did respond.

    Compensation must reach excitation onto PRINCIPAL cells only. Applying it
    to excitation onto interneurons -- whether an E->I projection or a Poisson
    drive aimed at a basket population -- amplifies the inhibitory side harder
    than the excitatory one, because interneurons sit closer to threshold.
    Measured when the drives were compensated indiscriminately: CA1 basket
    52.8 -> 99.5 Hz and CA1 PYR 4.05 -> 0.06 Hz at wcomp 2.5. Call sites
    targeting interneurons pass compensate=False.
    """
    w = float(weight)
    return w * _HET["wcomp"] if (w > 0 and _HET["wcomp"] != 1.0) else w


# Izhikevich parameters that may be spread per cell. a and d are strictly
# positive; b enters the fixed points as (5-b) so it is kept well away from 5;
# c is the reset potential. I_e shifts rheobase directly and is the persistent
# excitability knob (V_m heterogeneity is not -- it washes out at rest).
# A CV is only meaningful for a parameter with a true zero. a, b and d are
# rates/gains and scale relatively; c and I_e are voltages/currents whose zero
# point is arbitrary, so a relative CV is nonsense for them -- 30% of c = -65 mV
# is 19.5 mV, which drew cells with a reset potential ABOVE the -50 mV threshold
# and made them self-ignite (measured: interneurons 20 -> 48 Hz, CA1 PYR
# collapsing to 0.32 Hz, rho_rev failing at -0.309). Those get an absolute
# reference scale instead, multiplied by the same cv knob.
_HET_NEURON_KEYS = ("a", "b", "c", "d", "I_e")
# b is the BIFURCATION parameter: rheobase = (5-b)^2/0.16 - 140 - I_e goes
# NEGATIVE (the cell fires tonically with no input at all) once
# (5-b)^2 < 0.16*(140+I_e). At cv 0.30 a symmetric Gaussian around b=0.2 puts
# 13% of cells past that for I_e=0, and 39% for EC LV's I_e=3.0. Measured
# consequence at 12%: 25-30% of DG baskets ran at 38-189 Hz and clamped the
# granule population to 0.49% active against a 2-4% target. b therefore gets a
#
# The model deliberately places cells NEAR that bifurcation -- EC LV sits at
# b=0.2 with b_crit=0.217, a 1.0 pA rheobase -- so b cannot carry much spread.
# It gets B_CV_FACTOR of the global cv plus a hard ceiling just below b_crit,
# and populations already at or above that ceiling keep their calibrated b
# unchanged rather than being clipped down to it (clipping there would move the
# mean and undo the f-I calibration). Excitability heterogeneity is carried by
# I_e instead, which is monotone and has no bifurcation.
B_CV_FACTOR = 0.10
B_MARGIN = 0.02
_HET_SCALE = {"a": ("rel", None), "b": ("rel", None), "d": ("rel", None),
              "c": ("abs", 5.0),      # cv 0.3 -> 1.5 mV spread of reset
              "I_e": ("abs", 3.0)}    # cv 0.3 -> 0.9 pA spread of drive
# c stays clear of the ~-50 mV threshold: a cell must not reset above it.
_HET_CLIP = {"a": (1e-3, 1.0), "b": (0.01, 4.5), "c": (-80.0, -55.0),
             "d": (0.05, 30.0), "I_e": (None, None)}


def het_params(params, n, rng, cv=None):
    """Expand a scalar Izhikevich param dict into per-cell distributions.

    Keeps the calibrated value as the MEAN, so f-I calibration (e.g. the
    MC_HIGH/MC_LOW 4.97x rheobase ratio) is preserved in expectation while the
    cells stop being copies of one another.
    """
    cv = _HET["neuron_cv"] if cv is None else float(cv)
    if cv <= 0 or n <= 0:
        return dict(params)
    out = dict(params)
    for k in _HET_NEURON_KEYS:
        if k not in out:
            continue
        if not isinstance(out[k], (int, float)):
            continue            # already per-cell (e.g. I_e set explicitly)
        base = float(out[k])
        mode, ref = _HET_SCALE.get(k, ("rel", None))
        if mode == "rel":
            if base == 0.0:
                continue        # nothing to scale relatively
            sd = abs(base) * cv
        else:
            sd = float(ref) * cv    # absolute: applies even when base is 0
        if sd <= 0:
            continue
        vals = rng.normal(base, sd, n)
        lo, hi = _HET_CLIP.get(k, (None, None))
        if k == "b":
            # keep every cell on the quiescent side of the saddle-node
            _ie = params.get("I_e", 0.0)
            _ie = float(np.mean(_ie)) if not isinstance(_ie, (int, float)) else float(_ie)
            b_crit = 5.0 - np.sqrt(max(0.16 * (140.0 + _ie), 0.0))
            cap = b_crit - B_MARGIN
            if base >= cap:
                continue        # already at the edge by design: leave it alone
            vals = rng.normal(base, sd * B_CV_FACTOR, n)
            hi = cap if hi is None else min(hi, cap)
        if lo is not None or hi is not None:
            vals = np.clip(vals, lo if lo is not None else -np.inf,
                           hi if hi is not None else np.inf)
        out[k] = vals.tolist()
    return out


def jittered_weight(base, cv):
    """Per-synapse weight: scalar when cv<=0, else normal(base, |base|*cv).

    Sign is preserved by clipping at 5% of base -- a Gaussian around a small
    weight will otherwise cross zero and silently turn inhibitory synapses
    excitatory (the failure already seen once in the mPFC recurrent hook).

    Why this matters: with a scalar weight every one of a cell's K inputs is
    identical, so the only thing distinguishing two postsynaptic cells is which
    presynaptic cells they happened to draw. Combined with a scalar delay that
    makes the whole volley land at one instant with one amplitude, which is the
    synchrony ceiling that caps K*w at the 20 mV granule gap.
    """
    if cv is None or cv <= 0:
        return float(base)
    b = float(base)
    sd = abs(b) * float(cv)
    par = nest.random.normal(mean=b, std=sd)
    return nest.math.max(par, 0.05 * b) if b > 0 else nest.math.min(par, 0.05 * b)


def fixed_connect(pre, post, indegree, weight, delay, w_cv=None,
                  compensate=True):
    """
    NEST native fixed_indegree — C++, fully MPI-parallel.
    Pre-flight checks cumulative static_synapse count vs NEST's 134M/VP limit.
    Each post neuron receives exactly `indegree` inputs drawn from pre.

    `weight` and `delay` may each be a scalar or a NEST parameter; w_cv>0 turns
    a scalar weight into normal(weight, |weight|*w_cv).
    """
    if w_cv is None:
        w_cv = _HET["w_cv"]
    # Compensation applies to EXCITATORY weights only. Scaling inhibition by
    # the same factor leaves E/I unchanged and merely raises loop gain, which
    # in this inhibition-dominated network suppresses the pyramids further:
    # measured at wcomp 1.5, ca1_basket held at 52.8 Hz (unmoved) while CA3 SUP
    # fell 4.54 -> 3.57 Hz. What heterogeneity costs is coincident EXCITATORY
    # summation, so that is what has to be restored.
    if compensate and isinstance(weight, (int, float)) and weight > 0:
        weight = wcomp_w(weight)
    # Global delay heterogeneity: a scalar delay becomes uniform(d+/-cv*d).
    if isinstance(delay, (int, float)) and _HET["delay_cv"] > 0:
        delay = jittered_delay(float(delay), float(delay) * _HET["delay_cv"])
    pre_nc  = _to_nc(pre)
    post_nc = _to_nc(post)
    _preflight_conn_check(
        pre_size  = len(pre_nc),
        post_size = len(post_nc),
        indegree  = int(indegree),
        label     = f"pre={len(pre_nc)} post={len(post_nc)} K={indegree}",
    )
    nest.Connect(
        pre_nc, post_nc,
        conn_spec={"rule": "fixed_indegree", "indegree": int(indegree)},
        # delay may be a scalar OR a NEST parameter (e.g. nest.random.uniform)
        # for per-synapse heterogeneity -- see jittered_delay().
        syn_spec={"weight": (jittered_weight(weight, w_cv)
                             if isinstance(weight, (int, float)) else weight),
                  "delay": float(delay) if isinstance(delay, (int, float)) else delay},
    )


def clustered_fixed_connect(pre, post, indegree, weight, delay, field_centers,
                             cluster_sigma, w_cv=None, compensate=True, seed=42):
    """Like fixed_connect, but each post neuron's `indegree` sources are drawn
    from a LOCAL neighbourhood of `pre` (ring distance, sigma=cluster_sigma)
    around that post neuron's own topographic location, instead of uniformly
    at random across all of `pre`.

    Post neurons are assigned a topographic centre by an even tiling of the
    ring by creation index (post[0] -> 0.0, post[-1] -> just under 1.0) -- an
    arbitrary but well-defined stand-in for a medial-lateral anatomical
    gradient, since `post` (e.g. DG granule cells) has no spatial coordinate
    of its own. `field_centers` gives each `pre` cell's own ring location
    (e.g. EC LII's assigned place-field centre) and must be in the same
    creation-index order as `pre`.

    Modelling assumption: axons that are close on this ring are more likely
    to synapse onto the same dendritic-tree neighbourhood -- the biological
    motivation for clustering, vs. today's uniform-random sampling. Verified
    to restore a real pattern-identity signal at the single-cell level (see
    RESULTS.md SS17): DG's mean ring-distance from the active pattern's
    centre drops from 0.273 (uniform K=50, worse than the 0.25 chance null)
    to 0.236 (clustered K=50, sigma=0.05) -- closer to EC LII's own 0.22 than
    either the uniform baseline or a naive K=1 fan-in cut (0.253) reached.

    One nest.Connect call per post neuron (all_to_all, indegree pre cells to
    1 post cell) -- ~7s for 12,000 post neurons at 1% scale; not vectorized
    further since indegree differs in identity (not just count) per neuron.
    """
    if w_cv is None:
        w_cv = _HET["w_cv"]
    if compensate and isinstance(weight, (int, float)) and weight > 0:
        weight = wcomp_w(weight)
    weight_param = (jittered_weight(weight, w_cv)
                    if isinstance(weight, (int, float)) else weight)

    pre_nc = _to_nc(pre)
    post_nc = _to_nc(post)
    pre_ids = np.array(pre_nc.tolist())
    post_ids = np.array(post_nc.tolist())
    n_pre, n_post = len(pre_ids), len(post_ids)
    field_centers = np.asarray(field_centers)
    post_center = (np.arange(n_post) / n_post) % 1.0
    rng = np.random.default_rng(seed + 97)

    for i in range(n_post):
        d = np.abs(field_centers - post_center[i])
        d = np.minimum(d, 1.0 - d)
        w = np.exp(-(d ** 2) / (2 * cluster_sigma ** 2))
        w = w / w.sum()
        chosen = rng.choice(n_pre, size=int(indegree), replace=False, p=w)
        src_nc = nest.NodeCollection(sorted(pre_ids[chosen].tolist()))
        tgt_nc = nest.NodeCollection([int(post_ids[i])])
        nest.Connect(src_nc, tgt_nc, conn_spec="all_to_all",
                     syn_spec={"weight": weight_param, "delay": delay})


def grouped_fixed_connect(pre, post, indegree, weight, delay, n_groups, group_frac,
                           w_cv=None, compensate=True, seed=42):
    """Like fixed_connect, but each post neuron's `indegree` sources are drawn
    with a bias toward ONE of `n_groups` contiguous blocks `pre` is
    partitioned into, instead of uniformly across all of `pre`.

    Built for Schaffer collaterals (CA3->CA1), where clustered_fixed_connect's
    continuous-ring approach doesn't apply: CA3's sequence groups are
    deliberately INTERLEAVED by neuron index (patterns = groups {p, p+P, ...}
    for stride P, see build_replay_network's docstring on `patterns`) so that
    nearby-index CA3 cells belong to DIFFERENT patterns by design -- clustering
    by index the way clustered_fixed_connect does for EC LII->GC would be
    close to meaningless here. Group membership (not index proximity) is
    where identity actually lives in CA3, so that is the axis this clusters
    on: `pre` group g = pre[g*len(pre)//n_groups : (g+1)*len(pre)//n_groups]
    (matching build_replay_network's own `sup_nc`/`deep_nc` slicing), and each
    post neuron is assigned a home group by creation index modulo n_groups
    (post[i] -> group i % n_groups) -- post has no group of its own (e.g. CA1
    PYR), so this just needs an arbitrary, well-defined, evenly-distributed
    assignment.

    `group_frac` in [0, 1]: expected fraction of a post neuron's `indegree`
    drawn from its home group specifically; the rest is drawn uniformly from
    the REST of `pre` (excluding the home group). 0 reproduces fixed_connect's
    uniform behaviour exactly (up to RNG draw); only meaningful when
    `indegree` is well under len(pre) -- at indegree==len(pre) (today's
    default, 100% dense Schaffer) every pre cell is included regardless of
    this bias, so group_frac has no effect (see --schaffer-k, which must be
    set below len(pre) for this to do anything).

    One nest.Connect call per post neuron (all_to_all), same cost profile as
    clustered_fixed_connect. The per-group "rest of pre" index array is
    precomputed ONCE per group (not per post neuron) to keep this
    O(n_groups x len(pre) + n_post x indegree) rather than O(n_post x len(pre)),
    which would be infeasible at CA1's population sizes (55,195 at 12% scale).
    """
    if w_cv is None:
        w_cv = _HET["w_cv"]
    if compensate and isinstance(weight, (int, float)) and weight > 0:
        weight = wcomp_w(weight)
    weight_param = (jittered_weight(weight, w_cv)
                    if isinstance(weight, (int, float)) else weight)

    pre_nc = _to_nc(pre)
    post_nc = _to_nc(post)
    pre_ids = np.array(pre_nc.tolist())
    post_ids = np.array(post_nc.tolist())
    n_pre, n_post = len(pre_ids), len(post_ids)
    group_size = n_pre // n_groups
    rng = np.random.default_rng(seed + 151)

    # Precompute each group's own index range and its complement, once.
    home_range = []
    other_idx = []
    all_idx = np.arange(n_pre)
    for g in range(n_groups):
        g0 = g * group_size
        g1 = (g + 1) * group_size if g < n_groups - 1 else n_pre
        home_range.append((g0, g1))
        mask = np.ones(n_pre, dtype=bool)
        mask[g0:g1] = False
        other_idx.append(all_idx[mask])

    post_home_group = np.arange(n_post) % n_groups
    k_home_target = int(round(indegree * group_frac))

    for i in range(n_post):
        g0, g1 = home_range[post_home_group[i]]
        home_idx = all_idx[g0:g1]
        k_home = min(k_home_target, len(home_idx))
        k_other = min(int(indegree) - k_home, len(other_idx[post_home_group[i]]))
        chosen_home = (rng.choice(home_idx, size=k_home, replace=False)
                       if k_home > 0 else np.empty(0, dtype=int))
        chosen_other = (rng.choice(other_idx[post_home_group[i]], size=k_other, replace=False)
                        if k_other > 0 else np.empty(0, dtype=int))
        chosen = np.concatenate([chosen_home, chosen_other]).astype(int)
        src_nc = nest.NodeCollection(sorted(pre_ids[chosen].tolist()))
        tgt_nc = nest.NodeCollection([int(post_ids[i])])
        nest.Connect(src_nc, tgt_nc, conn_spec="all_to_all",
                     syn_spec={"weight": weight_param, "delay": delay})


def bernoulli_connect(pre, post, prob, weight, delay):
    """
    NEST native pairwise_bernoulli — C++, fully MPI-parallel.
    Used for group-to-group sequence wiring (small groups, variable p).
    """
    if prob <= 0.0:
        return
    nest.Connect(
        _to_nc(pre), _to_nc(post),
        conn_spec={"rule": "pairwise_bernoulli", "p": min(float(prob), 1.0)},
        syn_spec={"weight": float(weight), "delay": float(delay)},
    )


def conn_stats(label, n_pre, n_post, n_conn_expected):
    """
    Lightweight connectivity summary using expected synapse count.
    Does NOT call nest.GetConnections() — that does a serial Python-level
    scan over all synapses and takes hours on large populations.
    Counts are exact for fixed_indegree, approximate for pairwise_bernoulli.
    """
    density = n_conn_expected / (n_pre * n_post) if n_pre * n_post > 0 else 0.0
    print(f"  {label:32s}: ~{n_conn_expected:9,d} conns | density={density:.5f} | "
          f"out~{n_conn_expected/max(n_pre,1):.1f} | in~{n_conn_expected/max(n_post,1):.1f}")


def mean_rate(pop, spk, sim_ms):
    ev = nest.GetStatus(spk, "events")[0]
    return len(ev["senders"]) / (len(pop) * (sim_ms / 1000.0))


# ============================================================================
# sequence_connect_ca3_layered  (Watson et al. 2025 UPDATE-4)
# All group-to-group steps use NEST pairwise_bernoulli (C++, MPI-parallel).
# ============================================================================

def sequence_connect_ca3_layered(
    ca3_sup, ca3_deep, n_groups,
    p_sup_fwd,     w_sup_fwd,
    p_sup_bwd,     w_sup_bwd,
    p_sup_local,   w_sup_local,
    p_sup_to_deep, w_sup_to_deep,
    p_deep_local,  w_deep_local,
    p_deep_fwd,    w_deep_fwd,
    p_deep_to_sup, w_deep_to_sup,
    delay,
):
    """
    Wire CA3 SUP and DEEP with group-level connectivity using
    NEST-native pairwise_bernoulli (C++, MPI-parallel).

    Watson 2025 asymmetry: S->S ~3.64%  S->D ~3.03%  D->D ~2.25%  D->S ~0.18%
    Returns (sup_groups, deep_groups) as list[list[int]].
    """
    sup_ids  = list(ca3_sup.tolist())
    deep_ids = list(ca3_deep.tolist())
    n_sup    = len(sup_ids);  n_deep = len(deep_ids)

    assert n_sup  % n_groups == 0, f"N_ca3_sup ({n_sup}) not divisible by n_groups ({n_groups})"
    assert n_deep % n_groups == 0, f"N_ca3_deep ({n_deep}) not divisible by n_groups ({n_groups})"

    gs_sup  = n_sup  // n_groups
    gs_deep = n_deep // n_groups

    # Build NodeCollection slices once upfront
    sup_nc  = [nest.NodeCollection(sup_ids [k*gs_sup  : (k+1)*gs_sup ])
               for k in range(n_groups)]
    deep_nc = [nest.NodeCollection(deep_ids[k*gs_deep : (k+1)*gs_deep])
               for k in range(n_groups)]

    for k in range(n_groups):
        s = sup_nc[k];  d = deep_nc[k]

        # SUP local recurrence (Watson S->S ~3.64%)
        bernoulli_connect(s, s, p_sup_local, w_sup_local, delay)

        # SUP sequence chain (forward and backward for bidirectional replay)
        if k + 1 < n_groups:
            bernoulli_connect(s, sup_nc[k+1], p_sup_fwd, w_sup_fwd, delay)
        if k - 1 >= 0:
            bernoulli_connect(s, sup_nc[k-1], p_sup_bwd, w_sup_bwd, delay)

        # SUP->DEEP unidirectional (Watson S->D ~3.03%, NO D->S return)
        bernoulli_connect(s, d, p_sup_to_deep, w_sup_to_deep, delay)

        # DEEP local recurrence (Watson D->D ~2.25%)
        bernoulli_connect(d, d, p_deep_local, w_deep_local, delay)
        if k + 1 < n_groups:
            bernoulli_connect(d, deep_nc[k+1], p_deep_fwd, w_deep_fwd, delay)

        # DEEP->SUP near-absent (Watson D->S ~0.18% — critical asymmetry)
        bernoulli_connect(d, s, p_deep_to_sup, w_deep_to_sup, delay)

    # Return as plain int lists for triggers, stats, and plotting
    return ([list(g.tolist()) for g in sup_nc],
            [list(g.tolist()) for g in deep_nc])


# ============================================================================
# Replay trigger / scaffold
# ============================================================================

def make_replay_trigger(group_ids, trigger_start_ms, trigger_dur_ms=16.0,
                        trigger_rate=2600.0, weight=0.95, delay=1.0):
    ids  = [int(i) for i in group_ids]
    gens = nest.Create("poisson_generator", len(ids), params={
        "rate": float(trigger_rate),
        "start": float(trigger_start_ms),
        "stop":  float(trigger_start_ms + trigger_dur_ms),
    })
    nest.Connect(gens, nest.NodeCollection(ids), conn_spec="one_to_one",
                 syn_spec={"weight": float(weight), "delay": float(delay)})
    return gens


def make_staggered_replay_drive(seq_groups, swr_start_ms, direction="forward",
                                inter_step_ms=8.0, drive_dur_ms=10.0,
                                drive_rate=750.0, weight=0.55, delay=1.0):
    n     = len(seq_groups)
    order = list(range(n)) if direction == "forward" else list(range(n-1, -1, -1))
    all_gens = []
    for step, k in enumerate(order):
        # Round to 0.1 ms resolution — NEST BadProperty if not a multiple
        t0  = round(swr_start_ms + step * inter_step_ms, 1)
        ids = [int(i) for i in seq_groups[k]]
        gens = nest.Create("poisson_generator", len(ids), params={
            "rate": float(drive_rate), "start": float(t0),
            "stop": float(t0 + drive_dur_ms),
        })
        nest.Connect(gens, nest.NodeCollection(ids), conn_spec="one_to_one",
                     syn_spec={"weight": float(weight), "delay": float(delay)})
        all_gens.append(gens)
    return all_gens


# ---- Phase 7: encoding-direction pattern drive (place fields on a ring) ----
# The trigger/scaffold above inject pattern identity straight into CA3 SUP --
# the replay/consolidation direction (CA3 -> CA1 -> EC -> cortex), not the
# encoding direction (cortex -> EC -> DG -> CA3). These three functions give
# EC LII a pattern-specific drive instead, so CA3 only ever learns which
# pattern is active via the existing EC LII -> DG perforant path -> mossy
# fibre route, making CA3's (and DG's) assembly identity a CONSEQUENCE of
# DG's own separation ability rather than an externally-given label.
#
# Patterns are place-field-like tuning curves rather than disjoint groups:
# each EC LII cell gets a fixed random preferred location on a ring, and each
# pattern is a location on that same ring. Two patterns close together on the
# ring recruit overlapping cell sets (nearby fields respond to both); two far
# apart recruit disjoint sets. `sigma` (field width relative to the ring)
# is the tunable overlap knob -- this is the "overlapping but not fully
# random" input the DG selectivity question actually needs to be tested
# against (Leutgeb et al.-style pattern separation), rather than the fully
# disjoint CA3-group split used by the replay-direction trigger above.

def assign_place_fields(n_cells, rng):
    """Fixed random preferred location per cell on a ring of circumference 1."""
    return rng.uniform(0.0, 1.0, n_cells)


def place_field_gain(field_centers, pattern_center, sigma):
    """Per-cell tuning-curve gain in [0, 1] for one pattern location.

    Circular distance on the ring (circumference 1), Gaussian tuning curve.
    sigma is in the same ring units (e.g. 0.15 -> a field influences roughly
    the nearest 30-40% of the ring).
    """
    d = np.abs(field_centers - pattern_center)
    d = np.minimum(d, 1.0 - d)  # wrap around the ring
    return np.exp(-(d ** 2) / (2.0 * sigma ** 2))


def make_place_field_drive(population, field_centers, pattern_center, sigma,
                            t0_ms, dur_ms, base_rate, peak_rate,
                            weight, delay=1.0):
    """Per-cell poisson_generator batch, rate = base_rate + gain*peak_rate.

    Same mechanics as make_replay_trigger/make_staggered_replay_drive (fixed
    start/stop generators, one_to_one connect) -- just parameterized by the
    tuning-curve gain per cell instead of by group membership. base_rate/
    peak_rate need the same empirical bracketing every other DG/EC parameter
    in this model has needed; there is no principled a-priori value.
    """
    gain  = place_field_gain(field_centers, pattern_center, sigma)
    rates = base_rate + gain * peak_rate
    ids   = population.tolist() if hasattr(population, "tolist") else list(population)
    gens  = nest.Create("poisson_generator", len(ids), params={
        "start": float(t0_ms), "stop": float(t0_ms + dur_ms),
    })
    nest.SetStatus(gens, "rate", rates.tolist())
    nest.Connect(gens, nest.NodeCollection([int(i) for i in ids]),
                 conn_spec="one_to_one",
                 syn_spec={"weight": float(weight), "delay": float(delay)})
    return gens


# ============================================================================
# Network builder
# ============================================================================

def build_replay_network(
    # Population sizes
    N_ca3_sup=2_640, N_ca3_deep=660,
    N_ca3_int_sup=240, N_ca3_int_deep=80,
    N_ca1_pyr=4_600, N_ca1_basket=140, N_ca1_olm=90,
    # Sequence
    n_seq_groups=20,
    # SWR windows [ms]
    swr_fwd_start=300.0, swr_fwd_stop=420.0,
    swr_rev_start=600.0, swr_rev_stop=720.0,
    # SWR generator params
    swr_sharpwave_rate=280.0, swr_ripple_hz=180.0,
    swr_ripple_mean=1100.0,   swr_ripple_amp=850.0,
    # Replay trigger
    trigger_dur_ms=16.0, trigger_rate=2600.0, trigger_weight=0.95,
    # Staggered scaffold
    # scaffold_step_ms=None → auto: fit n_seq_groups steps inside 90% of SWR window
    # With 35 groups and 120 ms window: step = 120*0.90/35 ≈ 3.1 ms
    # Previous default of 8 ms gave 35×8=280 ms >> 120 ms → groups 15-34
    # never received scaffold, making the second half of each replay fail.
    scaffold_on=True, scaffold_step_ms=None,
    # trigger_on gates the two SWR replay triggers on group0/group[-1]. The
    # pattern-completion protocol turns them off (and the scaffold) so a partial
    # cue is the only stimulus, isolating recurrent completion from driven replay.
    trigger_on=True,
    # d_seq: axonal delay for the SEQUENCE CHAIN only (ms).
    # Must be > scaffold_step (≈2.9ms) to ensure the scaffold fires each
    # group BEFORE the recurrent cascade from the previous group arrives.
    # d_fast (1.5ms) is still used for E↔I, Schaffer, and CA1 local.
    d_seq=4.0,
    # Scaffold now DOMINANT: each group fires from scaffold alone.
    # 2000 Hz × 4 ms × 2.50 wt ≈ 20 mV net drive → suprathreshold
    # without needing cascade from the previous group.
    scaffold_rate=2000.0, scaffold_weight=2.50,
    scaffold_dur_ms=4.0,    # ms per group (was 10; shorter = sharper)
    # Background drive rates [Hz]
    # Watson UPDATE-2: SUP receives ~3.4x stronger DG/EC input than DEEP
    rate_ec_ca1_pyr=40.0,          # reduced 70→40 Hz: with Schaffer 8× reduced,
                                    # EC background can also be lower.  CA1 target: ~5 Hz.
                                    # At 200 Hz CA1 fired at 39.7 Hz (4× bio target),
                                    # causing every EC neuron to fire in every SWR
                                    # window → non-selective PRP → mass L-LTP at event 3.
    rate_dg_ca3_sup=820.0,  rate_dg_ca3_deep=220.0,
    rate_ec_ca3_sup=530.0,  rate_ec_ca3_deep=150.0,
    rate_ca3_drive_sup=400.0, rate_ca3_drive_deep=120.0,
    # CA3 tonic excitability, applied ONLY when suppress_dg_drive=True (real DG).
    # The Poisson DG proxy (rate_dg_ca3_sup=820 @ 3.0) had bundled in CA3's tonic
    # drive; removing it dropped CA3_SUP to 2.8 Hz, which collapsed CA1 below its
    # E/I threshold (0.2 Hz) and left EC/STC consolidation completely inert at
    # 12% (res/2026-08-02 Job C: 0 EC fired, 0 L-LTP). Biologically this tonic
    # floor is the EC LII direct perforant path + recurrent/neuromodulatory
    # background, NOT the mossy fibre (a sparse detonator). Heterogeneous rate,
    # like the old proxy, so CA3 firing stays structured (a flat drive
    # over-synchronises CA3 and starves CA1). At 600 Hz @ 3.0 (1% scale):
    # CA3_SUP ~7.5 Hz, CA1_PYR ~3.4 Hz, and consolidation runs again (EC fires,
    # L-LTP forms, weights grow). DEEP already sits ~8 Hz, so it gets none.
    # NOTE (1% caveat): forward replay + DG separation PASS at every tonic
    # tried; reverse replay is weak at 1% (10 groups, noise-dominated), but was
    # robust at 12% (rho_rev -0.54) -- the 12% Job C re-run is the acceptance
    # test for reverse replay AND consolidation together.
    ca3_tonic_rate_sup=600.0, ca3_tonic_rate_deep=0.0, ca3_tonic_weight=3.0,
    rate_drive_ca1_basket=820.0,
    rate_drive_ca3_int_sup=820.0, rate_drive_ca3_int_deep=820.0,
    # Theta
    theta_on=True, theta_hz=8.0, theta_mean=1100.0, theta_amp=1000.0,
    # Synaptic weights (mV; fixed across scales — synapse count scales instead)
    # w_seq_bwd raised 0.30→0.70: the backward chain must self-sustain reverse
    # replay until STDP asymmetry takes over.  w_fwd/w_bwd ≈ 2.1 still gives
    # a clear forward bias; STDP will sharpen it epoch-by-epoch (Frey & Morris).
    # w_seq_fwd: 1.50→0.60 — feedforward drive must be sub-threshold alone.
    # Combined with scaffold (now dominant), cascade cannot pre-empt it.
    # Ratio fwd/bwd = 0.60/0.50 = 1.2: barely forward-biased; STDP will
    # asymmetrically strengthen forward traces over epochs (Frey & Morris).
    # d_seq: separate delay for the sequence chain (MUST exceed scaffold_step
    # ≈2.9 ms so scaffold beats the cascade to each group).
    w_seq_fwd=0.60,  w_seq_bwd=0.50,  w_sup_local=0.90,
    # CA3 INT->SUP feedback-inhibition weight. Exposed (was hardcoded -2.0) so
    # the pattern-completion probe can rebalance E/I for its isolated build
    # without touching replay runs, which keep the -2.0 default.
    w_ca3_ie_sup=-2.0,
    w_sup_to_deep=1.30, w_deep_local=0.85, w_deep_fwd=0.70, w_deep_to_sup=0.20,
    # Schaffer collateral weights (mV per synapse)
    # Bio: single Schaffer EPSP ~0.1-0.3 mV (Andersen 2007).
    # Previous values (1.8/2.2) were 9-11× too high → CA1 fired at 296 Hz.
    # Reduced by 6× to bring CA1 into the 5-20 Hz biological range.
    # Schaffer weights: 8× reduction from 0.30→0.04 (biological: single
    # Schaffer EPSP ~0.1-0.3 mV; with K=3000 convergence the total drive
    # at 0.30 mV gave CA1 40 Hz — 8× bio target. At 0.04 mV:
    # 3000×8.4×0.04 + 1000×13.8×0.05 = 1,008+690 = 1,698 mV/s → ~5 Hz
    w_schaffer_sup_pyr=0.04,    w_schaffer_deep_pyr=0.05,
    w_schaffer_sup_basket=0.15, w_schaffer_deep_basket=0.18,
    # CA1 basket gets stronger Schaffer drive (0.15 vs 0.04) so fast-spiking
    # interneurons remain faster than pyramidal cells (bio: FS cells have
    # higher rheobase but still dominate at 30-80 Hz vs PYR 2-5 Hz in sleep)
    w_ca1_ie=-6.0,    # basket→PYR increased -3.5→-6.0: restore E/I balance
    w_ca1_ee=0.5,     # PYR→PYR (unchanged)
    w_ca1_ei=0.5,     # PYR→basket (unchanged)
    w_ca1_oe=-1.5,    # OLM→PYR (unchanged)
    # Parallel
    n_threads=8,
    seed_connect=42,  # RNG seed for V_m heterogeneity + connectivity
    # NEST kernel RNG (Poisson drive, connectivity draws, delay jitter).
    # Separate from seed_connect, which drives the numpy side (V_m spread,
    # DG wiring). Replication must vary BOTH or runs differ only partially.
    master_seed=20260111,
    # Multi-pattern replay. n_patterns > 1 partitions the CA3 sequence groups
    # into disjoint interleaved assemblies, one replayed per epoch, so that
    # downstream selectivity ("engram for A but not B") becomes measurable.
    # n_epochs/epoch_ms are needed here because SWR generators and scaffolds
    # carry ABSOLUTE times and must be created for every epoch.
    n_patterns=1, n_epochs=1, epoch_ms=1000.0,
    # Restrict replay to ONE pattern index while keeping the n_patterns
    # partition. Needed to ask whether the cortical trace is pattern-
    # SPECIFIC: train A-only and B-only on the SAME network (same seed, so
    # the same synapses exist) and compare the resulting weight changes.
    train_pattern=None,
    # Phase 7: where the pattern-defining external drive attaches.
    # "ca3" (default) is the original Watson-replay direction: the trigger/
    # scaffold below inject pattern identity straight into CA3 SUP, modelling
    # an SWR reactivating an already-encoded memory. "ec-lii" is the encoding
    # direction (sensory/cortex -> EC -> DG -> CA3): CA3 gets NO direct
    # injection at all here, and the pattern-specific place-field drive is
    # wired onto EC LII instead, in main(), after build_ec_lii() runs (EC LII
    # does not exist yet at this point in the build). patterns/epoch_pattern
    # are still computed and returned in both modes so that drive can reuse
    # the same epoch-to-pattern-index sequence.
    pattern_source="ca3",
    # Phase C: per-synapse delay jitter (ms) on the FEEDFORWARD readout
    # projections (Schaffer, and the cortical hops in the modules). 0 = the
    # original single-scalar delays. See jittered_delay().
    delay_jitter=0.0,
    # Weight compensation for the jittered projections. Spreading arrival
    # times reduces coincident summation, so jitter alone LOWERS downstream
    # rates (measured: EC LII -51%, mPFC -61%) and that drop confounds any
    # discrimination comparison. Scaling the jittered weights restores drive
    # so the two conditions can be compared at matched firing rates.
    delay_jitter_wcomp=1.0,
    # Schaffer in-degree override. The default (3000, clamped to the CA3 SUP
    # size) makes every CA1 cell sample essentially ALL of CA3 at small
    # scales -- density 1.0 at 1% -- so every CA1 cell sees the same input
    # and 12.1M synapses make a Python-side plasticity hook impractical.
    # Weights are scaled by K_default/K so mean CA1 drive is preserved.
    schaffer_k=None,
    # Static multiplier on the Schaffer weights ONLY. This is the control for
    # the Schaffer-STDP experiment: STDP raised cortical rates 28-68%, so a
    # static run at the same elevated Schaffer drive isolates 'did the
    # SELECTION of delay-matched paths matter' from 'did the drive go up'.
    # Using delay_jitter_wcomp for this would be wrong -- that also scales
    # CA1->EC and EC->mPFC, which STDP never touched.
    schaffer_w_scale=1.0,
    # Phase 12: >0 biases each CA1 PYR cell's (reduced, via schaffer_k) K
    # Schaffer sources toward ONE CA3 sequence group instead of sampling
    # uniformly -- see grouped_fixed_connect's docstring for why group
    # membership, not neuron index, is the right clustering axis here.
    # Only meaningful together with schaffer_k < the population size.
    schaffer_group_frac=0.0,
    # Phase 6.2: when a real DG circuit (--dg) drives CA3 via mossy fibres,
    # the Poisson DG proxy on CA3 SUP/DEEP is suppressed so drive is not
    # double-counted. The EC and background CA3 drives are kept -- they model
    # the direct perforant path and tonic background, distinct from DG input.
    suppress_dg_drive=False,
):
    """
    Build Watson 2025 two-layer CA1+CA3 replay network.

    v2: All connections use NEST-native fixed_indegree or pairwise_bernoulli
    (C++, MPI-parallel). The Python bernoulli_connect() serial loop is gone.
    """
    t0 = time.perf_counter()

    nest.ResetKernel()
    nest.SetKernelStatus({
        "resolution":        0.1,
        "local_num_threads": n_threads,
        "print_time":        True,
        "overwrite_files":   True,
    })

    # --- VP verification (critical: NEST may silently ignore local_num_threads) ---
    ks_post = nest.GetKernelStatus()
    actual_thr = ks_post.get("local_num_threads",
                  ks_post.get("num_threads", ks_post.get("threads", 1)))
    actual_mpi = ks_post.get("total_num_processes",
                  ks_post.get("num_processes",
                  ks_post.get("mpi_num_processes", 1)))
    n_vp_actual = actual_thr * actual_mpi
    print(f"  NEST kernel: {actual_mpi} MPI rank(s) × {actual_thr} thread(s) = {n_vp_actual} VP(s)")
    if actual_thr != n_threads:
        import warnings
        warnings.warn(
            f"[NEST thread mismatch] requested {n_threads} threads but NEST "
            f"accepted {actual_thr}. VPs={n_vp_actual}. "
            f"Connection limit may be hit if VPs < MPI_ranks × requested_threads.",
            RuntimeWarning, stacklevel=2,
        )

    safe_set_seeds(master_seed)

    try:
        available = list(nest.node_models)
    except AttributeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            available = list(nest.Models("nodes"))
    if "izhikevich" not in available:
        raise RuntimeError("NEST model 'izhikevich' not found.")

    # -------------------------------------------------------------------------
    # Izhikevich parameters (Watson et al. 2025, Fig 4H)
    # SUP: regular adapting  (rheobase ~128 pA, Rn ~240 MOhm, Vm -63.2 mV)
    # DEEP: intrinsic burst  (rheobase  ~77 pA, Rn ~261 MOhm, Vm -64.7 mV)
    # -------------------------------------------------------------------------
    ca3_sup_params  = dict(a=0.02, b=0.2,  c=-65.0, d=8.0, V_m=-63.2, U_m=-13.0,  I_e=0.0)
    ca3_deep_params = dict(a=0.02, b=0.2,  c=-55.0, d=4.0, V_m=-64.7, U_m=-13.0,  I_e=3.0)
    ca1_pyr_params  = dict(a=0.02, b=0.2,  c=-65.0, d=8.0, V_m=-65.0, U_m=-13.0,  I_e=0.0)
    basket_params   = dict(a=0.10, b=0.2,  c=-65.0, d=2.0, V_m=-65.0, U_m=-13.0,  I_e=0.0)
    olm_params      = dict(a=0.02, b=0.25, c=-65.0, d=2.0, V_m=-65.0, U_m=-16.25, I_e=0.0)

    # -------------------------------------------------------------------------
    # Populations — with Gaussian V_m heterogeneity
    # Neurons start at heterogeneous membrane potentials (σ=4–5 mV).
    # This creates Gaussian-distributed spike timing within each population:
    # near-threshold neurons fire first, sub-threshold ones fire later.
    # Critical for visible sequential replay and smooth burst envelopes.
    # -------------------------------------------------------------------------
    print("  Creating populations (with V_m heterogeneity)...")
    _rng_vm = np.random.default_rng(seed_connect + 1)  # reproducible

    CA1_PYR      = nest.Create("izhikevich", N_ca1_pyr,      params=het_params(ca1_pyr_params, N_ca1_pyr, _rng_vm))
    # CA1 PYR: σ=5 mV spread → Gaussian burst envelope (bio: ~15 mV cell-to-cell)
    nest.SetStatus(CA1_PYR, "V_m",
                   _rng_vm.normal(-65.0, 5.0, N_ca1_pyr).clip(-75,-55).tolist())

    CA1_BASKET   = nest.Create("izhikevich", N_ca1_basket,   params=het_params(basket_params, N_ca1_basket, _rng_vm))
    CA1_OLM      = nest.Create("izhikevich", N_ca1_olm,      params=het_params(olm_params, N_ca1_olm, _rng_vm))

    CA3_SUP      = nest.Create("izhikevich", N_ca3_sup,      params=het_params(ca3_sup_params, N_ca3_sup, _rng_vm))
    # CA3 SUP: σ=4 mV → spread within each group → smooth diagonal heatmap
    nest.SetStatus(CA3_SUP, "V_m",
                   _rng_vm.normal(-63.2, 4.0, N_ca3_sup).clip(-72,-54).tolist())

    CA3_DEEP     = nest.Create("izhikevich", N_ca3_deep,     params=het_params(ca3_deep_params, N_ca3_deep, _rng_vm))
    nest.SetStatus(CA3_DEEP, "V_m",
                   _rng_vm.normal(-64.7, 4.0, N_ca3_deep).clip(-72,-55).tolist())

    CA3_INT_SUP  = nest.Create("izhikevich", N_ca3_int_sup,  params=het_params(basket_params, N_ca3_int_sup, _rng_vm))
    CA3_INT_DEEP = nest.Create("izhikevich", N_ca3_int_deep, params=het_params(basket_params, N_ca3_int_deep, _rng_vm))

    # -------------------------------------------------------------------------
    # Background inputs (one_to_one Poisson — cheap, no bottleneck)
    # UPDATE-2: SUP receives ~3.4x stronger DG/EC drive than DEEP
    # -------------------------------------------------------------------------
    print("  Connecting background inputs...")
    d_fast = 1.5;  d_slow = 3.0

    def _drive(n, rate, target, weight, delay, compensate=True):
        gen = nest.Create("poisson_generator", n, params={"rate": float(rate)})
        nest.Connect(gen, target, conn_spec="one_to_one",
                     syn_spec={"weight": (wcomp_w(weight) if compensate
                                          else float(weight)),
                               "delay": float(delay)})

    _drive(N_ca1_pyr,      rate_ec_ca1_pyr,         CA1_PYR,      2.0, d_slow)
    _drive(N_ca1_basket,   rate_drive_ca1_basket,   CA1_BASKET,   2.0, d_fast, compensate=False)
    if suppress_dg_drive:
        # A real DG circuit will drive CA3 via mossy fibres (build_dg_module);
        # skip the Poisson proxy so the mossy-fibre input is not double-counted.
        # BUT restore CA3's tonic excitability (which the proxy had bundled in)
        # through a generic tonic drive standing for the EC LII direct perforant
        # path + recurrent/neuromodulatory background — otherwise CA3_SUP falls
        # to ~2.8 Hz, CA1 collapses below its E/I threshold, and EC/STC
        # consolidation goes inert (observed at 12%, res/2026-08-02 Job C).
        print("    [DG] Poisson DG->CA3 proxy suppressed; real DG mossy fibres "
              "(detonator) + heterogeneous CA3 tonic drive replace it")
        # Heterogeneous tonic (rates spread like the old proxy) -- a flat drive
        # over-synchronises CA3, which both under-drives CA1 and kills the weak
        # reverse chain. The spread reproduces the proxy's structured firing that
        # gave CA3~5 -> CA1~5 and bidirectional replay; it is now attributed to
        # EC-direct/recurrent excitability, with the mossy fibre as a separate
        # sparse detonator on top.
        if ca3_tonic_rate_sup > 0:
            _tr = _rng_vm.normal(ca3_tonic_rate_sup, 100.0,
                                 N_ca3_sup).clip(400.0, 1400.0)
            _tg = nest.Create("poisson_generator", N_ca3_sup)
            nest.SetStatus(_tg, "rate", _tr.tolist())
            nest.Connect(_tg, CA3_SUP, conn_spec="one_to_one",
                         syn_spec={"weight": wcomp_w(ca3_tonic_weight), "delay": d_fast})
        if ca3_tonic_rate_deep > 0:
            _drive(N_ca3_deep, ca3_tonic_rate_deep, CA3_DEEP, ca3_tonic_weight, d_fast)
    else:
        # DG→CA3 SUP: heterogeneous rates (σ=100 Hz) to spread membrane potentials
        _dg_rates_sup = _rng_vm.normal(rate_dg_ca3_sup, 100.0,
                                        N_ca3_sup).clip(400.0, 1400.0)
        _dg_gen_sup   = nest.Create("poisson_generator", N_ca3_sup)
        nest.SetStatus(_dg_gen_sup, "rate", _dg_rates_sup.tolist())
        nest.Connect(_dg_gen_sup, CA3_SUP, conn_spec="one_to_one",
                     syn_spec={"weight": 3.0, "delay": d_fast})
        # DG→CA3 DEEP: uniform (smaller pop, less impact on replay clarity)
        _drive(N_ca3_deep,     rate_dg_ca3_deep,        CA3_DEEP,     1.0, d_fast)
    _drive(N_ca3_sup,      rate_ec_ca3_sup,         CA3_SUP,      2.0, d_slow)
    _drive(N_ca3_deep,     rate_ec_ca3_deep,        CA3_DEEP,     1.2, d_slow)
    _drive(N_ca3_sup,      rate_ca3_drive_sup,      CA3_SUP,      2.0, d_fast)
    _drive(N_ca3_deep,     rate_ca3_drive_deep,     CA3_DEEP,     1.5, d_fast)
    _drive(N_ca3_int_sup,  rate_drive_ca3_int_sup,  CA3_INT_SUP,  2.0, d_fast, compensate=False)
    _drive(N_ca3_int_deep, rate_drive_ca3_int_deep, CA3_INT_DEEP, 2.0, d_fast, compensate=False)

    # -------------------------------------------------------------------------
    # Theta drive
    # -------------------------------------------------------------------------
    if theta_on:
        print("  Connecting theta drive...")
        for pop, w_th in [
            (CA3_SUP, 0.80), (CA3_DEEP, 0.60),
            (CA3_INT_SUP, 1.80), (CA3_INT_DEEP, 1.80),
            (CA1_PYR, 1.00), (CA1_BASKET, 2.00), (CA1_OLM, 2.00),
        ]:
            th = maybe_make_theta_generators(len(pop), theta_mean, theta_amp, theta_hz)
            if th is not None:
                nest.Connect(th, pop, conn_spec="one_to_one",
                             syn_spec={"weight": float(w_th), "delay": 1.0})

    # -------------------------------------------------------------------------
    # SWR event generators
    # -------------------------------------------------------------------------
    print("  Connecting SWR generators...")
    # NOTE: the sharp-wave/ripple BACKGROUND is created for epoch 0 only.
    # Repeating it per epoch was tried and is prohibitively expensive: each
    # window needs ~16k sinusoidal_poisson_generators (one per neuron across
    # CA3+CA1), and NEST steps every generator at every one of the ~60k
    # timesteps whether or not it is inside its start/stop window. At 6 epochs
    # that is ~195k generators / ~1e10 node updates — a 1% run that normally
    # takes ~7 min had not finished after 3 hours.
    # The per-epoch REPLAY (trigger + staggered scaffold, below) is ~10x
    # cheaper and is what carries pattern identity, so that is repeated per
    # epoch instead. Making the ripple background per-epoch cheaply would need
    # inhomogeneous_poisson_generator (one node, scheduled rate profile).
    for swr_s, swr_e in [(swr_fwd_start, swr_fwd_stop),
                         (swr_rev_start, swr_rev_stop)]:
        sw_sh_sup, sw_rip_sup = make_swr_event_generators(
            n=N_ca3_sup, start_ms=swr_s, stop_ms=swr_e,
            sharpwave_rate=swr_sharpwave_rate,
            ripple_rate_mean=swr_ripple_mean, ripple_rate_amp=swr_ripple_amp,
            ripple_hz=swr_ripple_hz)
        nest.Connect(sw_sh_sup,  CA3_SUP, conn_spec="one_to_one", syn_spec={"weight": wcomp_w(0.35), "delay": 1.0})
        nest.Connect(sw_rip_sup, CA3_SUP, conn_spec="one_to_one", syn_spec={"weight": wcomp_w(0.15), "delay": 1.0})

        sw_sh_deep, sw_rip_deep = make_swr_event_generators(
            n=N_ca3_deep, start_ms=swr_s, stop_ms=swr_e,
            sharpwave_rate=swr_sharpwave_rate * 0.35,
            ripple_rate_mean=swr_ripple_mean * 0.35,
            ripple_rate_amp=swr_ripple_amp   * 0.35,
            ripple_hz=swr_ripple_hz)
        nest.Connect(sw_sh_deep,  CA3_DEEP, conn_spec="one_to_one", syn_spec={"weight": wcomp_w(0.35), "delay": 1.0})
        nest.Connect(sw_rip_deep, CA3_DEEP, conn_spec="one_to_one", syn_spec={"weight": wcomp_w(0.15), "delay": 1.0})

        _, sw_rip_int_sup = make_swr_event_generators(
            n=N_ca3_int_sup, start_ms=swr_s, stop_ms=swr_e,
            sharpwave_rate=0.0, ripple_rate_mean=swr_ripple_mean,
            ripple_rate_amp=swr_ripple_amp, ripple_hz=swr_ripple_hz)
        nest.Connect(sw_rip_int_sup, CA3_INT_SUP, conn_spec="one_to_one",
                     syn_spec={"weight": 0.60, "delay": 1.0})

        _, sw_rip_int_deep = make_swr_event_generators(
            n=N_ca3_int_deep, start_ms=swr_s, stop_ms=swr_e,
            sharpwave_rate=0.0, ripple_rate_mean=swr_ripple_mean,
            ripple_rate_amp=swr_ripple_amp, ripple_hz=swr_ripple_hz)
        nest.Connect(sw_rip_int_deep, CA3_INT_DEEP, conn_spec="one_to_one",
                     syn_spec={"weight": 0.60, "delay": 1.0})

        sw_sh_c1, sw_rip_c1 = make_swr_event_generators(
            n=N_ca1_pyr, start_ms=swr_s, stop_ms=swr_e,
            sharpwave_rate=swr_sharpwave_rate * 0.6,
            ripple_rate_mean=swr_ripple_mean * 0.6,
            ripple_rate_amp=swr_ripple_amp   * 0.6, ripple_hz=swr_ripple_hz)
        nest.Connect(sw_sh_c1,  CA1_PYR, conn_spec="one_to_one", syn_spec={"weight": wcomp_w(0.20), "delay": 1.0})
        nest.Connect(sw_rip_c1, CA1_PYR, conn_spec="one_to_one", syn_spec={"weight": wcomp_w(0.10), "delay": 1.0})

        _, sw_rip_c1b = make_swr_event_generators(
            n=N_ca1_basket, start_ms=swr_s, stop_ms=swr_e,
            sharpwave_rate=0.0,
            ripple_rate_mean=swr_ripple_mean * 0.8,
            ripple_rate_amp=swr_ripple_amp   * 0.8, ripple_hz=swr_ripple_hz)
        nest.Connect(sw_rip_c1b, CA1_BASKET, conn_spec="one_to_one",
                     syn_spec={"weight": 0.55, "delay": 1.0})

    # =========================================================================
    # RECURRENT CONNECTIVITY — all NEST-native C++ rules (MPI-parallel)
    # =========================================================================

    gs_sup  = N_ca3_sup  // n_seq_groups
    gs_deep = N_ca3_deep // n_seq_groups

    # ---- CA3 sequence chain (pairwise_bernoulli) ----------------------------
    print("  Wiring CA3 sequence chain (pairwise_bernoulli)...")
    t_seq = time.perf_counter()

    p_seq_fwd     = p("seq_fwd",      gs_sup)
    p_seq_bwd     = p("seq_bwd",      gs_sup)
    p_sup_local   = p("sup_local",    gs_sup)
    p_sup_to_deep = p("sup_to_deep",  gs_sup)
    p_deep_local  = p("deep_local",   gs_deep)
    p_deep_fwd    = p("deep_fwd",     gs_deep)
    p_deep_to_sup = p("deep_to_sup",  gs_deep)

    print(f"    gs_sup={gs_sup}  gs_deep={gs_deep}")
    print(f"    p_fwd={p_seq_fwd:.4f}  p_bwd={p_seq_bwd:.4f}  "
          f"p_sup_local={p_sup_local:.4f}  p_S->D={p_sup_to_deep:.4f}  "
          f"p_D->S={p_deep_to_sup:.5f}")

    ca3_sup_groups, ca3_deep_groups = sequence_connect_ca3_layered(
        CA3_SUP, CA3_DEEP, n_seq_groups,
        p_sup_fwd=p_seq_fwd,      w_sup_fwd=w_seq_fwd,
        p_sup_bwd=p_seq_bwd,      w_sup_bwd=w_seq_bwd,
        p_sup_local=p_sup_local,  w_sup_local=w_sup_local,
        p_sup_to_deep=p_sup_to_deep, w_sup_to_deep=w_sup_to_deep,
        p_deep_local=p_deep_local,   w_deep_local=w_deep_local,
        p_deep_fwd=p_deep_fwd,       w_deep_fwd=w_deep_fwd,
        p_deep_to_sup=p_deep_to_sup, w_deep_to_sup=w_deep_to_sup,
        delay=d_seq,   # sequence chain uses longer delay to prevent avalanche
    )
    print(f"    done in {time.perf_counter()-t_seq:.1f}s")

    # ---- CA3 E<->I  (fixed_indegree, UPDATE-5) ------------------------------
    print("  Wiring CA3 E<->I (fixed_indegree)...")
    t_ei = time.perf_counter()

    fixed_connect(CA3_SUP,      CA3_INT_SUP,  K("ca3_EI_sup",   N_ca3_sup),      0.5,  d_fast, compensate=False)
    fixed_connect(CA3_DEEP,     CA3_INT_DEEP, K("ca3_EI_deep",  N_ca3_deep),     0.5,  d_fast, compensate=False)
    fixed_connect(CA3_INT_SUP,  CA3_SUP,      K("ca3_IE_sup",   N_ca3_int_sup),  w_ca3_ie_sup, d_fast)
    fixed_connect(CA3_INT_DEEP, CA3_DEEP,     K("ca3_IE_deep",  N_ca3_int_deep), -2.0, d_fast)
    fixed_connect(CA3_INT_SUP,  CA3_DEEP,     K("ca3_IE_cross", N_ca3_int_sup),  -0.2, d_fast)
    fixed_connect(CA3_INT_DEEP, CA3_SUP,      K("ca3_IE_cross", N_ca3_int_deep), -0.2, d_fast)
    fixed_connect(CA3_INT_SUP,  CA3_INT_SUP,  K("ca3_II",       N_ca3_int_sup),  -1.5, d_fast)
    fixed_connect(CA3_INT_DEEP, CA3_INT_DEEP, K("ca3_II",       N_ca3_int_deep), -1.5, d_fast)
    print(f"    done in {time.perf_counter()-t_ei:.1f}s")

    # ---- Schaffer collaterals (fixed_indegree, UPDATE-6) --------------------
    print("  Wiring Schaffer collaterals (fixed_indegree)...")
    t_sch = time.perf_counter()

    _d_sch = jittered_delay(d_slow, delay_jitter)
    _wc = delay_jitter_wcomp if delay_jitter > 0 else 1.0
    _K_sup_def  = K("schaffer_sup_pyr",  N_ca3_sup)
    _K_deep_def = K("schaffer_deep_pyr", N_ca3_deep)
    if schaffer_k is not None:
        _K_sup  = max(1, min(int(schaffer_k), N_ca3_sup))
        _K_deep = max(1, min(int(round(schaffer_k * _K_deep_def / max(_K_sup_def, 1))),
                             N_ca3_deep))
        # preserve mean drive: fewer inputs, proportionally stronger
        _sc_sup  = _K_sup_def  / _K_sup
        _sc_deep = _K_deep_def / _K_deep
        print(f"    Schaffer in-degree override: SUP {_K_sup_def}->{_K_sup} "
              f"(w x{_sc_sup:.1f})  DEEP {_K_deep_def}->{_K_deep} (w x{_sc_deep:.1f})")
    else:
        _K_sup, _K_deep, _sc_sup, _sc_deep = _K_sup_def, _K_deep_def, 1.0, 1.0
    if schaffer_w_scale != 1.0:
        print(f"    Schaffer static weight scale x{schaffer_w_scale:.2f} "
              f"(STDP control)")
    if schaffer_group_frac > 0:
        print(f"    Schaffer group clustering: group_frac={schaffer_group_frac} "
              f"({n_seq_groups} groups, home group by CA1 PYR creation index)")
        grouped_fixed_connect(CA3_SUP, CA1_PYR, _K_sup,
                              w_schaffer_sup_pyr*_wc*_sc_sup*schaffer_w_scale, _d_sch,
                              n_seq_groups, schaffer_group_frac, seed=seed_connect)
        grouped_fixed_connect(CA3_DEEP, CA1_PYR, _K_deep,
                              w_schaffer_deep_pyr*_wc*_sc_deep*schaffer_w_scale, _d_sch,
                              n_seq_groups, schaffer_group_frac, seed=seed_connect)
    else:
        fixed_connect(CA3_SUP,  CA1_PYR, _K_sup,
                      w_schaffer_sup_pyr*_wc*_sc_sup*schaffer_w_scale,   _d_sch)
        fixed_connect(CA3_DEEP, CA1_PYR, _K_deep,
                      w_schaffer_deep_pyr*_wc*_sc_deep*schaffer_w_scale, _d_sch)
    fixed_connect(CA3_SUP,  CA1_BASKET, K("schaffer_sup_basket",  N_ca3_sup),  w_schaffer_sup_basket, d_fast, compensate=False)
    fixed_connect(CA3_DEEP, CA1_BASKET, K("schaffer_deep_basket", N_ca3_deep), w_schaffer_deep_basket,d_fast, compensate=False)
    print(f"    done in {time.perf_counter()-t_sch:.1f}s")

    # ---- CA1 local (fixed_indegree) -----------------------------------------
    print("  Wiring CA1 local (fixed_indegree)...")
    t_ca1 = time.perf_counter()

    fixed_connect(CA1_PYR,    CA1_PYR,    K("ca1_EE", N_ca1_pyr),     w_ca1_ee,  d_slow)
    fixed_connect(CA1_PYR,    CA1_BASKET, K("ca1_EI", N_ca1_pyr),     w_ca1_ei,  d_fast, compensate=False)
    fixed_connect(CA1_BASKET, CA1_PYR,    K("ca1_IE", N_ca1_basket),  w_ca1_ie,  d_fast)
    fixed_connect(CA1_OLM,    CA1_PYR,    K("ca1_OE", N_ca1_olm),     w_ca1_oe,  d_slow)
    print(f"    done in {time.perf_counter()-t_ca1:.1f}s")

    # ---- Replay triggers and scaffold ---------------------------------------
    print("  Connecting replay triggers and scaffold...")

    # ---- Pattern definitions -------------------------------------------------
    # An engram is only meaningful if there is more than one thing to remember:
    # "selective" means selective for A rather than B. With a single stored
    # sequence, uniform potentiation downstream is the CORRECT outcome, and no
    # selectivity metric can say anything. n_patterns partitions the CA3
    # sequence groups into that many disjoint assemblies, each replayed in its
    # own SWR events, so cortical discrimination becomes measurable.
    patterns = [list(range(n_seq_groups))] if n_patterns <= 1 else [
        list(range(i, n_seq_groups, n_patterns)) for i in range(n_patterns)
    ]
    # Interleaved (stride) rather than contiguous blocks so the assemblies are
    # spatially intermingled in CA3 — a contiguous split would let downstream
    # cells discriminate on gross topography rather than on assembly identity.
    gs_per_pattern = min(len(p) for p in patterns)
    if n_patterns > 1:
        print(f"    {n_patterns} patterns x ~{gs_per_pattern} groups each "
              f"(interleaved); each SWR epoch replays one pattern in turn")

    # Auto-compute scaffold step so one pattern's groups fit within the SWR
    # window. We use 85% of the window so the last group still fires 15% before
    # window end, leaving time for Schaffer → CA1 propagation.
    if scaffold_step_ms is None:
        raw_step = (swr_fwd_stop - swr_fwd_start) * 0.85 / max(gs_per_pattern, 1)
        # Round to nearest 0.1 ms: NEST requires start times to be exact
        # multiples of the resolution.  120*0.85/35 = 2.9142... fails.
        scaffold_step_ms = round(raw_step, 1)
    print(f"    scaffold_step_ms={scaffold_step_ms:.1f}  "
          f"(span = {scaffold_step_ms * gs_per_pattern:.1f} ms, "
          f"SWR window = {swr_fwd_stop - swr_fwd_start:.0f} ms)")

    # ---- Per-epoch replay ----------------------------------------------------
    # The replay drive IS created for every epoch (unlike the ripple background
    # above). Previously it carried absolute times from epoch 0 only, so in an
    # n-epoch run epochs 1..n-1 contained no replay at all and the STC hook
    # tagged on background activity. Each epoch now replays
    # patterns[epoch % n_patterns], which is also what lets different patterns
    # occupy different epochs so downstream selectivity can be measured.
    epoch_pattern = []
    for ep in range(max(1, n_epochs)):
        t0_ep = ep * epoch_ms
        pat   = (patterns[ep % len(patterns)] if train_pattern is None
                 else patterns[train_pattern % len(patterns)])
        epoch_pattern.append(ep % len(patterns) if train_pattern is None
                             else train_pattern % len(patterns))
        grp   = [ca3_sup_groups[i] for i in pat]
        f_s, r_s = t0_ep + swr_fwd_start, t0_ep + swr_rev_start
        # pattern_source="ec-lii": CA3 gets no direct pattern-specific
        # injection at all -- the encoding-direction drive is wired onto
        # EC LII instead, in main(), after build_ec_lii() runs.
        if trigger_on and pattern_source == "ca3":
            make_replay_trigger(grp[0],  f_s, trigger_dur_ms=trigger_dur_ms,
                                trigger_rate=trigger_rate, weight=trigger_weight)
            make_replay_trigger(grp[-1], r_s, trigger_dur_ms=trigger_dur_ms,
                                trigger_rate=trigger_rate, weight=trigger_weight)
        if scaffold_on and pattern_source == "ca3":
            make_staggered_replay_drive(
                grp, f_s, direction="forward",
                inter_step_ms=scaffold_step_ms, drive_rate=scaffold_rate,
                drive_dur_ms=scaffold_dur_ms, weight=scaffold_weight)
            make_staggered_replay_drive(
                grp, r_s, direction="reverse",
                inter_step_ms=scaffold_step_ms, drive_rate=scaffold_rate,
                drive_dur_ms=scaffold_dur_ms, weight=scaffold_weight)

    # ---- Recorders ----------------------------------------------------------
    spk_ca1_pyr      = nest.Create("spike_recorder")
    spk_ca1_ba       = nest.Create("spike_recorder")
    spk_ca1_olm      = nest.Create("spike_recorder")
    spk_ca3_sup      = nest.Create("spike_recorder")
    spk_ca3_deep     = nest.Create("spike_recorder")
    spk_ca3_int_sup  = nest.Create("spike_recorder")
    spk_ca3_int_deep = nest.Create("spike_recorder")

    nest.Connect(CA1_PYR,      spk_ca1_pyr)
    nest.Connect(CA1_BASKET,   spk_ca1_ba)
    nest.Connect(CA1_OLM,      spk_ca1_olm)
    nest.Connect(CA3_SUP,      spk_ca3_sup)
    nest.Connect(CA3_DEEP,     spk_ca3_deep)
    nest.Connect(CA3_INT_SUP,  spk_ca3_int_sup)
    nest.Connect(CA3_INT_DEEP, spk_ca3_int_deep)

    # Fixed small sample for Vm — does NOT grow with scale
    try:
        vm = nest.Create("multimeter", params={"record_from": ["V_m", "U_m"], "interval": 0.2})
    except Exception:
        vm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 0.2})
    nest.Connect(vm, CA1_PYR[:5])
    nest.Connect(vm, CA3_SUP[:5])
    nest.Connect(vm, CA3_DEEP[:min(3, N_ca3_deep)])

    print(f"\n  Total build time: {time.perf_counter()-t0:.1f}s")

    # ---- Connectivity stats (lightweight — no GetConnections) ---------------
    print("\n=== Connectivity stats (expected synapse counts) ===")
    gs_sup_n  = N_ca3_sup  // n_seq_groups
    gs_deep_n = N_ca3_deep // n_seq_groups

    # Sequence chain (pairwise_bernoulli — expected counts)
    conn_stats("CA3 SUP->SUP  (S-S)", N_ca3_sup,  N_ca3_sup,
               int(n_seq_groups * gs_sup_n * (p_sup_local * gs_sup_n        # local
               + p_seq_fwd * gs_sup_n + p_seq_bwd * gs_sup_n)))             # fwd+bwd
    conn_stats("CA3 SUP->DEEP (S-D)", N_ca3_sup,  N_ca3_deep,
               int(n_seq_groups * gs_sup_n * p_sup_to_deep * gs_deep_n))
    conn_stats("CA3 DEEP->DEEP(D-D)", N_ca3_deep, N_ca3_deep,
               int(n_seq_groups * gs_deep_n * (p_deep_local * gs_deep_n
               + p_deep_fwd * gs_deep_n)))
    conn_stats("CA3 DEEP->SUP (D-S)", N_ca3_deep, N_ca3_sup,
               int(n_seq_groups * gs_deep_n * p_deep_to_sup * gs_sup_n))
    conn_stats("Seq SUP g0->g1",      gs_sup_n,   gs_sup_n,
               int(p_seq_fwd * gs_sup_n * gs_sup_n))

    # E<->I (fixed_indegree — exact)
    conn_stats("CA3 SUP->INT_SUP",  N_ca3_sup,       N_ca3_int_sup,
               N_ca3_int_sup  * K("ca3_EI_sup",   N_ca3_sup))
    conn_stats("CA3 INT_SUP->SUP",  N_ca3_int_sup,   N_ca3_sup,
               N_ca3_sup      * K("ca3_IE_sup",   N_ca3_int_sup))
    conn_stats("CA3 DEEP->INT_DEEP",N_ca3_deep,      N_ca3_int_deep,
               N_ca3_int_deep * K("ca3_EI_deep",  N_ca3_deep))
    conn_stats("CA3 INT_DEEP->DEEP",N_ca3_int_deep,  N_ca3_deep,
               N_ca3_deep     * K("ca3_IE_deep",  N_ca3_int_deep))

    # Schaffer (fixed_indegree — exact)
    conn_stats("Sch SUP->CA1 PYR",  N_ca3_sup,  N_ca1_pyr,
               N_ca1_pyr    * K("schaffer_sup_pyr",  N_ca3_sup))
    conn_stats("Sch DEEP->CA1 PYR", N_ca3_deep, N_ca1_pyr,
               N_ca1_pyr    * K("schaffer_deep_pyr", N_ca3_deep))
    conn_stats("Sch SUP->CA1 BSK",  N_ca3_sup,  N_ca1_basket,
               N_ca1_basket * K("schaffer_sup_basket",  N_ca3_sup))
    conn_stats("Sch DEEP->CA1 BSK", N_ca3_deep, N_ca1_basket,
               N_ca1_basket * K("schaffer_deep_basket", N_ca3_deep))

    # Total synapse count
    total_synapses = (
        N_ca1_pyr    * K("schaffer_sup_pyr",   N_ca3_sup)
        + N_ca1_pyr  * K("schaffer_deep_pyr",  N_ca3_deep)
        + N_ca3_sup  * K("ca3_IE_sup",         N_ca3_int_sup)
        + N_ca3_deep * K("ca3_IE_deep",        N_ca3_int_deep)
        + N_ca3_int_sup  * K("ca3_EI_sup",     N_ca3_sup)
        + N_ca3_int_deep * K("ca3_EI_deep",    N_ca3_deep)
    )
    print(f"  {'Total synapses (approx)':32s}: ~{total_synapses:,d}")

    return dict(
        PYR=CA1_PYR, BASKET=CA1_BASKET, OLM=CA1_OLM,
        spk_pyr=spk_ca1_pyr, spk_ba=spk_ca1_ba, spk_olm=spk_ca1_olm,
        CA3_SUP=CA3_SUP, CA3_DEEP=CA3_DEEP,
        CA3_INT_SUP=CA3_INT_SUP, CA3_INT_DEEP=CA3_INT_DEEP,
        spk_ca3_sup=spk_ca3_sup, spk_ca3_deep=spk_ca3_deep,
        spk_ca3_int_sup=spk_ca3_int_sup, spk_ca3_int_deep=spk_ca3_int_deep,
        CA3_PYR=CA3_SUP, CA3_INT=CA3_INT_SUP,
        spk_ca3_pyr=spk_ca3_sup, spk_ca3_int=spk_ca3_int_sup,
        vm=vm,
        patterns=patterns, epoch_pattern=epoch_pattern, n_patterns=len(patterns),
        ca3_seq_groups=ca3_sup_groups, ca3_sup_groups=ca3_sup_groups,
        ca3_deep_groups=ca3_deep_groups, n_seq_groups=n_seq_groups,
        swr_on=True,
        swr_fwd=(swr_fwd_start, swr_fwd_stop),
        swr_rev=(swr_rev_start, swr_rev_stop),
        swr_events=[(swr_fwd_start, swr_fwd_stop), (swr_rev_start, swr_rev_stop)],
        swr_ripple_hz=swr_ripple_hz, theta_on=theta_on, theta_hz=theta_hz,
    )


# ============================================================================
# Dentate gyrus module  (Phase 6.2 — real DG replacing the CA3 Poisson proxy)
# ============================================================================
#
# Motivation
# ----------
# "DG" in build_replay_network() was a Poisson generator, not a spiking
# population -- the largest bio-plausibility gap in the EC->DG->CA3->CA1->EC
# loop. This module builds a real DG: granule cells (the pattern-separation
# stage), two mossy-cell classes, and a basket-cell feedback loop, then drives
# CA3 through the mossy-fibre "detonator" synapse.
#
# Neuron parameters come verbatim from CANDIDATE_PARAMS in
# nest_dg_ca3_fi_calibration.py, whose MC_LOW/MC_HIGH threshold split was
# confirmed on MN5 (NEST 3.9) at a 4.97x DC-rheobase ratio against the 5.0x
# Kassab & Alexandre target (MC_HIGH I_e = -15.1).
#
# Circuit
# -------
#   perforant path (Poisson, models EC LII)  --> GC        [sparse, strong]
#   GC        --> DG_BASKET   (E->I)   recruits feedback inhibition
#   DG_BASKET --> GC          (I->E)   winner-take-all -> ~2-4% sparse coding
#   GC        --> MC_LOW/HIGH (mossy collaterals in the hilus)
#   MC        --> GC          (associational back-projection, weak excitatory)
#   MC        --> DG_BASKET   (mossy cells drive feedback inhibition of GC)
#   GC        --> CA3 SUP/DEEP (mossy fibre, LOW in-degree, HIGH weight)
#
# Sparse coding is the DG hallmark and the substrate of pattern separation:
# strong DG_BASKET->GC feedback keeps only the most-driven granule cells above
# threshold, so overlapping cortical inputs map to near-orthogonal CA3 inputs.
# dg_pattern_separation_stats() measures the active fraction as the validation
# metric, alongside the existing CA3 replay-score check.

@dataclass
class DGModule:
    """Dentate gyrus: granule cells, two mossy-cell classes, basket cells.

    Built optionally via --dg, self-contained like ECModule so no existing
    hippocampal code changes when DG is absent. When present, the caller passes
    suppress_dg_drive=True to build_replay_network so the Poisson DG proxy on
    CA3 is replaced by this module's mossy fibres.
    """
    GC          : object   # nest.NodeCollection — granule cells
    MC_LOW      : object   # nest.NodeCollection — low-threshold mossy cells
    MC_HIGH     : object   # nest.NodeCollection — high-threshold mossy cells
    BASKET      : object   # nest.NodeCollection — DG basket/HIPP interneurons
    spk_gc      : object
    spk_mc_low  : object
    spk_mc_high : object
    spk_basket  : object
    N_gc        : int
    N_mc_low    : int
    N_mc_high   : int
    N_basket    : int
    K_mf_sup    : int      # mossy-fibre in-degree onto CA3 SUP
    K_mf_deep   : int      # mossy-fibre in-degree onto CA3 DEEP
    ec_driven   : bool = False   # True when the real EC LII perforant path is wired
                                 # (i.e. the EC->DG->CA3->CA1->EC loop is closed)
    pp_gen       : object = None   # nest.NodeCollection -- original population's
                                    # Poisson-residual generators (one per GC),
                                    # re-randomized every epoch by
                                    # run_dg_residual_refresh_hook
    pp_rate_mean : float  = 0.0
    pp_rate_sigma: float  = 0.0
    pp_scale     : float  = 1.0    # pp_residual if ec_driven else 1.0


def build_dg_module(
    ca3_sup, ca3_deep,
    N_gc, N_mc_low, N_mc_high, N_basket,
    # Real EC LII population for the perforant path. When given, the loop
    # EC LII -> DG -> CA3 -> CA1 -> EC LII is CLOSED; when None, DG falls back
    # to the Poisson stand-in and the loop is open at its entry point.
    # Loop gain: EC LII->GC must contribute only a MODEST share of granule
    # drive, because it closes a positive-feedback loop
    # (EC->DG->CA3->CA1->EC). Budget per granule cell, at 1% scale:
    # The binding constraint is SYNCHRONY, not mean rate. EC LII fires in
    # SWR-locked bursts, so a granule cell's K perforant inputs arrive together:
    # the instantaneous kick is K * w_ec_dg, and the granule rest->threshold gap
    # is 20 mV (-70 -> -50, from the f-I calibration). Any K*w near 20 detonates
    # every granule cell the moment EC bursts, destroying the sparse code.
    #   K=100 w=1.2 (120 mV) residual=0.5 -> runaway: CA3 73 Hz, DG 100% active
    #   K=50  w=0.4 ( 20 mV) residual=0.9 -> DG fine in SWR-1 (1.4%) but
    #                                        saturates to 98.8% by SWR-2 once
    #                                        the loop has built up
    #   K=50  w=0.15 (7.5 mV) residual=0.95 -> subthreshold alone: EC modulates
    #                                        which granule cells win rather than
    #                                        firing them outright. <- current
    # Mean-rate budget at the current setting: Poisson 0.95 * 520 = 494 mV/s
    # plus EC 50 * ~3 Hz * 0.15 = 23, total ~517 vs the tuned 520.
    ec_lii=None, w_ec_dg=0.15, pp_residual=0.95,
    # Perforant-path sampling. field_centers/cluster_sigma together turn on
    # LOCAL (topographic) sampling of the perforant path instead of uniform-
    # random -- see clustered_fixed_connect's docstring and RESULTS.md SS17.
    # field_centers must be in ec_lii's own creation-index order (i.e. the
    # array assign_place_fields(N_ec_lii, ...) returned in main()).
    ec_field_centers=None, dg_ec_cluster_sigma=0.0,
    # Heterogeneity for the DG pathway. The whole module used scalar weights and
    # delays, so a granule cell's K perforant inputs arrived at one instant with
    # one amplitude -- the synchrony ceiling that forces w_ec_dg down to 0.15
    # and leaves the pattern-carrying input at 0.6% of granule drive (§13).
    # Spreading arrival over delay_jitter ms drops the instantaneous kick from
    # K*w to roughly (K/spread_bins)*w, which is what buys headroom on w_ec_dg.
    delay_jitter=0.0, w_cv=None,   # None => inherit the global --het w_cv
                                   # (an explicit 0.0 here silently disabled
                                   #  weight heterogeneity on every DG synapse:
                                   #  JOB H1/H2 both printed cv=0.0 despite --het 0.30)
    # Perforant path (EC LII proxy) — heterogeneous Poisson onto GC.
    # Bracketing the granule sparse-coding target across three 1% runs:
    #   pp_weight 8.0 -> 97% active per window (dense; drowned CA3 replay)
    #   pp_weight 3.0 -> 0.1% active per window (too sparse; CA3_SUP 2.9 Hz)
    # Both bidirectional replay directions PASSED at 3.0 (rho_fwd +0.855,
    # rho_rev -0.782), so the regime is right -- but 0.1%/window is below the
    # 2-4% DG target and CA3 is under-driven. pp_weight 4.0 steps back up toward
    # 2-4% and should lift CA3_SUP back toward ~5 Hz. The relationship is steep,
    # so expect to bracket once more.
    pp_rate_mean=130.0, pp_rate_sigma=70.0, pp_weight=4.0,
    # mossy-fibre DG->CA3 detonator weights (LOW in-degree set in TARGET_INDEGREE).
    # Division of labour: CA3's ~5 Hz baseline comes from the tonic floor
    # (build_replay_network ca3_tonic_rate_sup = EC-direct/recurrent); the mossy
    # fibre adds a sparse pattern-specific kick. Tuning (2026-08-04) showed a
    # strong detonator (w>=6) over-drives CA3 with CA1-ineffective bursty firing
    # that breaks the replay-tuned E/I balance (reverse replay collapses, CA1
    # stays low). At w=2.5 (still ~60x the Schaffer weight of 0.04) the mossy
    # fibre is a real, pattern-specific input that perturbs rather than dominates
    # CA3, so the tonic floor sets the CA1-effective baseline.
    w_mf_ca3_sup=2.5, w_mf_ca3_deep=1.5,
    # intra-DG weights. w_basket_gc strengthened -4.5 -> -7.0 for a sharper
    # k-winners-take-all cutoff.
    w_gc_basket=2.5, w_basket_gc=-7.0,
    w_gc_mc=2.0, w_mc_gc=0.6, w_mc_basket=1.4,
    # Background drive to keep mossy cells / interneurons near threshold.
    # First 1% run (2026-07-24) fired the interneurons and mossy cells at 0 Hz
    # at rate_bg=200-300 / w=1.5, so DG_BASKET->GC feedback never engaged and
    # granule cells ran to ~98% active (dense, no pattern separation). These
    # values are anchored to the PROVEN CA1-basket recipe in this same model
    # (rate_drive_ca1_basket=820 @ w=2.0 -> 47 Hz). MC gets identical background
    # to both classes, so MC_HIGH (I_e=-15.1) fires less than MC_LOW under equal
    # drive -- the confirmed rheobase split. NEEDS a local re-run to confirm the
    # interneurons now fire and to tune w_basket_gc for the 2-4% target.
    rate_bg_mc=700.0, w_bg_mc=2.0,
    rate_bg_basket=820.0, w_bg_basket=2.0,
    seed_connect=42,
) -> DGModule:
    """Create the DG populations and wire them, incl. mossy fibres onto CA3.

    Weights below reproduce the *net* excitatory drive the suppressed Poisson
    proxy delivered to CA3 SUP, but through a sparse, spiking, feedback-shaped
    granule population. The absolute values need confirmation on MN5 (the local
    environment has no NEST); they are set from the confirmed single-neuron f-I
    gains and the removed proxy's drive budget, and flagged for tuning.
    """
    import nest
    t0 = time.perf_counter()

    print(f"\n  [DGModule] Building dentate gyrus")
    print(f"  [DGModule]   N_gc={N_gc:,}  N_mc_low={N_mc_low:,}  "
          f"N_mc_high={N_mc_high:,}  N_basket={N_basket:,}")

    # ---- Populations (params verbatim from the confirmed calibration) -------
    gc_params      = dict(a=0.02, b=0.2, c=-65.0, d=8.0, V_m=-65.0, U_m=-13.0, I_e=0.0)
    mc_low_params  = dict(a=0.02, b=0.2, c=-65.0, d=8.0, V_m=-65.0, U_m=-13.0, I_e=0.0)
    mc_high_params = dict(a=0.02, b=0.2, c=-65.0, d=8.0, V_m=-65.0, U_m=-13.0, I_e=-15.1)
    basket_params  = dict(a=0.10, b=0.2, c=-65.0, d=2.0, V_m=-65.0, U_m=-13.0, I_e=0.0)

    _rng = np.random.default_rng(seed_connect + 7)

    GC = nest.Create("izhikevich", N_gc, params=het_params(gc_params, N_gc, _rng))
    # V_m heterogeneity → graded excitability → sparse, ordered recruitment
    nest.SetStatus(GC, "V_m",
                   _rng.normal(-65.0, 4.0, N_gc).clip(-75, -55).tolist())
    MC_LOW  = nest.Create("izhikevich", N_mc_low,  params=het_params(mc_low_params, N_mc_low, _rng))
    MC_HIGH = nest.Create("izhikevich", N_mc_high, params=het_params(mc_high_params, N_mc_high, _rng))
    BASKET  = nest.Create("izhikevich", N_basket,  params=het_params(basket_params, N_basket, _rng))

    d_fast = 1.5

    # ---- Perforant path onto GC ---------------------------------------------
    # Two sources, mirroring the biology:
    #   * the REAL EC LII projection (when available) -- this is what closes
    #     the EC->DG->CA3->CA1->EC loop;
    #   * a heterogeneous Poisson residual standing for the cortical input not
    #     modelled here. It is also what breaks the cold-start deadlock: EC LII
    #     is silent until CA1 drives it, and CA1 needs DG->CA3 first, so DG
    #     needs some input that does not depend on the loop being already
    #     running. Scaled by pp_residual when the real projection is present so
    #     total granule drive (and hence the tuned 2-4% sparse code) is roughly
    #     preserved.
    #
    # pp_rates is drawn ONCE here and was never touched again -- confirmed
    # (2026-09-05, MN5 12% jobs 45358807/45358815) to bake in a persistent,
    # EC-content-independent per-cell "loudness" bias that dominates DG's
    # winner-take-all: ~1,000+ granule cells were active in >=8/14 epochs
    # regardless of which pattern played (chance predicts ~0), and 66% of
    # that "always-on" set was IDENTICAL between two runs with different
    # --seed, because seed_connect (below, and at the build_dg_module call
    # site) was never threaded from --seed either -- see main()'s dg_module
    # construction. This is why EC LII's real, measured pattern-identity
    # signal (sep +0.18) never reached DG (sep ~0.00) across every mechanism
    # tried: the perforant path's per-epoch kick (K=50 * w_ec_dg) is small
    # next to this fixed per-cell bias, so the same cells win regardless of
    # input. run_dg_residual_refresh_hook (below) re-draws these rates every
    # epoch to fix this -- pp_gen/pp_rate_mean/pp_rate_sigma/pp_scale are
    # kept on DGModule so it can do that.
    ec_driven = ec_lii is not None
    pp_scale  = pp_residual if ec_driven else 1.0
    pp_rates = _rng.normal(pp_rate_mean * pp_scale, pp_rate_sigma * pp_scale,
                           N_gc).clip(5.0, None)
    pp = nest.Create("poisson_generator", N_gc)
    nest.SetStatus(pp, "rate", pp_rates.tolist())
    nest.Connect(pp, GC, conn_spec="one_to_one",
                 syn_spec={"weight": float(pp_weight), "delay": d_fast})
    if ec_driven:
        K_pp = K("ec_dg_pp", len(ec_lii))
        # 3 ms: entorhinal->dentate conduction, same as the other cortical hops
        if dg_ec_cluster_sigma > 0:
            if ec_field_centers is None:
                raise ValueError("dg_ec_cluster_sigma > 0 needs ec_field_centers "
                                  "(the real EC LII place-field centres)")
            clustered_fixed_connect(ec_lii, GC, K_pp, w_ec_dg,
                                    jittered_delay(3.0, delay_jitter),
                                    ec_field_centers, dg_ec_cluster_sigma,
                                    w_cv=w_cv, seed=seed_connect)
        else:
            fixed_connect(ec_lii, GC, K_pp, w_ec_dg,
                          jittered_delay(3.0, delay_jitter), w_cv=w_cv)
        print(f"  [DGModule] perforant path EC LII->GC: K={K_pp} w={w_ec_dg} "
              f"(cv={w_cv}, delay 3.0+/-{delay_jitter} ms"
              f"{f', CLUSTERED sigma={dg_ec_cluster_sigma}' if dg_ec_cluster_sigma > 0 else ''}) "
              f"({len(ec_lii):,} EC -> {N_gc:,} GC)  ** loop CLOSED **")
    else:
        print("  [DGModule] perforant path: Poisson stand-in only "
              "(no --ec-lii; EC->DG loop OPEN)")

    # ---- Background drive: mossy cells + interneurons near threshold ---------
    # These hold cells NEAR THRESHOLD -- they set an operating point, they do
    # not carry signal, so weight compensation must not touch the interneuron
    # ones. Compensating the basket background at wcomp 2.3 drove DG baskets
    # 0.79 -> 58 Hz (73x) and, through w_basket_gc = -7.0, silenced the granule
    # cells completely: DG active fraction 1.88% -> 0.00%.
    # NONE of these are compensated. The mossy-cell drive was left compensated
    # in the first pass on the argument that mossy cells are principal cells;
    # JOB H1/H2 at 12% showed that is wrong in effect. At wcomp 2.3 it drove
    # MC_LOW to 11.8 Hz (27x its homogeneous 0.43 Hz), which through MC->BASKET
    # put DG baskets at 22.3 Hz and clamped granule cells to 0.45% active
    # against the 2-4% target. With GC inhibition-clamped, an 8x rise in the
    # perforant weight moved nothing: H2 changed DG selectivity by 0.002.
    for pop, rate, w, comp in [(MC_LOW, rate_bg_mc, w_bg_mc, False),
                               (MC_HIGH, rate_bg_mc, w_bg_mc, False),
                               (BASKET, rate_bg_basket, w_bg_basket, False)]:
        bg = nest.Create("poisson_generator", len(pop), params={"rate": float(rate)})
        nest.Connect(bg, pop, conn_spec="one_to_one",
                     syn_spec={"weight": wcomp_w(w) if comp else float(w),
                               "delay": d_fast})

    # ---- Intra-DG feedback loops (fixed_indegree, C++/MPI-parallel) ---------
    print("  [DGModule] Wiring intra-DG feedback (fixed_indegree)...")
    MC = MC_LOW + MC_HIGH        # combined mossy-cell NodeCollection for GC/basket wiring

    # GC -> basket (E->I): recruit feedback inhibition
    fixed_connect(GC, BASKET, K("dg_gc_basket", N_gc), w_gc_basket,
                  jittered_delay(d_fast, delay_jitter), w_cv=w_cv,
                  compensate=False)
    # basket -> GC (I->E): the sparsifying feedback loop
    fixed_connect(BASKET, GC, K("dg_basket_gc", N_basket), w_basket_gc,
                  jittered_delay(d_fast, delay_jitter), w_cv=w_cv)
    # GC -> mossy cells (mossy collaterals in the hilus)
    fixed_connect(GC, MC_LOW,  K("dg_gc_mc", N_gc), w_gc_mc,
                  jittered_delay(d_fast, delay_jitter), w_cv=w_cv)
    fixed_connect(GC, MC_HIGH, K("dg_gc_mc", N_gc), w_gc_mc,
                  jittered_delay(d_fast, delay_jitter), w_cv=w_cv)
    # mossy cells -> GC (associational back-projection, net excitatory but weak)
    fixed_connect(MC, GC, K("dg_mc_gc", len(MC)), w_mc_gc,
                  jittered_delay(d_fast, delay_jitter), w_cv=w_cv)
    # mossy cells -> basket (drive feedback inhibition of GC: the MC "gain control")
    fixed_connect(MC, BASKET, K("dg_mc_basket", len(MC)), w_mc_basket, d_fast, compensate=False)

    # ---- Mossy fibre DG -> CA3 (detonator: low in-degree, high weight) -------
    print("  [DGModule] Wiring mossy fibres GC->CA3 (detonator)...")
    t_mf = time.perf_counter()
    K_mf_sup  = K("dg_mf_ca3_sup",  N_gc)
    K_mf_deep = K("dg_mf_ca3_deep", N_gc)
    fixed_connect(GC, ca3_sup,  K_mf_sup,  w_mf_ca3_sup,
                  jittered_delay(d_fast, delay_jitter), w_cv=w_cv)
    fixed_connect(GC, ca3_deep, K_mf_deep, w_mf_ca3_deep,
                  jittered_delay(d_fast, delay_jitter), w_cv=w_cv)
    n_mf = len(ca3_sup) * K_mf_sup + len(ca3_deep) * K_mf_deep
    print(f"  [DGModule] mossy fibres: ~{n_mf:,} synapses "
          f"(K_sup={K_mf_sup}, K_deep={K_mf_deep})  "
          f"in {time.perf_counter()-t_mf:.2f}s")

    # ---- Recorders ----------------------------------------------------------
    spk_gc      = nest.Create("spike_recorder")
    spk_mc_low  = nest.Create("spike_recorder")
    spk_mc_high = nest.Create("spike_recorder")
    spk_basket  = nest.Create("spike_recorder")
    nest.Connect(GC,      spk_gc)
    nest.Connect(MC_LOW,  spk_mc_low)
    nest.Connect(MC_HIGH, spk_mc_high)
    nest.Connect(BASKET,  spk_basket)

    print(f"  [DGModule] Total DG build: {time.perf_counter()-t0:.1f}s")

    return DGModule(
        GC=GC, MC_LOW=MC_LOW, MC_HIGH=MC_HIGH, BASKET=BASKET,
        spk_gc=spk_gc, spk_mc_low=spk_mc_low, spk_mc_high=spk_mc_high,
        spk_basket=spk_basket,
        N_gc=N_gc, N_mc_low=N_mc_low, N_mc_high=N_mc_high, N_basket=N_basket,
        K_mf_sup=K_mf_sup, K_mf_deep=K_mf_deep, ec_driven=ec_driven,
        pp_gen=pp, pp_rate_mean=pp_rate_mean, pp_rate_sigma=pp_rate_sigma,
        pp_scale=pp_scale,
    )


def dg_pattern_separation_stats(dg_module, sim_ms, window=None):
    """Sparse-coding / pattern-separation metric for the granule population.

    The DG hallmark is a very low active fraction (~2-4% of granule cells per
    input pattern; Chawla et al. 2005, Jung & McNaughton 1993). A high active
    fraction means feedback inhibition is too weak and pattern separation is
    lost -- the DG-specific analog of a failed replay score.

    Returns a dict: active_fraction, n_active, n_gc, mean_rate_hz,
    plus a PASS/FLAG verdict against the 2-4% sparse-coding band.
    """
    ev = nest.GetStatus(dg_module.spk_gc, "events")[0]
    t = np.asarray(ev["times"]);  s = np.asarray(ev["senders"])
    if window is not None:
        m = (t >= window[0]) & (t <= window[1])
        t, s = t[m], s[m]
        dur_s = (window[1] - window[0]) / 1000.0
    else:
        dur_s = sim_ms / 1000.0
    n_active = int(np.unique(s).size)
    frac = n_active / max(dg_module.N_gc, 1)
    mean_hz = len(t) / (max(dg_module.N_gc, 1) * max(dur_s, 1e-9))
    verdict = "PASS" if 0.005 <= frac <= 0.06 else "FLAG"
    return dict(active_fraction=frac, n_active=n_active, n_gc=dg_module.N_gc,
                mean_rate_hz=mean_hz, verdict=verdict)


def run_dg_residual_refresh_hook(dg_module, rng):
    """Re-draw the original GC population's Poisson-residual firing rates.

    build_dg_module drew dg_module.pp_gen's per-cell rates ONCE and never
    touched them again -- confirmed (2026-09-05) to bake in a persistent,
    pattern-independent "loudness" bias strong enough to dominate DG's
    winner-take-all over the real, measurably pattern-locked EC LII
    perforant-path input (see the comment at pp_rates' first draw in
    build_dg_module). Redrawing every epoch, between nest.Simulate() calls
    (same interleaving as every other per-epoch hook here), turns this
    input back into pure trial-to-trial noise instead of a fixed per-cell
    identity, so it can no longer determine which cells win regardless of
    which pattern is active.

    Call once per epoch whenever --dg is on, independent of neurogenesis
    (this refreshes the ORIGINAL population; neurogenesis cohorts' own
    residual generators are refreshed inside run_dg_neurogenesis_hook).
    """
    import nest
    n = len(dg_module.pp_gen)
    rates = rng.normal(dg_module.pp_rate_mean * dg_module.pp_scale,
                        dg_module.pp_rate_sigma * dg_module.pp_scale,
                        n).clip(5.0, None)
    nest.SetStatus(dg_module.pp_gen, "rate", rates.tolist())


# ---- Phase 8: DG neurogenesis (adult-born granule cell turnover) ----------
# Young adult-born granule cells are transiently hyperexcitable (lower
# rheobase), receive substantially less perisomatic GABAergic inhibition, and
# show enhanced afferent synaptic gain (Aimone et al. 2009/2011; Marin-Burgin
# et al. 2012) -- then mature into ordinary granule cells indistinguishable
# from the rest of the population over a period here compressed to a few
# epochs. The leading functional hypothesis is TEMPORAL pattern separation:
# a different cohort of hyperexcitable cells is present at any given time, so
# similar experiences separated in time get different codes even when a
# static network would conflate them.
#
# Population size only grows here -- NEST has no clean mid-run node deletion,
# and the claim under test (young cells behave differently) does not require
# old cells to actually die, only new ones to exist. Over a typical 8-14
# epoch run that is a bounded, modest increase (~10-15% of the original
# granule count).

@dataclass
class DGNeurogenesisCohort:
    """One birth-cohort of granule cells, aging toward the mature baseline.

    Holds ONE target-only SynapseCollection (all of this cohort's incoming
    synapses) plus boolean masks and a full weight vector, rather than
    separate per-source-population handles -- SetStatus is always called on
    the full collection with a full-length weight list (non-aged entries,
    e.g. the Poisson residual and MC associational input, write back their
    own unchanged value as a no-op). This is the same technique
    build_homeostasis_hook uses, and for the same reason: see the note in
    run_dg_neurogenesis_hook about why source+target GetConnections is
    never used here.
    """
    gc           : object       # nest.NodeCollection -- this cohort's cells
    birth_epoch  : int
    in_conns     : object       # nest.SynapseCollection, target=gc (ALL incoming)
    pp_mask      : np.ndarray   # bool -- which in_conns entries are EC LII->gc
    inhib_mask   : np.ndarray   # bool -- which in_conns entries are BASKET->gc
    base_weights : np.ndarray   # current full weight vector (mutated + written back)
    pre_idx      : np.ndarray   # index into EC LII population (Phase 10 plasticity;
                                 # only meaningful where pp_mask is True, else -1)
    post_idx     : np.ndarray   # index into THIS cohort's own gc (every entry has
                                 # a valid target here, since in_conns is target=gc)
    pp_gen       : object = None   # nest.NodeCollection -- this cohort's own
                                    # Poisson-residual generators, re-randomized
                                    # every epoch for the same reason
                                    # run_dg_residual_refresh_hook does for the
                                    # original population (see that docstring)


@dataclass
class DGNeurogenesisHook:
    """State for Phase 8 neurogenesis. Built once, then run_dg_neurogenesis_
    hook() is called once per epoch, between nest.Simulate() calls -- exactly
    the same interleaving the STC/mPFC-association hooks already use, which
    is what makes creating/wiring new cells here safe: NEST supports Create/
    Connect/SetStatus freely between Simulate() calls."""
    ec_lii            : object   # nest.NodeCollection, perforant-path source
    ca3_sup           : object
    ca3_deep          : object
    original_n_gc     : int
    maturation_epochs : int
    young_ie          : float
    young_inhib_scale : float
    young_pp_scale    : float
    w_ec_dg           : float    # mature baseline weights, to relax toward
    w_basket_gc       : float
    w_gc_basket       : float
    w_gc_mc           : float
    w_mc_gc           : float
    w_mf_ca3_sup      : float
    w_mf_ca3_deep     : float
    pp_rate_mean      : float
    pp_rate_sigma     : float
    pp_weight         : float
    rng               : object
    cohorts           : list = None


def _dg_cache_cohort_incoming(gc_pop, ec_lii, basket_pop):
    """Target-only GetConnections snapshot of gc_pop's incoming synapses,
    post-filtered by source in Python. NOT nest.GetConnections(source=,
    target=) -- see run_dg_neurogenesis_hook's note on why. Shared by
    build_dg_neurogenesis_hook (the original population, cohort -1) and
    run_dg_neurogenesis_hook (every later cohort), so the two never drift
    out of sync with each other.
    """
    import nest
    in_conns     = nest.GetConnections(target=gc_pop)
    src          = np.array(nest.GetStatus(in_conns, "source"), dtype=np.int64)
    tgt          = np.array(nest.GetStatus(in_conns, "target"), dtype=np.int64)
    base_weights = np.array(nest.GetStatus(in_conns, "weight"), dtype=np.float64)
    ec_map  = {g: i for i, g in enumerate(ec_lii.tolist())}
    bsk_ids = set(basket_pop.tolist())
    gc_map  = {g: i for i, g in enumerate(gc_pop.tolist())}
    pre_idx    = np.array([ec_map.get(g, -1) for g in src], dtype=np.int64)
    pp_mask    = pre_idx >= 0
    inhib_mask = np.array([g in bsk_ids for g in src], dtype=bool)
    post_idx   = np.array([gc_map[g] for g in tgt], dtype=np.int64)
    return in_conns, pp_mask, inhib_mask, base_weights, pre_idx, post_idx


def build_dg_neurogenesis_hook(
    dg_module, ec_lii, ca3_sup, ca3_deep,
    maturation_epochs=4, young_ie=4.0, young_inhib_scale=0.3, young_pp_scale=2.0,
    w_ec_dg=0.15,
    # The rest match build_dg_module's own defaults verbatim -- new cohorts
    # must converge on the SAME mature baseline the original GC population
    # was built with, not independently-chosen values.
    w_basket_gc=-7.0, w_gc_basket=2.5, w_gc_mc=2.0, w_mc_gc=0.6,
    w_mf_ca3_sup=2.5, w_mf_ca3_deep=1.5,
    pp_rate_mean=130.0, pp_rate_sigma=70.0, pp_weight=4.0,
    seed=42,
) -> DGNeurogenesisHook:
    """Builds the hook with an empty cohort list -- the original
    (pre-neurogenesis) granule population is NOT tracked here and gets NO
    plasticity; only cohorts born by run_dg_neurogenesis_hook are ever
    added to hook.cohorts, so run_dg_perforant_plasticity_hook has nothing
    to do until neurogenesis actually births a cohort.

    An earlier version of this also seeded a synthetic "cohort -1" for the
    original population, giving it the same mature-baseline plasticity as
    aged neurogenesis cohorts. Reverted: on MN5 at 12% scale, dg_module.GC
    is 143,990 cells / ~28.65M incoming synapses, and the one-time
    GetConnections(target=...) cache alone cost 50,130-56,536s (14-16
    hours) -- confirmed by two killed MN5 runs (job 45305940, 45305945),
    versus build_mpfc_assoc_hook's comparable cache (~28,800 synapses,
    3.5-4h) at essentially the SAME total kernel size at call time (~5.2M
    conns/VP both cases) -- so the cost tracks the target population's own
    size, not overall kernel size, and does not scale down by reordering.
    Re-touching that many synapses via SetStatus every epoch would not
    scale either. --dg-perforant-stdp therefore now REQUIRES
    --dg-neurogenesis (enforced in argument validation) since without any
    cohorts it would be a silent no-op.
    """
    hook = DGNeurogenesisHook(
        ec_lii=ec_lii, ca3_sup=ca3_sup, ca3_deep=ca3_deep,
        original_n_gc=dg_module.N_gc, maturation_epochs=maturation_epochs,
        young_ie=young_ie, young_inhib_scale=young_inhib_scale,
        young_pp_scale=young_pp_scale, w_ec_dg=w_ec_dg, w_basket_gc=w_basket_gc,
        w_gc_basket=w_gc_basket, w_gc_mc=w_gc_mc, w_mc_gc=w_mc_gc,
        w_mf_ca3_sup=w_mf_ca3_sup, w_mf_ca3_deep=w_mf_ca3_deep,
        pp_rate_mean=pp_rate_mean, pp_rate_sigma=pp_rate_sigma, pp_weight=pp_weight,
        rng=np.random.default_rng(seed + 31), cohorts=[],
    )
    return hook


def _dg_cohort_age_frac(hook, age):
    """0 at birth -> 1 at maturation_epochs, clamped. Shared by
    run_dg_neurogenesis_hook (intrinsic-property relaxation) and
    run_dg_perforant_plasticity_hook (Phase 10 learning-rate relaxation) so
    both age a cohort identically."""
    return min(1.0, age / hook.maturation_epochs) if hook.maturation_epochs > 0 else 1.0


def run_dg_neurogenesis_hook(hook, dg_module, epoch, rate):
    """Birth one new cohort and age every tracked cohort's INTRINSIC
    properties (excitability, inhibition received) toward maturity. Does
    NOT touch perforant-path weights -- those are governed entirely by
    run_dg_perforant_plasticity_hook now (Phase 10); see that function for
    why a static young-cell weight multiplier was retired in favour of an
    age-dependent LEARNING RATE.

    Call once per epoch (including epoch 0), BEFORE that epoch's
    nest.Simulate(), so newly created cells are wired in time to participate.
    """
    import nest

    def _intrinsic_state(age):
        frac = _dg_cohort_age_frac(hook, age)
        inhib_scale = hook.young_inhib_scale + frac * (1.0 - hook.young_inhib_scale)
        ie_val      = hook.young_ie * (1.0 - frac)
        return inhib_scale, ie_val

    # ---- birth a new cohort --------------------------------------------------
    n_new = max(1, round(rate * hook.original_n_gc))
    inhib0, ie0 = _intrinsic_state(0)
    gc_params = dict(a=0.02, b=0.2, c=-65.0, d=8.0, V_m=-65.0, U_m=-13.0, I_e=ie0)
    new_gc = nest.Create("izhikevich", n_new, params=gc_params)
    nest.SetStatus(new_gc, "V_m",
                   hook.rng.normal(-65.0, 4.0, n_new).clip(-75, -55).tolist())

    # perforant path IN: plain baseline weight, same as the mature
    # population, for every cohort regardless of age -- Phase 10 moved the
    # young-cell effect from a static weight multiplier here to the
    # LEARNING RATE in run_dg_perforant_plasticity_hook, so what makes a
    # young cohort's synapses end up different is what they experience
    # while young, not an arbitrary age-indexed scalar.
    K_pp = K("ec_dg_pp", len(hook.ec_lii))
    fixed_connect(hook.ec_lii, new_gc, K_pp, hook.w_ec_dg, 3.0,
                  compensate=False)

    pp_rates = hook.rng.normal(hook.pp_rate_mean, hook.pp_rate_sigma,
                               n_new).clip(5.0, None)
    pp = nest.Create("poisson_generator", n_new)
    nest.SetStatus(pp, "rate", pp_rates.tolist())
    nest.Connect(pp, new_gc, conn_spec="one_to_one",
                 syn_spec={"weight": float(hook.pp_weight), "delay": 1.5})

    K_basket_gc = K("dg_basket_gc", len(dg_module.BASKET))
    fixed_connect(dg_module.BASKET, new_gc, K_basket_gc,
                  hook.w_basket_gc * inhib0, 1.5, compensate=False)

    # associational back-projection IN (MC -> new cells): full K, same
    # reasoning as perforant path / inhibition above. Wired here, BEFORE the
    # GetConnections snapshot below, so that snapshot captures every one of
    # new_gc's incoming connections in a single pass.
    MC = dg_module.MC_LOW + dg_module.MC_HIGH
    fixed_connect(MC, new_gc, K("dg_mc_gc", len(MC)), hook.w_mc_gc, 1.5,
                  compensate=False)

    # Cache ONE target-only GetConnections snapshot of ALL of new_gc's
    # incoming synapses, post-filtered by source in Python -- NOT
    # nest.GetConnections(source=, target=), which forces NEST 3.9 to walk
    # the FULL kernel connectome (>15h on MN5 at 12% scale, confirmed: this
    # exact mistake, made twice per epoch, is what killed the first MN5
    # neurogenesis run at epoch 2/14 after >13h -- see build_homeostasis_
    # hook's docstring for the same lesson learned earlier for CA3). A
    # target-only query walks only new_gc's own incoming slots (a few
    # hundred synapses per cell), the same trick STCHook/homeostasis use.
    in_conns, pp_mask, inhib_mask, base_weights, pre_idx, post_idx = \
        _dg_cache_cohort_incoming(new_gc, hook.ec_lii, dg_module.BASKET)

    # new cells as SOURCE into fixed-size populations (basket recruitment,
    # mossy collaterals, mossy fibre onto CA3): a TOKEN indegree only, not
    # the full K. A fresh full-strength K from every cohort would compound
    # -- each cohort adding another whole K's worth of convergence would
    # inflate basket drive and CA3's mossy-fibre input epoch over epoch.
    # Real young cells are a small fraction of the total granule pool and
    # should perturb their targets proportionally, not repeat the full
    # detonator every time a cohort is born.
    def _token_k(key):
        return max(1, round(TARGET_INDEGREE[key] * n_new / hook.original_n_gc))

    fixed_connect(new_gc, dg_module.BASKET, _token_k("dg_gc_basket"),
                  hook.w_gc_basket, 1.5, compensate=False)
    fixed_connect(new_gc, dg_module.MC_LOW, _token_k("dg_gc_mc"),
                  hook.w_gc_mc, 1.5, compensate=False)
    fixed_connect(new_gc, dg_module.MC_HIGH, _token_k("dg_gc_mc"),
                  hook.w_gc_mc, 1.5, compensate=False)
    fixed_connect(new_gc, hook.ca3_sup, _token_k("dg_mf_ca3_sup"),
                  hook.w_mf_ca3_sup, 1.5, compensate=False)
    fixed_connect(new_gc, hook.ca3_deep, _token_k("dg_mf_ca3_deep"),
                  hook.w_mf_ca3_deep, 1.5, compensate=False)

    # Grow the module: every existing consumer (dg_pattern_separation_stats,
    # pattern_discrimination) just reads dg_module.GC/N_gc, so this is the
    # only change needed for them to see the new cells automatically.
    nest.Connect(new_gc, dg_module.spk_gc)
    dg_module.GC = dg_module.GC + new_gc
    dg_module.N_gc = dg_module.N_gc + n_new

    hook.cohorts.append(DGNeurogenesisCohort(
        gc=new_gc, birth_epoch=epoch, in_conns=in_conns,
        pp_mask=pp_mask, inhib_mask=inhib_mask, base_weights=base_weights,
        pre_idx=pre_idx, post_idx=post_idx, pp_gen=pp))

    # ---- age every tracked cohort's INTRINSIC properties, including the
    # one just born (age 0). Perforant-path weights are NOT touched here --
    # see run_dg_perforant_plasticity_hook.
    for coh in hook.cohorts:
        age = epoch - coh.birth_epoch
        # Poisson-residual refresh runs for EVERY cohort EVERY epoch,
        # regardless of age -- same reason run_dg_residual_refresh_hook
        # refreshes the original population: a fixed per-cell rate becomes
        # a persistent, pattern-independent bias (confirmed 2026-09-05) that
        # would otherwise dominate this cohort's contribution to DG's
        # winner-take-all just as it did for the original population.
        pp_rates = hook.rng.normal(hook.pp_rate_mean, hook.pp_rate_sigma,
                                   len(coh.pp_gen)).clip(5.0, None)
        nest.SetStatus(coh.pp_gen, "rate", pp_rates.tolist())
        if age > hook.maturation_epochs:
            continue  # intrinsic properties already relaxed to baseline
        inhib_scale, ie_val = _intrinsic_state(age)
        coh.base_weights[coh.inhib_mask] = hook.w_basket_gc * inhib_scale
        # SetStatus on the full collection -- non-aged entries (perforant
        # path, Poisson residual, MC associational input) write back their
        # own unchanged value as a no-op, same as run_homeostasis_hook.
        nest.SetStatus(coh.in_conns, "weight", coh.base_weights.tolist())
        nest.SetStatus(coh.gc, "I_e", float(ie_val))

    print(f"  [DGNeurogenesis] epoch {epoch}: +{n_new} new GC (I_e={ie0:+.1f}), "
          f"total N_gc={dg_module.N_gc:,}, {len(hook.cohorts)} cohort(s) tracked")


# ---- Phase 10: EC LII -> GC perforant-path plasticity ----------------------
# Retires the Phase 8 static young_pp_scale weight multiplier. Without any
# learning, "young" vs "old" was purely a function of time-since-birth --
# there was no mechanism by which a granule cell's response could depend on
# WHICH pattern it has been exposed to, so there was no way for a young
# cohort to respond differently to a novel pattern than to a familiar one
# even in principle. This gives EC LII->GC an actual Hebbian rule, using the
# exact same co-activation-potentiates / post-without-pre-depresses
# algorithm as run_mpfc_assoc_hook (EC LV->mPFC) -- see that function's
# docstring for the rule itself. young_pp_scale now multiplies the LEARNING
# RATE (not the weight directly): a young cohort's synapses move toward
# whatever is co-active WHILE THEY ARE YOUNG much faster than a mature
# cohort's would, which is the actual mechanism real adult-born granule
# cells are thought to use (enhanced LTP during a critical period), not an
# arbitrary age-indexed scalar.
#
# Applies ONLY to neurogenesis cohorts (hook.cohorts, birthed by
# run_dg_neurogenesis_hook) -- the original mature GC population is NOT
# tracked and keeps fixed baseline perforant-path weights. An earlier
# version also tracked a synthetic "cohort -1" for the original
# population so every GC had some baseline plasticity; reverted after two
# MN5 runs (12% scale) showed its one-time GetConnections cache alone
# (143,990 cells / ~28.65M synapses) cost 14-16 hours -- most of the 20h
# wall-time budget -- with no epoch of simulation completed either time.
# See build_dg_neurogenesis_hook's docstring for the measurements. This
# means --dg-perforant-stdp is a no-op without --dg-neurogenesis (enforced
# in argument validation), and the effect under test is now simply
# "do young, plastic cohorts show a different response to a novel pattern
# than the static mature population does", not a within-population
# young-vs-baseline comparison.

def run_dg_perforant_plasticity_hook(hook, dg_module, ec_module,
                                     t_swr_start, t_swr_end, epoch,
                                     A_assoc=0.01, A_hetero=0.002,
                                     w_max=1.5, w_min=0.05):
    """Hebbian EC LII -> GC learning, per cohort. Call twice per epoch
    (forward + reverse SWR window), same as run_stc_hook/run_mpfc_assoc_hook.
    """
    import nest

    def fired(spk_rec, pop):
        ev = nest.GetStatus(spk_rec, "events")[0]
        t = np.asarray(ev["times"]); s = np.asarray(ev["senders"])
        act = np.unique(s[(t >= t_swr_start) & (t <= t_swr_end)])
        idx = {g: i for i, g in enumerate(pop.tolist())}
        m = np.zeros(len(pop), dtype=bool)
        for g in act:
            i = idx.get(int(g))
            if i is not None:
                m[i] = True
        return m

    pre_fired = fired(ec_module.spike_rec, ec_module.population)

    for coh in hook.cohorts:
        age = epoch - coh.birth_epoch
        frac = _dg_cohort_age_frac(hook, age)
        a_eff = A_assoc * (hook.young_pp_scale + frac * (1.0 - hook.young_pp_scale))

        post_fired = fired(dg_module.spk_gc, coh.gc)
        pre_a  = np.zeros(len(coh.base_weights), dtype=bool)
        pre_a[coh.pp_mask] = pre_fired[coh.pre_idx[coh.pp_mask]]
        post_a = post_fired[coh.post_idx]

        both   = coh.pp_mask & pre_a & post_a     # Hebbian: co-activation -> potentiate
        hetero = coh.pp_mask & post_a & ~pre_a    # post without pre -> weak depression
        coh.base_weights[both]   = np.minimum(coh.base_weights[both]   + a_eff,   w_max)
        coh.base_weights[hetero] = np.maximum(coh.base_weights[hetero] - A_hetero, w_min)
        # SetStatus on the full collection -- non-pp entries (inhibition,
        # Poisson residual, MC associational input) write back their own
        # unchanged value as a no-op, same as run_homeostasis_hook.
        nest.SetStatus(coh.in_conns, "weight", coh.base_weights.tolist())


# ============================================================================
# CA3 pattern completion  (auto-association probe)
# ============================================================================
#
# Pattern separation (DG, above) and pattern completion (CA3) are the classic
# complementary pair. Completion is CA3's auto-associative hallmark: a PARTIAL
# cue of a stored assembly is restored to the full pattern by the recurrent
# collaterals. The sequence-replay experiment never tests this -- its triggers
# activate a whole group. This probe cues only a fraction of one group and
# measures how much of the REST the recurrent loop reactivates.
#
# Clean isolation: the probe runs on a network built with the between-group
# sequence chain OFF (w_seq_fwd = w_seq_bwd = 0) so no hetero-associative
# propagation contaminates the measurement -- only the within-group recurrence
# (sup_local, the auto-associative loop) can complete the pattern. The control
# ablates that loop (w_sup_local = 0); completion should then collapse, proving
# the recurrent collaterals -- not the cue -- did the work (Marr 1971;
# Nakazawa et al. 2002, CA3-NMDA knockout abolishes completion).

def make_partial_cue(group_cells, cue_frac, start_ms, dur_ms,
                     rate, weight, rng, delay=1.0):
    """Drive only a random `cue_frac` subset of one group's cells.

    Returns (cued_ids, uncued_ids) as int arrays so the completion measurement
    can separate the cue from the cells the recurrent loop must recover.
    """
    cells   = np.asarray([int(i) for i in group_cells], dtype=np.int64)
    n_cue   = max(1, int(round(cue_frac * len(cells))))
    cued    = np.sort(rng.choice(cells, size=n_cue, replace=False))
    uncued  = np.setdiff1d(cells, cued)
    gens = nest.Create("poisson_generator", len(cued), params={
        "rate": float(rate), "start": float(start_ms),
        "stop":  float(start_ms + dur_ms),
    })
    nest.Connect(gens, nest.NodeCollection([int(i) for i in cued]),
                 conn_spec="one_to_one",
                 syn_spec={"weight": float(weight), "delay": float(delay)})
    return cued, uncued


def completion_index(spk_times, spk_senders, cued, uncued, win_start, win_stop):
    """Fraction of un-cued assembly cells reactivated within the window.

    completion = |uncued cells that fired| / |uncued cells|
    cue_recall = |cued cells that fired|  / |cued cells|   (sanity: ~1.0)

    A high completion with a small cue is auto-association; near-zero
    completion (with cue_recall still ~1) is the ablated / no-recurrence case.
    """
    m = (spk_times >= win_start) & (spk_times <= win_stop)
    fired = np.unique(spk_senders[m])
    # NaN (not 0) when the cue is the whole assembly: there is nothing to
    # complete, so completion is undefined rather than failed.
    comp = (np.intersect1d(uncued, fired).size / uncued.size
            if uncued.size else float("nan"))
    rec  = np.intersect1d(cued,   fired).size / max(cued.size, 1)
    return dict(completion=comp, cue_recall=rec,
                n_uncued=int(uncued.size), n_cued=int(cued.size),
                n_uncued_fired=int(np.intersect1d(uncued, fired).size))


def run_pattern_completion(cfg, n_threads, cue_fracs,
                           ablate=False, cue_rate=2600.0, cue_weight=2.5,
                           cue_dur_ms=16.0, win_ms=60.0, gap_ms=150.0,
                           prime_rate=250.0, w_sup_local=3.0, w_ca3_ie_sup=-0.5,
                           seed=42):
    """Auto-association probe: partial cue -> recurrent completion of a group.

    Builds a CA3 with the between-group sequence chain OFF (so no replay
    propagation contaminates the measurement). Only the within-group recurrence
    (sup_local) can restore the pattern. `ablate=True` zeroes sup_local -- the
    control in which completion should vanish while cue_recall stays ~1.

    Priming
    -------
    The first (cold-baseline) version returned 0 completion everywhere: from
    rest, cued cells recruited CA3's fast feedback inhibition (E->I->E ~3 ms)
    before the slower within-group recurrence (sup_local at the 4 ms sequence
    delay) could ignite the un-cued cells. Biologically, completion happens
    during a sharp-wave, when CA3 is broadly depolarised. `prime_rate` gives
    CA3 SUP a subthreshold background so cells sit near threshold (low baseline
    firing) and cue-driven recurrence can tip them over. The pre-cue baseline
    window is measured over an equal-length slice and subtracted, so priming
    cannot manufacture a false completion signal -- if priming alone fired the
    un-cued cells, baseline would rise by the same amount.

    Each cue fraction is delivered to a DISTINCT group at a DISTINCT time
    (spaced by gap_ms) so all fractions are swept in one build.
    """
    # Auto-association needs the within-group recurrence to be able to win
    # locally against feedback inhibition -- the replay-tuned default
    # (w_sup_local 0.90 vs INT->SUP -2.0) cannot, so the probe strengthens the
    # recurrence and weakens the inhibition. These apply ONLY to this isolated
    # probe build; replay runs keep the -2.0 / 0.90 defaults.
    print(f"\n  [PatternCompletion] build: ablate={ablate}  "
          f"cue_fracs={cue_fracs}  cue(rate={cue_rate},w={cue_weight})  "
          f"prime_rate={prime_rate}  w_sup_local={0.0 if ablate else w_sup_local}  "
          f"w_ca3_ie_sup={w_ca3_ie_sup}")

    net = build_replay_network(
        N_ca3_sup      = cfg["N_ca3_sup"],
        N_ca3_deep     = cfg["N_ca3_deep"],
        N_ca3_int_sup  = cfg["N_ca3_int_sup"],
        N_ca3_int_deep = cfg["N_ca3_int_deep"],
        N_ca1_pyr      = cfg["N_ca1_pyr"],
        N_ca1_basket   = cfg["N_ca1_basket"],
        N_ca1_olm      = cfg["N_ca1_olm"],
        n_seq_groups   = cfg["n_seq_groups"],
        n_threads      = n_threads,
        # isolate auto-association: no sequence chain, no replay stimulation
        trigger_on=False, scaffold_on=False, theta_on=False,
        w_seq_fwd=0.0, w_seq_bwd=0.0, w_deep_fwd=0.0,
        w_sup_local=(0.0 if ablate else w_sup_local),
        w_ca3_ie_sup=w_ca3_ie_sup,
        # sharp-wave-like priming: CA3 SUP near threshold, low baseline firing,
        # so cue-driven recurrence can complete. Other external drive off; INT
        # drive off leaves feedback inhibition responsive rather than tonic.
        suppress_dg_drive=True,
        rate_ec_ca3_sup=0.0, rate_ec_ca3_deep=0.0,
        rate_ca3_drive_sup=prime_rate, rate_ca3_drive_deep=0.0,
        rate_drive_ca3_int_sup=0.0, rate_drive_ca3_int_deep=0.0,
    )

    rng    = np.random.default_rng(seed)
    groups = net["ca3_sup_groups"]
    n_grp  = len(groups)
    if len(cue_fracs) > n_grp:
        raise ValueError(f"{len(cue_fracs)} cue fractions but only {n_grp} groups; "
                         f"use --scale with more groups or fewer fractions.")

    # Assign each fraction to its own group, spread across the middle of the
    # sequence (avoid the endpoint groups 0 and n-1, which are smaller targets
    # for replay elsewhere and keep the probe comparable across scales).
    grp_choices = np.linspace(1, n_grp - 2, num=len(cue_fracs)).round().astype(int)
    t0 = 100.0   # baseline window is [0, t0]
    cues = []
    t = t0
    for frac, gidx in zip(cue_fracs, grp_choices):
        cued, uncued = make_partial_cue(groups[int(gidx)], frac, t, cue_dur_ms,
                                        cue_rate, cue_weight, rng)
        cues.append((frac, int(gidx), t, cued, uncued))
        t += gap_ms
    total_ms = t + gap_ms

    print(f"  [PatternCompletion] simulating {total_ms:.0f} ms "
          f"({len(cues)} cues, gap={gap_ms:.0f} ms)...")
    nest.Simulate(total_ms)

    t_sup, s_sup = _get_spikes(net["spk_ca3_sup"])
    results = []
    for frac, gidx, tc, cued, uncued in cues:
        ci   = completion_index(t_sup, s_sup, cued, uncued, tc, tc + win_ms)
        # baseline over an EQUAL-length quiet slice just before the first cue,
        # so completion-above-baseline is a fair comparison
        base = completion_index(t_sup, s_sup, cued, uncued, t0 - win_ms, t0)
        results.append(dict(cue_frac=float(frac), group=gidx, t_cue=tc,
                            completion=ci["completion"],
                            completion_baseline=base["completion"],
                            cue_recall=ci["cue_recall"],
                            n_cued=ci["n_cued"], n_uncued=ci["n_uncued"],
                            n_uncued_fired=ci["n_uncued_fired"]))
    return results


def print_pattern_completion(intact, ablated):
    print(f"\n{'='*72}")
    print("CA3 PATTERN COMPLETION  (auto-association: partial cue -> full pattern)")
    print(f"{'='*72}")
    print(f"  {'cue%':>6s} {'grp':>4s} | {'intact':>18s} | {'ablated (sup_local=0)':>22s}")
    print(f"  {'':>6s} {'':>4s} | {'compl':>7s} {'recall':>7s} base | {'compl':>7s} {'recall':>7s}")
    abl = {r["group"]: r for r in ablated} if ablated else {}
    passes = 0
    n_valid = 0
    for r in intact:
        a = abl.get(r["group"], {})
        ci, ai = r["completion"], a.get("completion", float("nan"))
        if np.isnan(ci):                       # full-cue row: nothing to complete
            print(f"  {r['cue_frac']*100:5.0f}% {r['group']:>4d} | "
                  f"{'   n/a':>7s} {r['cue_recall']:7.2f} {'  - ':>4s} | "
                  f"{'   n/a':>7s} {a.get('cue_recall', float('nan')):7.2f}")
            continue
        n_valid += 1
        # auto-association signature: substantial completion intact, near-zero ablated
        good = (ci - r["completion_baseline"] > 0.3) and (np.isnan(ai) or ci - ai > 0.3)
        passes += int(good)
        mark = "  <-" if good else ""
        print(f"  {r['cue_frac']*100:5.0f}% {r['group']:>4d} | "
              f"{ci:7.2f} {r['cue_recall']:7.2f} {r['completion_baseline']:4.2f} | "
              f"{ai:7.2f} {a.get('cue_recall', float('nan')):7.2f}{mark}")
    if ablated:
        print(f"\n  Auto-association confirmed for {passes}/{n_valid} testable cue levels "
              f"(intact completion >> baseline AND >> ablated).")
        print("  A sharp rise in the intact column as cue% grows, with the ablated")
        print("  column flat near 0, is the recurrent-completion signature (Marr 1971).")
    print(f"{'='*72}")


# ============================================================================
# EC LII/III module  (Phase 1 cortical addition)
# ============================================================================

@dataclass
class ECModule:
    """
    Entorhinal cortex layer II/III — minimal cortical consolidation target.

    Kept as a self-contained dataclass so it can be built optionally (via
    --ec-lii) without touching any existing hippocampal code.

    CA1→EC synapses use static_synapse (fast parallel connect on NEST 3.9.0).
    STDP weight updates are applied by the Python STC hook between SWR events
    via GetConnections(pre_nc, population) + GetStatus/SetStatus (Phase 2).

    Why GetConnections is NOT called at build time
    -----------------------------------------------
    nest.GetConnections(source, target) scans ALL synapses in the kernel to
    find matching source/target pairs.  With 226M hippocampal synapses already
    in the kernel, that scan runs at ~32k syn/s and takes ~2 hours.  Instead,
    we store the pre/post NodeCollections and call GetConnections only when the
    STC hook actually needs weights — at which point we will use
    GetConnections(target=EC_LII) which scans only EC neurons' incoming slots
    (~600k synapses) rather than all 226M.

    Attributes
    ----------
    population  NEST NodeCollection  — EC LII/III stellate cells
    spike_rec   NEST NodeCollection  — spike recorder
    pre_nc      NEST NodeCollection  — CA1 PYR source population (for STC hook)
    N           neuron count
    K_ca1_ec    in-degree of the CA1→EC projection
    w_init      initial synaptic weight
    """
    population    : object   # nest.NodeCollection  — EC LII/III
    spike_rec     : object   # nest.NodeCollection  — EC spike_recorder
    ca1_spike_rec : object   # nest.NodeCollection  — CA1 PYR spike_recorder (for STC)
    pre_nc        : object   # nest.NodeCollection  — CA1 PYR neurons (reference)
    N             : int
    K_ca1_ec   : int
    w_init     : float



# ============================================================================
# Phase 3 — EC Layer V and mPFC modules
# ============================================================================

@dataclass
class ECLVModule:
    """
    Entorhinal cortex Layer V — closes the hippocampo-cortical loop.

    EC LV pyramidal cells (burst-capable, layer-5 type) receive:
      • CA1 PYR input  : direct projection from hippocampus (K=30, static)
      • EC LII input   : within-EC feedforward (K=20, static)
    And project back to:
      • CA3 SUP        : feedback closes the loop (K=5, static_synapse)
      • mPFC           : downstream cortical consolidation target

    Biological references
    ----------------------
    EC LV principal cells receive direct CA1 input (Köhler 1985) and project
    to deep-layer cortex including mPFC (Insausti et al. 1987).  The CA1→LV
    projection is denser than CA1→LII, making EC LV the primary readout of
    hippocampal output.  The LV→CA3 feedback (via the angular bundle) provides
    the cortical re-entry that sustains offline replay without external drive.
    """
    population  : object  # NodeCollection — EC LV pyramids
    spike_rec   : object  # NodeCollection — spike recorder
    N           : int
    K_ca1_lv    : int     # in-degree from CA1 PYR
    K_eclii_lv  : int     # in-degree from EC LII
    w_init      : float
    # Incoming SynapseCollection + boolean mask marking the CA1-sourced
    # entries, both captured at build time. lesion_hippocampus() needs exactly
    # this and cannot afford to look it up later: GetConnections cost grows
    # with total kernel size (30k incoming took 2.8 s on a bare kernel and
    # 47.9 s once DG and the Schaffer STDP set existed), so the same scan at
    # 12% after training is unbounded. Fetching here costs the same scan while
    # the kernel is still small.
    in_conns    : object = None
    ca1_mask    : object = None


@dataclass
class MPFCModule:
    """
    Medial prefrontal cortex — terminal cortical consolidation target.

    Receives convergent input from EC LV.  The weight distribution of
    EC LV → mPFC synapses (tracked via the STC hook analogue in Phase 3)
    represents the final "cortical engram" — the stable long-term memory
    trace that persists after hippocampal lesion.

    Biological references
    ----------------------
    mPFC receives a dense projection from EC LV (Insausti 1993) and shows
    place-cell-like activity during sleep replay (Peyrache et al. 2009).
    Remote memory is maintained by mPFC even after hippocampal lesion
    (Frankland & Bontempi 2005) — the endpoint this module models.
    """
    population  : object  # NodeCollection — mPFC layer-5 pyramids
    spike_rec   : object  # NodeCollection — spike recorder
    N           : int
    K_eclv_mpfc : int     # in-degree from EC LV
    w_init      : float
    INT         : object = None   # NodeCollection — FS interneurons (lateral inhibition)
    spk_int     : object = None
    N_int       : int   = 0

def build_ec_lii(
    ca1_pyr,
    ca1_spike_rec,           # spike_recorder for CA1 PYR (net["spk_pyr"])
    N_ec_lii     : int,
    K_ca1_ec     : int   = 50,
    w_ca1_ec     : float = 0.30,   # K=50 -> 15 mV volley (0.75x threshold)
    delay_ca1_ec : float = 3.0,   # axonal conduction delay CA1→EC [ms]
    delay_jitter : float = 0.0,   # per-synapse jitter (Phase C); 0 = scalar
    delay_jitter_wcomp: float = 1.0,  # weight scale when jitter>0 (rate matching)
    rate_bg      : float = 0.0,   # EC background Poisson drive set to ZERO.
                                   # CA1→EC K=50 inputs at 7.5 Hz provides 375 Hz
                                   # of drive → EC fires at ~5-8 Hz from CA1 alone.
                                   # During SWR bursts only CA1-driven EC neurons
                                   # fire → selective PRP → gradual consolidation.
                                  # (2-6× bio target of 5-15 Hz), causing every EC
                                  # neuron to fire in every SWR window → non-selective
                                  # PRP accumulation.  Target: EC baseline ~8 Hz.
    w_bg         : float = 1.5,
) -> ECModule:
    """
    Create EC LII/III population and wire it to CA1 with static synapses.

    Neuron model
    ------------
    Izhikevich stellate-cell parameters (regular spiking, lightly adapting):
    a=0.02, b=0.2, c=-65, d=6.  Initial membrane potential -65 mV.

    CA1 → EC projection
    -------------------
    Rule         : pairwise_bernoulli  p = K / N_ca1
    Synapse model: static_synapse  ← fast parallel C++ kernel on NEST 3.9.0
    Delay        : 3 ms (hippocampal-entorhinal axonal conduction)

    Why static_synapse, not stdp_synapse
    -------------------------------------
    On NEST 3.9.0 / MN5, ANY Connect() call using stdp_synapse runs at
    ~80-300 synapses/s regardless of connection rule.  static_synapse
    uses the vectorised parallel kernel.  STDP weight updates are applied
    by the Python STC hook between SWR events (Phase 2).

    Why fixed_indegree, not pairwise_bernoulli
    -------------------------------------------
    Three combinations were tested on MN5 before finding what works:

      fixed_indegree  + stdp_synapse   → ~300 syn/s  (stdp serial path)
      pairwise_bern.  + stdp_synapse   → ~86  syn/s  (both problems)
      pairwise_bern.  + static_synapse → hangs ~2h   (O(N_pre×N_post) pair
                                          iteration: 55k×12k = 662M checks)
      fixed_indegree  + static_synapse → <0.1s        ← this one

    fixed_indegree samples K sources per post neuron — O(K × N_post) — and
    never iterates the full pre×post matrix, matching the Schaffer collateral
    pattern that completes 165M synapses in 3s.

    Returns
    -------
    ECModule — holds the synapse collection handle for the STC hook.
    """
    import nest

    t0    = time.perf_counter()
    N_ca1 = len(ca1_pyr)
    K     = min(K_ca1_ec, N_ca1)          # clamp to pre-population size
    n_exp = int(N_ec_lii * K)

    print(f"\n  [ECModule] Building EC LII/III")
    print(f"  [ECModule]   N_ec={N_ec_lii:,}  N_ca1={N_ca1:,}  "
          f"K={K}  expected_synapses={n_exp:,}")

    # ---- Stellate cell (Izhikevich) ----------------------------------------
    EC_LII = nest.Create("izhikevich", N_ec_lii,
                         params=het_params(dict(a=0.02, b=0.2, c=-65.0, d=6.0,
                                                V_m=-65.0, U_m=-13.0, I_e=0.0),
                                           N_ec_lii, np.random.default_rng(21)))
    # Graded excitability, as every hippocampal population already has. With a
    # uniform V_m every cell is identical, so a synchronous volley fires all of
    # them or none -- nothing for lateral inhibition to select between, and no
    # sparse code can form however strong that inhibition is.
    nest.SetStatus(EC_LII, "V_m",
                   np.random.default_rng(11).normal(-65.0, 4.0, N_ec_lii)
                   .clip(-75, -55).tolist())
    # PERSISTENT excitability spread. V_m heterogeneity above is only a
    # transient -- every initial condition relaxes to the same rest (-70), which
    # is why adding it left EC LII at 87% active. I_e shifts the rest/threshold
    # gap itself (gap = 25*sqrt(0.64 - 0.16*I_e)), so with I_e ~ N(0.0, 1.5) the
    # gap spans ~15-24 mV and a near-threshold volley recruits only the
    # excitable tail.
    nest.SetStatus(EC_LII, "I_e",
                   np.random.default_rng(21).normal(0.0, 1.5, len(EC_LII))
                   .clip(-3.0, 3.0).tolist())

    # ---- Tonic background drive --------------------------------------------
    bg = nest.Create("poisson_generator", N_ec_lii, params={"rate": float(rate_bg)})
    nest.Connect(bg, EC_LII, conn_spec="one_to_one",
                 syn_spec={"weight": float(w_bg), "delay": 1.0})

    # ---- CA1 → EC : fixed_indegree + static_synapse ------------------------
    # Each EC neuron receives exactly K inputs drawn randomly from CA1.
    # O(K × N_post) — same algorithm as Schaffer collaterals, confirmed fast.
    # STDP updates applied by Python STC hook (Phase 2).
    t_conn = time.perf_counter()
    nest.Connect(
        ca1_pyr,
        EC_LII,
        conn_spec={"rule": "fixed_indegree", "indegree": K},
        syn_spec={"synapse_model": "static_synapse",
                  "weight": float(w_ca1_ec) * (delay_jitter_wcomp if delay_jitter>0 else 1.0),
                  "delay":  jittered_delay(delay_ca1_ec, delay_jitter)},
    )
    dt_conn = time.perf_counter() - t_conn

    # Do NOT call GetConnections here — it scans all 226M synapses in the
    # kernel at ~32k syn/s (2+ hours).  Phase 2 STC hook will use
    # GetConnections(target=EC_LII) which only scans EC neurons' ~600k
    # incoming slots.  Store NodeCollections for that deferred call.
    n_actual = int(N_ec_lii * K)   # exact for fixed_indegree
    rate     = n_actual / max(dt_conn, 1e-6)

    print(f"  [ECModule] CA1→EC static: {n_actual:,} synapses  "
          f"in {dt_conn:.2f}s  ({rate:,.0f} syn/s)")
    if rate < 1_000_000:
        print(f"  [ECModule] WARNING: {rate:,.0f} syn/s — expected >1M syn/s for "
              f"fixed_indegree + static_synapse on this build.")

    # ---- Spike recorder ----------------------------------------------------
    spk_ec = nest.Create("spike_recorder")
    nest.Connect(EC_LII, spk_ec)

    print(f"  [ECModule] Total EC build: {time.perf_counter()-t0:.1f}s")

    return ECModule(
        population    = EC_LII,
        spike_rec     = spk_ec,
        ca1_spike_rec = ca1_spike_rec,
        pre_nc        = ca1_pyr,
        N             = N_ec_lii,
        K_ca1_ec      = K,
        w_init        = w_ca1_ec,
    )


# ============================================================================
# Phase 2 — Python STC (Synaptic Tagging and Capture) hook
# ============================================================================

@dataclass
class STCHook:
    """
    Between-SWR synaptic tagging and capture state for the CA1→EC projection.

    Called once after each SWR simulation epoch via run_stc_hook().
    Maintains per-synapse tag strength and a per-EC-neuron PRP pool across
    calls, then applies L-LTP weight capture when PRP threshold is crossed.

    Why this is in Python, not NEST
    --------------------------------
    NEST 3.9.0 on MN5 serialises all stdp_synapse Connect() calls regardless
    of rule — see build_ec_lii() docstring.  Python-level STDP using
    GetStatus / SetStatus on a pre-fetched SynapseCollection is both faster
    and gives full control over the two-timescale (tag + PRP) logic.

    Algorithm (Frey & Morris 1997 / Redondo & Morris 2011)
    -------------------------------------------------------
    After each SWR epoch [t_swr_start, t_swr_end]:

    1. GetConnections(target=EC_LII)  — only scans EC's ~600k incoming slots
    2. Identify coincident CA1→EC pairs: CA1 pre fires before EC post within
       the STDP window → LTP tag;  post before pre → LTD tag.
    3. Tag strength decays exponentially with sim time (tau_tag).
    4. PRP pool per EC neuron accumulates with each SWR activation.
       When PRP_pool[i] ≥ PRP_threshold: all tagged synapses onto neuron i
       capture to L-LTP (permanent weight increase up to w_max).
    5. Weight changes written back via SetStatus.

    Attributes stored between calls
    --------------------------------
    conns        SynapseCollection — fetched once, reused every call
    w            float32 array     — current weights (mirrors NEST state)
    tag          float32 array     — per-synapse tag strength (0 = untagged)
    tag_time_ms  float32 array     — sim time when tag was last set
    prp_pool     float32 array     — per-EC-neuron PRP accumulation
    ltp_done     bool array        — synapses that have captured to L-LTP
    post_idx     int32 array       — EC neuron index for each synapse (for PRP)
    n_calls      int               — number of SWR events processed so far
    history      list[dict]        — per-event summary (for HDF5 export)
    """
    conns         : object             # nest.SynapseCollection
    w             : np.ndarray         # [n_syn] float32
    tag           : np.ndarray         # [n_syn] float32
    tag_time_ms   : np.ndarray         # [n_syn] float32
    prp_pool      : np.ndarray         # [n_ec]  float32  — per-EC-neuron PRP (units: SWR events)
    ltp_done      : np.ndarray         # [n_syn] bool
    post_idx      : np.ndarray         # [n_syn] int32  (EC neuron index for PRP)
    pre_ids       : np.ndarray         # [n_syn] int64  — source GIDs (cached at build)
    post_ids_g    : np.ndarray         # [n_syn] int64  — target GIDs (cached at build)
    ec_gids       : np.ndarray         # [n_ec]  int64  — EC neuron GIDs (for PRP lookup)
    # ---- Structural plasticity (3rd timescale) ---------------------------------
    # Analogy: after struct_threshold L-LTP epochs, a synapse undergoes spine
    # enlargement (Bhatt et al. 2009): w_max raised 1.5→2.0 and weight boosted.
    # This models AMPA-receptor clustering + actual spine volume increase that
    # makes long-term memory effectively permanent on a days-weeks timescale.
    struct_count  : np.ndarray = None  # [n_syn] int16 — cumulative L-LTP epochs
    struct_done   : np.ndarray = None  # [n_syn] bool  — structurally potentiated
    n_calls       : int = 0
    history       : list = None
    # baseline weight, so the L-LTP ceiling can be expressed RELATIVE to it
    # (an absolute ceiling re-saturates the cortex whenever w_ca1_ec is rescaled)
    w_init        : float = 1.0

    def __post_init__(self):
        if self.history is None:
            self.history = []
        # struct arrays initialised lazily in build_stc_hook (need n_syn)



def build_ec_lv(
    ca1_pyr,
    ec_lii_pop,           # ECModule.population — EC LII/III
    ca3_sup,              # CA3 SUP NodeCollection — receives feedback
    N_ec_lv   : int   = None,   # defaults to 60% of EC LII size
    K_ca1_lv  : int   = 30,     # CA1 → EC LV in-degree (dense, Köhler 1985)
    K_eclii_lv: int   = 20,     # EC LII → EC LV in-degree (feedforward)
    K_lv_ca3  : int   = 5,      # EC LV → CA3 SUP in-degree (feedback)
    w_ca1_lv  : float = 0.45,   # K=30 -> 13.5 mV volley (0.68x threshold)
    w_eclii_lv: float = 0.8,    # mV  — within-EC feedforward
    w_lv_ca3  : float = 0.6,    # mV  — feedback to CA3 (modest; avoids runaway)
    delay_ca1  : float = 3.0,   # ms  — CA1→EC conduction
    delay_eclii: float = 2.0,   # ms  — within-EC
    delay_ca3  : float = 5.0,   # ms  — EC→CA3 feedback (longer, angular bundle)
    delay_jitter: float = 0.0,  # per-synapse jitter (Phase C); 0 = scalar
    delay_jitter_wcomp: float = 1.0,  # weight scale when jitter>0 (rate matching)
) -> "ECLVModule":
    """
    Build EC Layer V population and wire it into the hippocampo-cortical loop.

    Network motif
    -------------
    CA1 PYR ──[K=30, w=1.2]──► EC LV ──[K=5, w=0.6]──► CA3 SUP (feedback)
    EC LII  ──[K=20, w=0.8]──► EC LV
    EC LV   ──[K=K_eclv_mpfc]──► mPFC  (wired in build_mpfc)

    EC LV uses Izhikevich layer-5 / intrinsic-burst parameters:
      a=0.02, b=0.2, c=-55.0, d=4.0, I_e=3.0
    This gives low-frequency bursting (~3–8 Hz baseline), matching
    in-vivo recordings of EC LV principal cells during NREM sleep
    (Hahn et al. 2012).

    The EC LV → CA3 feedback uses static_synapse (fast connect).
    Weight is intentionally sub-maximal (0.6 mV) so that feedback
    alone cannot drive CA3 — it only amplifies ongoing replay activity.
    """
    import nest, time as _time

    t0 = _time.perf_counter()
    N_lii = len(ec_lii_pop)
    N_lv  = N_ec_lv if N_ec_lv is not None else max(10, int(N_lii * 0.60))
    N_ca1 = len(ca1_pyr)
    N_ca3 = len(ca3_sup)

    print(f"\n  [ECLVModule] Building EC Layer V")
    print(f"  [ECLVModule]   N_lv={N_lv:,}  N_ca1={N_ca1:,}  N_lii={N_lii:,}  N_ca3={N_ca3:,}")

    # Izhikevich layer-5 intrinsic-burst params
    EC_LV = nest.Create("izhikevich", N_lv,
                        params=het_params(dict(a=0.02, b=0.2, c=-55.0, d=4.0,
                                               V_m=-64.7, U_m=-13.0, I_e=3.0),
                                          N_lv, np.random.default_rng(22)))
    nest.SetStatus(EC_LV, "V_m",
                   np.random.default_rng(12).normal(-64.7, 4.0, N_lv)
                   .clip(-75, -55).tolist())
    # PERSISTENT excitability spread. V_m heterogeneity above is only a
    # transient -- every initial condition relaxes to the same rest (-70), which
    # is why adding it left EC LII at 87% active. I_e shifts the rest/threshold
    # gap itself (gap = 25*sqrt(0.64 - 0.16*I_e)), so with I_e ~ N(3.0, 1.5) the
    # gap spans ~15-24 mV and a near-threshold volley recruits only the
    # excitable tail.
    nest.SetStatus(EC_LV, "I_e",
                   np.random.default_rng(22).normal(3.0, 1.5, len(EC_LV))
                   .clip(0.0, 6.0).tolist())

    # CA1 → EC LV  (fixed_indegree, static — fast connect)
    K1 = min(K_ca1_lv, N_ca1)
    t_c = _time.perf_counter()
    nest.Connect(ca1_pyr, EC_LV,
                 conn_spec={"rule": "fixed_indegree", "indegree": K1},
                 syn_spec={"synapse_model": "static_synapse",
                           "weight": float(w_ca1_lv) * (delay_jitter_wcomp if delay_jitter>0 else 1.0), "delay": jittered_delay(delay_ca1, delay_jitter)})
    print(f"  [ECLVModule] CA1→LV: {N_lv*K1:,} synapses in {_time.perf_counter()-t_c:.2f}s")

    # EC LII → EC LV  (within-EC feedforward)
    K2 = min(K_eclii_lv, N_lii)
    t_c = _time.perf_counter()
    nest.Connect(ec_lii_pop, EC_LV,
                 conn_spec={"rule": "fixed_indegree", "indegree": K2},
                 syn_spec={"synapse_model": "static_synapse",
                           "weight": float(w_eclii_lv) * (delay_jitter_wcomp if delay_jitter>0 else 1.0), "delay": jittered_delay(delay_eclii, delay_jitter)})
    print(f"  [ECLVModule] ECLII→LV: {N_lv*K2:,} synapses in {_time.perf_counter()-t_c:.2f}s")

    # EC LV → CA3 SUP feedback  (closes the hippocampo-cortical loop)
    K3 = min(K_lv_ca3, N_lv)
    t_c = _time.perf_counter()
    nest.Connect(EC_LV, ca3_sup,
                 conn_spec={"rule": "fixed_indegree", "indegree": K3},
                 syn_spec={"synapse_model": "static_synapse",
                           "weight": float(w_lv_ca3), "delay": float(delay_ca3)})
    print(f"  [ECLVModule] LV→CA3: {N_ca3*K3:,} synapses in {_time.perf_counter()-t_c:.2f}s")

    spk_lv = nest.Create("spike_recorder")
    nest.Connect(EC_LV, spk_lv)

    # Cache the incoming synapses now, for lesion_hippocampus() (see the
    # ECLVModule docstring: this scan gets dramatically more expensive once DG
    # and the Schaffer STDP set are in the kernel).
    t_c = _time.perf_counter()
    in_conns = nest.GetConnections(target=EC_LV)
    _src = np.asarray(in_conns.get("source"), dtype=np.int64)
    ca1_mask = np.isin(_src, np.asarray(ca1_pyr.tolist(), dtype=np.int64))
    print(f"  [ECLVModule] cached {len(_src):,} incoming for lesion "
          f"({int(ca1_mask.sum()):,} from CA1) in {_time.perf_counter()-t_c:.2f}s")

    print(f"  [ECLVModule] Total build: {_time.perf_counter()-t0:.1f}s")

    return ECLVModule(
        population  = EC_LV,
        spike_rec   = spk_lv,
        N           = N_lv,
        K_ca1_lv    = K1,
        K_eclii_lv  = K2,
        w_init      = w_ca1_lv,
        in_conns    = in_conns,
        ca1_mask    = ca1_mask,
    )


def build_mpfc(
    ec_lv_pop,
    N_mpfc       : int   = None,   # defaults to 20% of EC LV size
    K_eclv_mpfc  : int   = 20,     # EC LV → mPFC in-degree
    # A synchronous EC LV volley must be SUBTHRESHOLD on its own: K=20 x 1.0 =
    # 20 mV was exactly the rest->threshold gap, so every mPFC cell fired ~1.5 ms
    # before feedback inhibition could arbitrate, and no subset could ever be
    # selected. At 0.5 the volley is ~10 mV, so only cells whose V_m
    # heterogeneity or recurrent input carries them over will fire -- the same
    # recipe that makes the DG sparse (pp_weight subthreshold + feedback
    # inhibition + graded V_m).
    # Raised once EC LV became sparse. The volley arithmetic (K x w vs the 20 mV
    # gap) only applies to SYNCHRONOUS arrival; with EC LV at 6.3 Hz and 62%
    # active the ~15 spikes a cell receives are spread over the 120 ms window and
    # each EPSP decays before the next lands, so peak depolarisation is far below
    # K x w. Empirically mPFC sat at 2% active / 0.12 Hz with mpfc_int at 0.00 Hz
    # -- excitation-starved, not inhibition-crushed.
    w_eclv_mpfc  : float = 1.8,    # mV
    delay_lv_mpfc: float = 8.0,    # ms — longer cortico-cortical delay
    delay_jitter : float = 0.0,    # per-synapse jitter (Phase C); 0 = scalar
    delay_jitter_wcomp: float = 1.0,  # weight scale when jitter>0 (rate matching)
    # Lateral inhibition — the same k-winners-take-all motif the DG uses
    # (GC->basket->GC) to turn dense input into a sparse code. Without it every
    # mPFC cell fires on every SWR, so every EC LV->mPFC synapse co-activates
    # and the associative hook potentiates ALL of them uniformly: association
    # without specificity, i.e. no engram (measured final weight std 0.0018).
    # With it, only the most-driven mPFC cells win each replay event, so
    # different replayed patterns recruit different mPFC subsets.
    lateral_inhibition: bool = True,
    # Recurrent EXCITATORY mPFC->mPFC collaterals. Without these the cortex is
    # purely feedforward (EC LV -> mPFC, plus an inhibitory mPFC<->INT loop), so
    # a partial cortical cue can never reactivate the rest of a pattern -- there
    # is nothing to complete it with. CA3 passes the completion test only
    # because of its sup_local collaterals; this is the cortical equivalent, and
    # it is what a hippocampus-independent (remote) memory would have to use.
    recurrent: bool = False,
    # Recurrent excitation must be able to sustain activity with the
    # hippocampus removed (Test 3). At 0.6 the volley is 12 mV against a 20 mV
    # gap -- subthreshold even before inhibition. 0.9 gives 18 mV, near enough
    # that the I_e-excitable tail can carry a reactivation.
    K_rec: int = 20, w_rec: float = 0.9, delay_rec: float = 4.0,
    int_frac     : float = 0.20,   # FS interneurons as a fraction of mPFC
    w_mpfc_ei    : float = 2.5,    # mPFC -> INT   (mirrors w_gc_basket)
    # Rebalanced for the sparse-cortex regime. At -7.0 the feedback volley was
    # 84 mV against 29 mV of total excitation (2.9x), which drove mPFC to 2%
    # active / 0.12 Hz once EC LV became sparse -- ~2 cells, too few to form an
    # assembly. -3.5 gives 42 mV, still dominant but leaving a workable subset.
    w_mpfc_ie    : float = -3.5,   # INT -> mPFC   (mirrors w_basket_gc)
) -> "MPFCModule":
    """
    Build mPFC population receiving EC LV input.

    mPFC uses layer-5 prefrontal parameters (regular spiking, high adaptation):
      a=0.02, b=0.2, c=-65.0, d=8.0
    This gives sparse, burst-resistant firing (~2–5 Hz) matching in-vivo
    mPFC recordings during NREM sleep (Peyrache et al. 2009).

    The EC LV → mPFC weight distribution is the 'cortical engram':
    after repeated hippocampal replay + EC LV consolidation, these weights
    encode the replayed sequence in a hippocampus-independent form.
    """
    import nest, time as _time

    t0 = _time.perf_counter()
    N_lv  = len(ec_lv_pop)
    N_pfc = N_mpfc if N_mpfc is not None else max(10, int(N_lv * 0.20))
    K     = min(K_eclv_mpfc, N_lv)

    print(f"\n  [MPFCModule] Building mPFC")
    print(f"  [MPFCModule]   N_mpfc={N_pfc:,}  N_lv={N_lv:,}  K={K}")

    # Izhikevich layer-5 prefrontal: regular spiking, strongly adapting
    MPFC = nest.Create("izhikevich", N_pfc,
                       params=het_params(dict(a=0.02, b=0.2, c=-65.0, d=8.0,
                                              V_m=-65.0, U_m=-13.0, I_e=0.0),
                                         N_pfc, np.random.default_rng(23)))
    nest.SetStatus(MPFC, "V_m",
                   np.random.default_rng(13).normal(-65.0, 4.0, N_pfc)
                   .clip(-75, -55).tolist())
    # PERSISTENT excitability spread. V_m heterogeneity above is only a
    # transient -- every initial condition relaxes to the same rest (-70), which
    # is why adding it left EC LII at 87% active. I_e shifts the rest/threshold
    # gap itself (gap = 25*sqrt(0.64 - 0.16*I_e)), so with I_e ~ N(0.0, 1.5) the
    # gap spans ~15-24 mV and a near-threshold volley recruits only the
    # excitable tail.
    nest.SetStatus(MPFC, "I_e",
                   np.random.default_rng(23).normal(0.0, 1.5, len(MPFC))
                   .clip(-3.0, 3.0).tolist())

    # EC LV → mPFC
    t_c = _time.perf_counter()
    nest.Connect(ec_lv_pop, MPFC,
                 conn_spec={"rule": "fixed_indegree", "indegree": K},
                 syn_spec={"synapse_model": "static_synapse",
                           "weight": float(w_eclv_mpfc) * (delay_jitter_wcomp if delay_jitter>0 else 1.0),
                           "delay": jittered_delay(delay_lv_mpfc, delay_jitter)})
    print(f"  [MPFCModule] LV→mPFC: {N_pfc*K:,} synapses in {_time.perf_counter()-t_c:.2f}s")

    # ---- Lateral inhibition: mPFC <-> FS interneurons --------------------
    MPFC_INT, spk_int, N_int = None, None, 0
    if lateral_inhibition:
        N_int = max(10, int(N_pfc * int_frac))
        MPFC_INT = nest.Create("izhikevich", N_int,
                               params=het_params(dict(a=0.10, b=0.2, c=-65.0, d=2.0,
                                                      V_m=-65.0, U_m=-13.0, I_e=0.0),
                                                 N_int, np.random.default_rng(24)))
        # In-degrees must be SCALE-INVARIANT, like every other in-degree in this
        # model. K_ie was originally N_int//2, which grows with the population:
        # 12 at 1% but 144 at 12%. Because the volley is synchronous, the
        # instantaneous hyperpolarisation is K_ie*|w_mpfc_ie| -- 84 mV at 1%
        # (fine, mPFC 3.6 Hz) but 1008 mV at 12%, which is PATHOLOGICAL for an
        # Izhikevich neuron: driven far enough negative the 0.04*v^2 term
        # dominates and depolarises the cell, so excess inhibition causes
        # runaway firing instead of silence. Measured on an isolated RS cell
        # with no excitation, a periodic synchronous volley alone gives:
        #     84 mV -> 0 Hz,  200 mV -> 0 Hz,  500 mV -> 49 Hz,  1008 mV -> 49 Hz
        # and in the 12% network mPFC saturated at 302 Hz with all 1,440 cells
        # within 3 spikes of each other. Fixed values reproduce the validated
        # 1% configuration at every scale.
        K_ei = max(1, min(50, N_pfc))
        K_ie = max(1, min(12, N_int))
        if K_ie * abs(w_mpfc_ie) > 200.0:
            warnings.warn(
                f"[MPFCModule] synchronous inhibitory amplitude "
                f"K_ie*|w| = {K_ie*abs(w_mpfc_ie):.0f} mV exceeds the ~200 mV "
                f"point where Izhikevich neurons start spiking FROM inhibition "
                f"(quadratic term). Reduce K_ie or w_mpfc_ie.",
                RuntimeWarning, stacklevel=2)
        fixed_connect(MPFC,     MPFC_INT, K_ei, w_mpfc_ei, 1.5, compensate=False)
        fixed_connect(MPFC_INT, MPFC,     K_ie, w_mpfc_ie, 1.5)
        spk_int = nest.Create("spike_recorder")
        nest.Connect(MPFC_INT, spk_int)
        print(f"  [MPFCModule] lateral inhibition: {N_int} FS cells  "
              f"E->I K={K_ei} w={w_mpfc_ei}  I->E K={K_ie} w={w_mpfc_ie}")
    else:
        print("  [MPFCModule] lateral inhibition DISABLED — mPFC will fire as a "
              "whole population (no sparse code, no selective engram)")

    if recurrent:
        Kr = max(1, min(K_rec, N_pfc - 1))
        fixed_connect(MPFC, MPFC, Kr, w_rec, delay_rec)
        print(f"  [MPFCModule] recurrent mPFC->mPFC: K={Kr} w={w_rec} "
              f"delay={delay_rec} ms  (substrate for cortical pattern completion)")

    spk_mpfc = nest.Create("spike_recorder")
    nest.Connect(MPFC, spk_mpfc)
    print(f"  [MPFCModule] Total build: {_time.perf_counter()-t0:.1f}s")

    return MPFCModule(
        population  = MPFC,
        spike_rec   = spk_mpfc,
        N           = N_pfc,
        K_eclv_mpfc = K,
        w_init      = w_eclv_mpfc,
        INT         = MPFC_INT,
        spk_int     = spk_int,
        N_int       = N_int,
    )

# ============================================================================
# Cortical association build-up  (EC LV -> mPFC Hebbian hook)
# ============================================================================
#
# build_mpfc() wires EC LV -> mPFC with static_synapse and nothing ever
# modified those weights, so the "cortical engram" its docstring describes
# could never form: the mPFC was a passive readout. This hook makes the
# association actually build up across SWR epochs.
#
# Rule: replay-gated Hebbian co-activation. In each SWR window, a synapse is
# potentiated when its EC LV source AND its mPFC target both fired -- i.e. the
# cortico-cortical link is strengthened exactly when hippocampal replay drove
# both ends together. Synapses whose post fired without the pre are weakly
# depressed (heterosynaptic competition), which is what makes the final weight
# distribution BIMODAL -- an engram -- rather than uniformly drifting upward.
#
# This is deliberately simpler than the CA1->EC STC hook: no tag/PRP cascade,
# because the cortical association is the SLOW integrator here -- it should
# accumulate gradually over many replay events rather than gate on a
# capture threshold.

@dataclass
class MPFCAssocHook:
    """State for the EC LV -> mPFC associative projection."""
    conns    : object        # nest.SynapseCollection
    w        : np.ndarray    # current weights
    w_init   : float
    pre_idx  : np.ndarray    # source index into EC LV population
    post_idx : np.ndarray    # target index into mPFC population
    n_calls  : int = 0
    history  : list = None


def build_mpfc_assoc_hook(mpfc_module, eclv_module) -> MPFCAssocHook:
    """Cache the EC LV -> mPFC synapses once (same pattern as build_stc_hook)."""
    import nest
    t0 = time.perf_counter()
    print("\n  [mPFCAssoc] Fetching EC LV->mPFC synapse collection...")
    # Restrict the collection to EC LV sources. GetConnections(target=mPFC) also
    # returns the INHIBITORY INT->mPFC lateral-inhibition synapses; letting the
    # Hebbian rule touch those would clamp them to w_min on the depression
    # branch, flipping them positive and destroying the lateral inhibition that
    # makes the engram selective in the first place.
    conns_all = nest.GetConnections(target=mpfc_module.population)
    src_all   = np.array(nest.GetStatus(conns_all, "source"), dtype=np.int64)
    lv_ids    = set(eclv_module.population.tolist())
    keep      = np.array([g in lv_ids for g in src_all], dtype=bool)
    n_drop    = int((~keep).sum())
    conns = nest.GetConnections(source=eclv_module.population,
                                target=mpfc_module.population)
    n = len(conns)
    w = np.array(nest.GetStatus(conns, "weight"), dtype=np.float32)
    src = np.array(nest.GetStatus(conns, "source"), dtype=np.int64)
    tgt = np.array(nest.GetStatus(conns, "target"), dtype=np.int64)
    lv_map  = {g: i for i, g in enumerate(eclv_module.population.tolist())}
    pfc_map = {g: i for i, g in enumerate(mpfc_module.population.tolist())}
    pre_idx  = np.array([lv_map.get(g, -1)  for g in src], dtype=np.int32)
    post_idx = np.array([pfc_map.get(g, -1) for g in tgt], dtype=np.int32)
    print(f"  [mPFCAssoc] {n:,} EC LV->mPFC synapses cached in "
          f"{time.perf_counter()-t0:.2f}s (excluded {n_drop:,} non-EC-LV inputs, "
          f"incl. lateral inhibition)")
    return MPFCAssocHook(conns=conns, w=w, w_init=float(w[0]) if n else 1.0,
                         pre_idx=pre_idx, post_idx=post_idx, history=[])


def build_mpfc_recurrent_hook(mpfc_module) -> MPFCAssocHook:
    """Cache the recurrent mPFC->mPFC synapses so they can learn.

    These are what a hippocampus-independent memory would actually be stored
    IN: after the hippocampus is removed, only intra-cortical weights remain to
    reactivate a pattern from a partial cue. Runs through the same Hebbian rule
    as the feedforward hook by passing mPFC as BOTH pre and post.

    Masking is essential: GetConnections(target=mPFC) also returns EC LV inputs
    and the INHIBITORY INT->mPFC synapses. Letting the depression branch touch
    the inhibitory ones would clamp them to w_min and flip them positive,
    destroying the lateral inhibition (the bug already fixed once for the
    feedforward hook).
    """
    import nest
    t0 = time.perf_counter()
    print("\n  [mPFCRec] Fetching recurrent mPFC->mPFC synapses...", flush=True)
    conns_all = nest.GetConnections(target=mpfc_module.population)
    src_all = np.array(nest.GetStatus(conns_all, "source"), dtype=np.int64)
    tgt_all = np.array(nest.GetStatus(conns_all, "target"), dtype=np.int64)
    w_all   = np.array(nest.GetStatus(conns_all, "weight"), dtype=np.float32)
    pfc_ids = np.array(mpfc_module.population.tolist(), dtype=np.int64)
    keep    = np.isin(src_all, pfc_ids)          # mPFC -> mPFC only
    pfc_map = {int(g): i for i, g in enumerate(pfc_ids)}
    pre_idx  = np.array([pfc_map.get(int(g), -1) for g in src_all], dtype=np.int32)
    post_idx = np.array([pfc_map.get(int(g), -1) for g in tgt_all], dtype=np.int32)
    pre_idx[~keep] = -1                          # gate updates off non-recurrent
    post_idx[~keep] = -1
    print(f"  [mPFCRec] {int(keep.sum()):,} recurrent of {len(src_all):,} "
          f"mPFC-incoming in {time.perf_counter()-t0:.1f}s", flush=True)
    return MPFCAssocHook(conns=conns_all, w=w_all,
                         w_init=float(w_all[keep].mean()) if keep.any() else 1.0,
                         pre_idx=pre_idx, post_idx=post_idx, history=[])


def run_mpfc_assoc_hook(hook, eclv_module, mpfc_module,
                        t_swr_start, t_swr_end,
                        A_assoc=0.02, A_hetero=0.004,
                        # w_max caps the volley below threshold: at K=20 a weight
                        # of 2.0 would reach 40 mV and undo the sparsening that
                        # the reduced w_eclv_mpfc buys. 0.8 -> at most ~16 mV.
                        w_max=2.4, w_min=0.05):
    """Strengthen EC LV -> mPFC where replay co-activated both ends.

    Called once per SWR event, after nest.Simulate() for that epoch.
    """
    import nest

    def fired(spk_rec, pop):
        ev = nest.GetStatus(spk_rec, "events")[0]
        t  = np.asarray(ev["times"]); s = np.asarray(ev["senders"])
        act = np.unique(s[(t >= t_swr_start) & (t <= t_swr_end)])
        idx = {g: i for i, g in enumerate(pop.tolist())}
        m = np.zeros(len(pop), dtype=bool)
        for g in act:
            i = idx.get(int(g))
            if i is not None:
                m[i] = True
        return m

    pre_fired  = fired(eclv_module.spike_rec, eclv_module.population)
    post_fired = fired(mpfc_module.spike_rec, mpfc_module.population)

    valid = (hook.pre_idx >= 0) & (hook.post_idx >= 0)
    pre_a  = np.zeros(len(hook.w), dtype=bool)
    post_a = np.zeros(len(hook.w), dtype=bool)
    pre_a[valid]  = pre_fired[hook.pre_idx[valid]]
    post_a[valid] = post_fired[hook.post_idx[valid]]

    # `valid` on BOTH branches: the depression branch must never touch a
    # synapse whose source is not EC LV (see build_mpfc_assoc_hook).
    both   = valid & pre_a & post_a    # Hebbian: co-activation -> potentiate
    hetero = valid & post_a & ~pre_a   # post without pre -> weak depression
    hook.w[both]   = np.minimum(hook.w[both]   + A_assoc,  w_max)
    hook.w[hetero] = np.maximum(hook.w[hetero] - A_hetero, w_min)
    # .tolist() here allocated ~1.5M Python floats per call, 64 calls per run --
    # this, not the (already vectorised) spike lookup, was the dominant cost.
    # NEST accepts a numpy array directly.
    nest.SetStatus(hook.conns, "weight", hook.w)

    hook.n_calls += 1
    # "associated" = potentiated in at least half the replay events so far.
    # Scales with run length, unlike an absolute threshold near w_max, which is
    # unreachable in a short run and reports a misleading 0%.
    thr = hook.w_init + 0.5 * hook.n_calls * A_assoc
    # Selectivity is what distinguishes an ENGRAM from uniform drift: a real
    # engram potentiates a SUBSET, so the weight distribution spreads out. CV
    # near 0 means every synapse moved together -- association without
    # specificity (see the note in build_mpfc_assoc_hook's section header).
    w_cv = float(hook.w.std() / max(hook.w.mean(), 1e-9))
    rec = dict(t_swr_start=float(t_swr_start),
               n_coactive=int(both.sum()), n_hetero=int(hetero.sum()),
               w_mean=float(hook.w.mean()), w_max_seen=float(hook.w.max()),
               w_cv=w_cv,
               n_associated=int((hook.w >= thr).sum()),
               frac_associated=float((hook.w >= thr).mean()))
    hook.history.append(rec)
    return rec


# ============================================================================
# Schaffer STDP  (Phase C step 2 — selecting delay-matched paths)
# ============================================================================
#
# Phase C step 1 showed that delay heterogeneity ALONE does not carry the CA3
# timing code downstream (CA3 separation +0.20, everything cortical within
# +/-0.06, across a +/-60% range of cortical firing rate). That is the expected
# result: in Izhikevich (2006) it is STDP that SELECTS delay-matched paths --
# delays are the substrate, STDP is the mechanism.
#
# This hook applies pair-based STDP to the Schaffer collateral, using each
# synapse's OWN delay:
#
#     dt = (t_post - t_pre) - delay_ij
#
# A synapse is potentiated when its delay MATCHES the pre->post interval, i.e.
# when its spike arrived just in time to help fire the postsynaptic cell. Since
# every CA1 cell has a different random draw of delays, different cells come to
# favour different temporal input patterns -- which is how a timing code gets
# converted into a cell-identity code that downstream rate-based readouts can
# actually see. This is the polychronous-group selection rule.
#
# It is applied at CA3->CA1 rather than at EC LV->mPFC because that is where
# the signal still exists: the timing code is strong in CA3 (+0.20) and already
# gone by CA1 (-0.06), so plasticity further downstream would have nothing to
# learn from.

@dataclass
class SchafferSTDPHook:
    conns    : object
    w        : np.ndarray
    delay    : np.ndarray     # per-synapse delay (ms) — the polychronization term
    pre_g    : np.ndarray
    post_g   : np.ndarray
    w_init   : float
    mask     : np.ndarray = None   # True where source is CA3 SUP (the Schaffer subset)
    n_calls  : int = 0
    history  : list = None


def build_schaffer_stdp_hook(net) -> SchafferSTDPHook:
    """Cache the CA3 SUP -> CA1 synapses, their weights AND their delays."""
    import nest
    t0 = time.perf_counter()
    print("\n  [SchafferSTDP] Fetching CA1-incoming synapses...", flush=True)
    # target= ONLY. Passing BOTH source and target makes NEST scan every synapse
    # in the kernel to find matching pairs -- the same trap documented in
    # build_stc_hook, and it did not finish in 10 min here. Scanning CA1's
    # incoming slots is bounded by CA1's in-degree instead. The CA3->CA1 subset
    # is then masked in numpy, and SetStatus writes the full collection back
    # with unchanged values off-mask (the homeostasis hook does the same).
    conns = nest.GetConnections(target=net["PYR"])
    n_all = len(conns)
    w = np.array(nest.GetStatus(conns, "weight"), dtype=np.float32)
    d = np.array(nest.GetStatus(conns, "delay"),  dtype=np.float32)
    pg = np.array(nest.GetStatus(conns, "source"), dtype=np.int64)
    qg = np.array(nest.GetStatus(conns, "target"), dtype=np.int64)
    ca3 = np.array(net["CA3_SUP"].tolist(), dtype=np.int64)
    mask = np.isin(pg, ca3)
    dm = d[mask]
    print(f"  [SchafferSTDP] {n_all:,} CA1-incoming, {int(mask.sum()):,} from CA3 SUP "
          f"in {time.perf_counter()-t0:.1f}s  delay {dm.min():.1f}-{dm.max():.1f} ms "
          f"(the spread STDP selects on)", flush=True)
    return SchafferSTDPHook(conns=conns, w=w, delay=d, pre_g=pg, post_g=qg,
                            mask=mask,
                            w_init=float(w[mask].mean()) if mask.any() else 0.0,
                            history=[])


def run_schaffer_stdp_hook(hook, net, t_start, t_end,
                           A_plus=0.008, A_minus=0.0055, tau=20.0,
                           w_min=0.001, w_max=None):
    """Delay-aware pair STDP over one SWR window."""
    import nest
    if w_max is None:
        w_max = 5.0 * max(hook.w_init, 1e-6)

    def last_spike(rec):
        """(gids, last_spike_time) as sorted arrays, for vectorised lookup."""
        ev = nest.GetStatus(rec, "events")[0]
        t = np.asarray(ev["times"]); s = np.asarray(ev["senders"], dtype=np.int64)
        m = (t >= t_start) & (t <= t_end)
        t, s = t[m], s[m]
        if not len(t):
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        o = np.argsort(s, kind="stable"); s, t = s[o], t[o]
        uq, idx = np.unique(s, return_index=True)
        ends = np.append(idx[1:], len(s))
        # last spike in the window, matching the STC hook's convention
        vals = np.array([t[st:en].max() for st, en in zip(idx, ends)],
                        dtype=np.float32)
        return uq, vals

    def lookup(gids, vals, want):
        """Vectorised gid -> value; NaN where the gid did not spike.

        Must NOT be a Python dict comprehension over `want`: that is ~1M
        interpreter-level lookups per population per SWR event, which dominated
        the whole run when this hook was first written.
        """
        out = np.full(len(want), np.nan, dtype=np.float32)
        if len(gids):
            pos = np.searchsorted(gids, want)
            pos_c = np.clip(pos, 0, len(gids) - 1)
            hit = gids[pos_c] == want
            out[hit] = vals[pos_c[hit]]
        return out

    pre_g_arr,  pre_v  = last_spike(net["spk_ca3_sup"])
    post_g_arr, post_v = last_spike(net["spk_pyr"])
    if not len(pre_g_arr) or not len(post_g_arr):
        hook.n_calls += 1
        hook.history.append(dict(n_pot=0, n_dep=0, w_mean=float(hook.w.mean()),
                                 w_cv=float(hook.w.std()/max(hook.w.mean(),1e-9))))
        return hook.history[-1]

    tp = lookup(pre_g_arr,  pre_v,  hook.pre_g)
    tq = lookup(post_g_arr, post_v, hook.post_g)
    ok = np.isfinite(tp) & np.isfinite(tq)
    if hook.mask is not None:
        ok &= hook.mask          # only the CA3->CA1 subset is plastic

    dw = np.zeros(len(hook.w), dtype=np.float32)
    # THE polychronization term: subtract each synapse's own conduction delay,
    # so dt is the arrival-vs-firing mismatch, not the raw soma-to-soma interval.
    dt = (tq[ok] - tp[ok]) - hook.delay[ok]
    pot = dt >= 0
    d_ok = np.zeros(ok.sum(), dtype=np.float32)
    d_ok[pot]  =  A_plus  * np.exp(-dt[pot] / tau)
    d_ok[~pot] = -A_minus * np.exp( dt[~pot] / tau)
    dw[ok] = d_ok
    hook.w = np.clip(hook.w + dw, w_min, w_max)
    nest.SetStatus(hook.conns, "weight", hook.w.tolist())

    hook.n_calls += 1
    _m = hook.mask if hook.mask is not None else slice(None)
    _wm = hook.w[_m]
    rec = dict(n_pot=int((dw > 0).sum()), n_dep=int((dw < 0).sum()),
               w_mean=float(_wm.mean()),
               w_cv=float(_wm.std() / max(abs(_wm.mean()), 1e-9)))
    hook.history.append(rec)
    return rec


def lesion_hippocampus(net, ec_module, eclv_module, stc=None):
    """Silence hippocampal output to cortex — the systems-consolidation lesion.

    Zeroes CA1 -> EC LII and CA1 -> EC LV. Everything cortical (EC LII, EC LV,
    mPFC, and the consolidated intra-cortical weights) is left intact, so what
    remains is exactly what a "remote" memory would have to run on
    (Frankland & Bontempi 2005: remote memory survives hippocampal lesion).

    Zeroing the projection rather than deleting neurons keeps the network
    topology and all cached synapse handles valid.

    NEVER call GetConnections(source=, target=) here: filtering on source makes
    NEST scan the whole kernel connection table, which at 12% with the Schaffer
    STDP set present is ~19M+ synapses and took >9 h (it is what killed the
    first Test-3 job). GetConnections(target=) hits only that population's
    incoming slots -- the same fast path build_stc_hook() documents -- and the
    source filtering is then done in numpy. Better still, reuse the handles the
    STC hook already fetched: cost zero.
    """
    import nest
    n = 0
    ca1 = np.asarray(net["PYR"].tolist(), dtype=np.int64)
    for m in (ec_module, eclv_module):
        if m is None:
            continue
        if stc is not None and m is ec_module and getattr(stc, "conns", None) is not None:
            c = stc.conns                       # already fetched, all CA1->EC
            if len(c):
                nest.SetStatus(c, "weight", np.zeros(len(c)))
                n += len(c)
            continue
        if getattr(m, "in_conns", None) is not None and getattr(m, "ca1_mask", None) is not None:
            c, keep = m.in_conns, m.ca1_mask    # cached at build time
        else:
            t0 = time.perf_counter()
            c = nest.GetConnections(target=m.population)
            src = np.asarray(c.get("source"), dtype=np.int64)
            keep = np.isin(src, ca1)
            print(f"  [lesion] scanned {len(src):,} incoming in "
                  f"{time.perf_counter()-t0:.1f}s, {int(keep.sum()):,} from CA1")
        if not keep.any():
            continue
        # A SynapseCollection cannot be sub-indexed by an array, so rewrite the
        # whole weight vector with the CA1 entries zeroed and everything else
        # left at its current value.
        w = np.asarray(c.get("weight"), dtype=float)
        w[keep] = 0.0
        nest.SetStatus(c, "weight", w)
        n += int(keep.sum())
    print(f"  [lesion] CA1->cortex silenced: {n:,} synapses zeroed "
          f"(cortex intact, hippocampus disconnected)")
    return n


def cortical_recall_probe(mpfc_module, pattern_cells, cue_frac,
                          t_start, dur_ms=16.0, rate=2600.0, weight=2.5,
                          win_ms=80.0, rng=None):
    """Cue part of a cortical assembly; measure how much of the REST revives.

    Same logic as the CA3 completion probe, applied to mPFC after the
    hippocampus is gone. Returns (cued, uncued) so the caller can score with
    completion_index() once the simulation has run.
    """
    import nest
    rng = rng or np.random.default_rng(0)
    cells = np.asarray([int(c) for c in pattern_cells], dtype=np.int64)
    n_cue = max(1, int(round(cue_frac * len(cells))))
    cued = np.sort(rng.choice(cells, size=n_cue, replace=False))
    uncued = np.setdiff1d(cells, cued)
    gens = nest.Create("poisson_generator", len(cued), params={
        "rate": float(rate), "start": float(t_start),
        "stop": float(t_start + dur_ms)})
    nest.Connect(gens, nest.NodeCollection([int(c) for c in cued]),
                 conn_spec="one_to_one",
                 syn_spec={"weight": float(weight), "delay": 1.0})
    return cued, uncued


def pattern_discrimination(net, pops, swr_fwd, swr_rev, epoch_ms, n_epochs):
    """Do different replayed patterns recruit different downstream cells?

    This is the operational definition of an engram in this model. For each
    population it collects the set of cells active during the SWR windows of
    every epoch, groups those sets by which pattern that epoch replayed, and
    reports the Jaccard overlap BETWEEN patterns against the overlap WITHIN a
    pattern (the same pattern replayed in different epochs).

        within  high  and  between  low   -> discriminating representation
        within ~= between                 -> no pattern identity downstream

    Reporting `between` alone is not enough: trial-to-trial variability alone
    could make it low. The within-pattern baseline is the control.
    """
    ep_pat = net.get("epoch_pattern") or [0]
    out = {}
    for label, pop, rec in pops:
        gids = np.asarray(pop.tolist(), dtype=np.int64)
        ev   = nest.GetStatus(rec, "events")[0]
        t    = np.asarray(ev["times"]); s = np.asarray(ev["senders"])
        # active-cell set per epoch (union of that epoch's two SWR windows)
        sets = []
        for ep in range(max(1, n_epochs)):
            t0 = ep * epoch_ms
            m = (((t >= t0 + swr_fwd[0]) & (t <= t0 + swr_fwd[1])) |
                 ((t >= t0 + swr_rev[0]) & (t <= t0 + swr_rev[1])))
            sets.append(set(int(x) for x in np.unique(s[m])))

        def jac(a, b):
            u = len(a | b)
            return (len(a & b) / u) if u else float("nan")

        within, between = [], []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                if not sets[i] and not sets[j]:
                    continue
                (within if ep_pat[i] == ep_pat[j] else between).append(
                    jac(sets[i], sets[j]))
        act = [len(x) / max(len(gids), 1) for x in sets if x]

        # --- TIMING code -------------------------------------------------
        # Identity (which cells fired) and timing (WHEN each cell fired) are
        # different codes and must be measured separately: a successful
        # temporal code is invisible to the set-overlap readout above. Per-cell
        # mean spike time within each epoch's SWR windows, correlated across
        # epochs, within-pattern vs between-pattern.
        # Score each SWR window SEPARATELY and average. Pooling both windows
        # into one per-cell mean mixes two events ~300 ms apart, and that offset
        # dominates the correlation -- it diluted CA3's separation from +0.14 to
        # +0.03 when first implemented that way.
        act_gids = np.unique(s) if len(s) else np.array([], dtype=np.int64)
        tw_in, tw_bt = [], []
        for win in (swr_fwd, swr_rev):
            tprofs = []
            for ep in range(max(1, n_epochs)):
                a_ms, b_ms = ep * epoch_ms + win[0], ep * epoch_ms + win[1]
                v = np.full(len(act_gids), np.nan)
                m = (t >= a_ms) & (t <= b_ms)
                tw, sw = t[m], s[m]
                if len(tw):
                    o = np.argsort(sw, kind="stable"); sw_s, tw_s = sw[o], tw[o]
                    uq, idx = np.unique(sw_s, return_index=True)
                    ends = list(idx[1:]) + [len(sw_s)]
                    for g, st, en in zip(uq, idx, ends):
                        k = np.searchsorted(act_gids, g)
                        if k < len(act_gids) and act_gids[k] == g:
                            # relative to WINDOW onset, so windows are comparable
                            v[k] = tw_s[st:en].mean() - a_ms
                tprofs.append(v)
            for i in range(len(tprofs)):
                for j in range(i + 1, len(tprofs)):
                    x, y = tprofs[i], tprofs[j]
                    ok = np.isfinite(x) & np.isfinite(y)
                    if ok.sum() < 4 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
                        continue
                    c = float(np.corrcoef(x[ok], y[ok])[0, 1])
                    (tw_in if ep_pat[i] == ep_pat[j] else tw_bt).append(c)
        t_within  = float(np.mean(tw_in)) if tw_in else float("nan")
        t_between = float(np.mean(tw_bt)) if tw_bt else float("nan")

        out[label] = dict(
            n_cells=len(gids),
            mean_active_frac=float(np.mean(act)) if act else 0.0,
            within=float(np.mean(within)) if within else float("nan"),
            between=float(np.mean(between)) if between else float("nan"),
            t_within=t_within, t_between=t_between,
            t_separation=t_within - t_between,
        )
        out[label]["separation"] = out[label]["within"] - out[label]["between"]
    return out


def print_pattern_discrimination(disc, n_patterns):
    print(f"\n--- Pattern discrimination ({n_patterns} patterns) ---")
    if n_patterns < 2:
        print("  only 1 pattern stored — selectivity is undefined; "
              "re-run with --n-patterns 2 or more")
        return
    print(f"  {'':12s} {'':>8s} {'-- IDENTITY (which cells) --':>30s}   "
          f"{'-- TIMING (when) --':>26s}")
    print(f"  {'population':12s} {'active':>8s} {'within':>9s} {'between':>9s} "
          f"{'sep':>9s}   {'within':>8s} {'between':>8s} {'sep':>8s}")
    for label, d in disc.items():
        print(f"  {label:12s} {d['mean_active_frac']*100:7.1f}% "
              f"{d['within']:9.3f} {d['between']:9.3f} {d['separation']:9.3f}   "
              f"{d.get('t_within', float('nan')):8.3f} "
              f"{d.get('t_between', float('nan')):8.3f} "
              f"{d.get('t_separation', float('nan')):8.3f}")
    # Where does pattern information survive to? Report the deepest population
    # that still discriminates by EITHER code -- the transmission question.
    def _ok(d, key):
        v = d.get(key, float("nan"))
        return (not np.isnan(v)) and v > 0.10
    deepest_id = [k for k, d in disc.items() if _ok(d, "separation")]
    deepest_t  = [k for k, d in disc.items() if _ok(d, "t_separation")]
    print(f"  discriminates by identity : {', '.join(deepest_id) if deepest_id else 'NONE'}")
    print(f"  discriminates by timing   : {', '.join(deepest_t)  if deepest_t  else 'NONE'}")
    cortical = {"EC LII", "EC LV", "mPFC"}
    if cortical & (set(deepest_id) | set(deepest_t)):
        print("  [OK] pattern information reaches CORTEX — an engram substrate exists.")
    elif deepest_id or deepest_t:
        print("  [FLAG] pattern information exists upstream but DIES before cortex.")
        print("         Transmission, not encoding, is the bottleneck.")
    else:
        print("  [FLAG] no population discriminates the patterns by either code.")


def build_stc_hook(ec_module, w_init_override=None) -> STCHook:
    """
    Initialise the STC hook by fetching the CA1→EC SynapseCollection.

    Uses GetConnections(target=EC_LII) which scans only EC's incoming slots
    (~600k synapses, ~0.09s) rather than the full kernel (~2h).
    Called once after build_ec_lii(), before the first nest.Simulate().
    """
    import nest

    t0 = time.perf_counter()
    print("\n  [STCHook] Fetching CA1→EC synapse collection "
          "(one-time scan, reused every epoch)...")

    conns = nest.GetConnections(target=ec_module.population)
    n_syn = len(conns)
    print(f"  [STCHook] {n_syn:,} synapses found in {time.perf_counter()-t0:.2f}s")

    w_arr = np.array(nest.GetStatus(conns, "weight"), dtype=np.float32)
    if w_init_override is not None:
        w_arr[:] = float(w_init_override)

    # Cache source/target GIDs once — reused every epoch, no re-scan needed
    print("  [STCHook] Caching source/target GIDs...")
    t_cache      = time.perf_counter()
    pre_ids_arr  = np.array(nest.GetStatus(conns, "source"), dtype=np.int64)
    post_ids_arr = np.array(nest.GetStatus(conns, "target"), dtype=np.int64)
    print(f"  [STCHook] GID cache done in {time.perf_counter()-t_cache:.2f}s")

    # Map each synapse to its EC neuron index (0-based) for PRP accumulation
    ec_ids       = np.array(ec_module.population.tolist(), dtype=np.int64)
    ec_id_to_idx = {gid: i for i, gid in enumerate(ec_ids)}
    post_idx     = np.array([ec_id_to_idx[gid] for gid in post_ids_arr], dtype=np.int32)

    return STCHook(
        conns        = conns,
        w            = w_arr,
        w_init       = float(w_arr.mean()) if len(w_arr) else 1.0,
        tag          = np.zeros(n_syn, dtype=np.float32),
        tag_time_ms  = np.full(n_syn, -1e9, dtype=np.float32),
        prp_pool     = np.zeros(ec_module.N, dtype=np.float32),
        ltp_done     = np.zeros(n_syn, dtype=bool),
        post_idx     = post_idx,
        pre_ids      = pre_ids_arr,
        post_ids_g   = post_ids_arr,
        ec_gids      = ec_ids,
        struct_count = np.zeros(n_syn, dtype=np.int16),
        struct_done  = np.zeros(n_syn, dtype=bool),
    )


def run_stc_hook(
    stc        : STCHook,
    ec_module,
    t_swr_start : float,
    t_swr_end   : float,
    current_t_ms: float,
    # STDP parameters
    tau_plus    : float = 20.0,   # ms — LTP time constant
    A_plus      : float = 0.01,   # LTP magnitude per coincidence
    A_minus     : float = 0.008,  # LTD magnitude per coincidence
    delay_ms    : float = 3.0,    # CA1→EC axonal delay (ms)
    # Tagging threshold: min |Δw| to set a synaptic tag.
    # A_plus * exp(-dt/tau_plus) > tag_threshold  ⟺  dt < tau_plus * ln(A_plus/tag_threshold)
    # With CA1 at ~10 Hz (v7), coincidences within 120 ms SWR window are far more
    # selective.  Default 0.005 → dt must be within ~28 ms of ideal Δt.
    tag_threshold : float = 0.005,
    # Tag parameters
    tau_tag_ms  : float = 2000.0, # tag decay time constant (sim ms)
    # PRP / L-LTP parameters
    # PRP is now counted in *discrete SWR events* per EC neuron (not Δw sum).
    # One unit is added each time an EC neuron fires during a SWR window.
    # PRP_threshold = minimum number of SWR activations needed to produce PRP
    # and capture tagged synapses to L-LTP.
    PRP_threshold : float = 6.0,  # raised 3→6: with EC now selective (~20-40%
                                   # of neurons per SWR vs 100% before), threshold
                                   # of 6 events produces a gradual staircase
                                   # matching Frey & Morris 1997 consolidation data.
    # RELATIVE to w_init, not absolute. These were 1.5 / 0.1 against a w_init of
    # 1.0; with the cortical weights rescaled to keep volleys subthreshold
    # (w_ca1_ec 1.0 -> 0.30), an absolute ceiling of 1.5 would let consolidation
    # grow the volley back to 3.75x threshold and undo the sparsening within a
    # few epochs -- the same trap already hit with the mPFC assoc hook's w_max.
    w_max_rel     : float = 1.5,  # ceiling as a multiple of w_init
    w_min_rel     : float = 0.1,  # floor as a multiple of w_init
    # Structural plasticity parameters
    struct_threshold : int   = 5,    # L-LTP epochs needed for spine enlargement
                                      # biology: ~3-7 LTP induction events (weeks)
    w_max_struct     : float = 2.0,  # weight ceiling after structural potentiation
                                      # models AMPA-receptor clustering + spine growth
    struct_boost     : float = 0.25, # one-off weight boost at structural capture
) -> dict:
    """
    Run one STC update cycle after a completed SWR epoch.

    Steps
    -----
    1. Read current CA1 and EC spike times from NEST recorders.
    2. For each CA1→EC synapse, find coincident (pre, post) pairs within
       the SWR window and compute STDP Δw using nearest-neighbour pairing.
       Only synapses where |Δw| ≥ tag_threshold are tagged (prevents
       spurious tagging of temporally distant pairs in high-rate networks).
    3. Decay existing tags, then refresh/set tags for updated synapses.
    4. Accumulate PRP pool per EC neuron: +1 per SWR event the EC neuron fired
       (biologically: PRP synthesis driven by postsynaptic activation, not
       individual synapse weight).
    5. Apply L-LTP capture where PRP ≥ threshold AND tag alive.
    6. Write updated weights back to NEST via SetStatus.
    7. Return a summary dict appended to stc.history.
    """
    import nest

    t_hook = time.perf_counter()
    stc.n_calls += 1

    # ---- 1. Read spike times from this SWR window ---------------------------
    # CA1 spikes — read from spike_recorder (neurons have no "events" key)
    ev_ca1    = nest.GetStatus(ec_module.ca1_spike_rec, "events")[0]
    ca1_t_all = np.array(ev_ca1["times"],   dtype=np.float32)
    ca1_s_all = np.array(ev_ca1["senders"], dtype=np.int64)

    # EC spikes
    ev_ec    = nest.GetStatus(ec_module.spike_rec, "events")[0]
    ec_t_all = np.array(ev_ec["times"],   dtype=np.float32)
    ec_s_all = np.array(ev_ec["senders"], dtype=np.int64)

    # Filter to SWR window (with ±delay margin)
    margin   = delay_ms + tau_plus * 3
    ca1_mask = (ca1_t_all >= t_swr_start - margin) & (ca1_t_all <= t_swr_end + margin)
    ec_mask  = (ec_t_all  >= t_swr_start - margin) & (ec_t_all  <= t_swr_end + margin)
    ca1_t = ca1_t_all[ca1_mask];  ca1_s = ca1_s_all[ca1_mask]
    ec_t  = ec_t_all[ec_mask];    ec_s  = ec_s_all[ec_mask]

    # ---- 2. STDP Δw per synapse (vectorised nearest-neighbour) --------------
    # Use GIDs cached at build time — no GetStatus call per epoch
    pre_ids    = stc.pre_ids
    post_ids_g = stc.post_ids_g

    delta_w = np.zeros(len(stc.w), dtype=np.float32)

    # Build lookup: CA1 neuron → last spike time in window
    ca1_last = {}
    for t, s in zip(ca1_t, ca1_s):
        if s not in ca1_last or t > ca1_last[s]:
            ca1_last[s] = float(t)

    # Build lookup: EC neuron → last spike time in window
    ec_last = {}
    for t, s in zip(ec_t, ec_s):
        if s not in ec_last or t > ec_last[s]:
            ec_last[s] = float(t)

    for i, (pre, post) in enumerate(zip(pre_ids, post_ids_g)):
        t_pre  = ca1_last.get(int(pre),  None)
        t_post = ec_last.get(int(post),  None)
        if t_pre is None or t_post is None:
            continue
        # Account for axonal delay: effective pre arrival = t_pre + delay
        dt = (t_post - t_pre) - delay_ms   # >0: LTP,  <0: LTD
        if dt >= 0:
            delta_w[i] = A_plus  * np.exp(-dt / tau_plus)
        else:
            delta_w[i] = -A_minus * np.exp(dt  / tau_plus)   # dt<0 → exp(pos)

    # ---- 3. Tag decay and refresh -------------------------------------------
    dt_since_tag = current_t_ms - stc.tag_time_ms           # ms since tag set
    decay_factor = np.exp(-np.clip(dt_since_tag, 0, 1e6) / tau_tag_ms)
    stc.tag *= decay_factor

    # Set/refresh tag ONLY for synapses with meaningful STDP coincidence.
    # tag_threshold filters out spurious low-amplitude delta_w from
    # temporally distant pairs that happen to both fire in the long SWR window.
    active = np.abs(delta_w) >= tag_threshold
    stc.tag[active]         = np.abs(delta_w[active])
    stc.tag_time_ms[active] = current_t_ms

    # ---- 4. PRP pool per EC neuron (event-count model) ----------------------
    # PRP is synthesised in the EC soma when the cell fires during a SWR.
    # Each SWR event in which a given EC neuron fires contributes +1 to its
    # PRP pool — independent of individual synapse weights.
    # This is biologically grounded (Frey & Morris 1997): PRP is a 'neuron-
    # level' signal triggered by strong post-synaptic activation, not a
    # weighted sum of individual EPSP tags.
    if len(ec_s) > 0:
        # Which EC neuron indices fired in this SWR window?
        ec_fired_gids  = np.unique(ec_s)
        fired_syn_mask = np.isin(stc.post_ids_g, ec_fired_gids)
        ec_fired_idx   = np.unique(stc.post_idx[fired_syn_mask])
        stc.prp_pool[ec_fired_idx] += 1.0
    n_ec_fired = int(len(np.unique(ec_s))) if len(ec_s) > 0 else 0

    # ---- 5. E-LTP (immediate) and L-LTP capture (threshold-gated) -----------
    # E-LTP: apply Δw immediately (reversible, will decay without L-LTP)
    _wmax = w_max_rel * stc.w_init
    _wmin = w_min_rel * stc.w_init
    stc.w = np.clip(stc.w + delta_w, _wmin, _wmax)

    # L-LTP: capture if PRP sufficient AND tag alive AND not already captured
    tag_alive   = stc.tag > 1e-4
    prp_above   = stc.prp_pool[stc.post_idx] >= PRP_threshold
    new_capture = tag_alive & prp_above & ~stc.ltp_done
    if new_capture.any():
        # Permanent weight consolidation: boost by 30% capped at w_max
        stc.w[new_capture] = np.minimum(stc.w[new_capture] * 1.3, _wmax)
        stc.ltp_done |= new_capture

    # ---- 5b. Structural plasticity — 3rd timescale (spine enlargement) -------
    # After struct_threshold L-LTP epochs a synapse is 'morphologically tagged':
    # the dendritic spine enlarges and AMPA receptors cluster permanently.
    # Model: increment struct_count for ALL currently-L-LTP synapses each call;
    # when count reaches threshold, apply one-off boost and raise w_max to 2.0.
    stc.struct_count[stc.ltp_done] += 1
    new_struct = (stc.struct_count >= struct_threshold) & ~stc.struct_done
    if new_struct.any():
        stc.w[new_struct] = np.minimum(
            stc.w[new_struct] + struct_boost, w_max_struct)
        stc.struct_done |= new_struct
    n_struct_new   = int(new_struct.sum())
    n_struct_total = int(stc.struct_done.sum())

    # ---- 6. Write weights back to NEST ---------------------------------------
    nest.SetStatus(stc.conns, "weight", stc.w.tolist())

    # ---- 7. Summary ----------------------------------------------------------
    n_active     = int(active.sum())
    n_tagged_syn = int((stc.tag > 1e-4).sum())
    n_ltp_new    = int(new_capture.sum())
    n_ltp_total  = int(stc.ltp_done.sum())
    w_mean       = float(stc.w.mean())
    w_ltp_mean   = float(stc.w[stc.ltp_done].mean()) if n_ltp_total > 0 else float('nan')
    prp_mean     = float(stc.prp_pool.mean())
    prp_max      = float(stc.prp_pool.max())
    dt_hook      = time.perf_counter() - t_hook

    summary = dict(
        event        = stc.n_calls,
        t_swr_start  = t_swr_start,
        t_swr_end    = t_swr_end,
        n_active_syn = n_active,
        n_tagged_syn = n_tagged_syn,
        n_ec_fired   = n_ec_fired,
        prp_mean     = prp_mean,
        prp_max      = prp_max,
        n_ltp_new    = n_ltp_new,
        n_ltp_total  = n_ltp_total,
        n_struct_new   = n_struct_new,
        n_struct_total = n_struct_total,
        w_mean       = w_mean,
        w_ltp_mean   = w_ltp_mean,
        dt_hook_s    = dt_hook,
    )
    stc.history.append(summary)

    print(f"  [STCHook] event={stc.n_calls:2d}  "
          f"tagged={n_active:,}  EC_fired={n_ec_fired:,}  "
          f"PRP_mean={prp_mean:.2f}/{PRP_threshold:.0f}  "
          f"L-LTP_new={n_ltp_new:,}  L-LTP_total={n_ltp_total:,}  "
          f"STRUCT_new={n_struct_new:,}  STRUCT_total={n_struct_total:,}  "
          f"w_mean={w_mean:.4f}  ({dt_hook:.2f}s)")
    return summary


# ============================================================================
# Phase 4 — Synaptic Homeostasis
# ============================================================================

def build_homeostasis_hook(net, alpha: float = 0.75):
    """
    Phase 4 synaptic homeostasis: downscale CA3 recurrent excitatory weights
    (CA3_SUP → CA3_SUP) by alpha after consolidation completes.

    Schaffer collaterals (CA3_SUP → CA1_PYR, ~345M synapses) are NOT touched.
    Only CA3 recurrent excitatory synapses are downscaled, which is the
    canonical test of EC trace independence per the Synaptic Homeostasis
    Hypothesis (Tononi & Cirelli).

    IMPORTANT — why we use target-only GetConnections:
    Calling nest.GetConnections(source, target) with BOTH filters forces NEST 3.9
    to walk the full ~471M-synapse connectome (>15 h on MN5, hangs the SLURM job).
    Calling nest.GetConnections(target=...) walks only the target's incoming
    slots. CA3_SUP receives ~12.7M synapses (recurrent + INT_SUP + DEEP +
    drives); we then post-filter in Python by source GID and weight sign.
    This is the same trick the STC hook uses for CA1→EC.
    """
    import dataclasses
    import nest

    @dataclasses.dataclass
    class HomeoHook:
        alpha:        float
        all_conns:    object       # full incoming collection (target=CA3_SUP)
        exc_mask:     np.ndarray   # bool mask: CA3_SUP→CA3_SUP excitatory
        weights_pre:  np.ndarray
        weights:      np.ndarray
        n_rec_exc:    int

    print("  [homeo] Querying connections incoming to CA3_SUP (target-only filter)...")
    print("  [homeo] (Schaffer collaterals skipped — 345M synapses, infeasible)")
    t0 = time.perf_counter()
    all_conns = nest.GetConnections(target=net["CA3_SUP"])
    n_total   = len(all_conns)
    print(f"  [homeo] CA3_SUP incoming scan: {n_total:,} synapses in {time.perf_counter()-t0:.1f}s")

    # Pull source GIDs and weights via vectorized GetStatus
    t1 = time.perf_counter()
    sources = np.array(nest.GetStatus(all_conns, "source"), dtype=np.int64)
    weights = np.array(nest.GetStatus(all_conns, "weight"), dtype=float)
    print(f"  [homeo] Pulled source GIDs + weights in {time.perf_counter()-t1:.1f}s")

    # Build set of CA3_SUP GIDs once (66k entries) for fast membership test
    ca3_sup_gids = set(net["CA3_SUP"].tolist())

    # Mark synapses whose source is a CA3_SUP neuron, AND excitatory (w>0)
    t2 = time.perf_counter()
    src_in_ca3sup = np.fromiter((s in ca3_sup_gids for s in sources),
                                 dtype=bool, count=n_total)
    exc_mask = src_in_ca3sup & (weights > 0)
    n_rec_exc = int(exc_mask.sum())
    print(f"  [homeo] Filtered CA3_SUP→CA3_SUP excitatory: {n_rec_exc:,} synapses "
          f"({time.perf_counter()-t2:.1f}s)")

    return HomeoHook(
        alpha        = alpha,
        all_conns    = all_conns,
        exc_mask     = exc_mask,
        weights_pre  = weights.copy(),
        weights      = weights,
        n_rec_exc    = n_rec_exc,
    )


def run_homeostasis_hook(homeo):
    """
    Apply multiplicative downscaling to CA3 recurrent excitatory synapses.
    Writes back the full weight vector to the cached SynapseCollection;
    non-excitatory entries retain their original values (no-op write).
    """
    import nest

    alpha = homeo.alpha
    t0 = time.perf_counter()

    # Downscale only the masked subset (CA3_SUP→CA3_SUP excitatory)
    homeo.weights[homeo.exc_mask] = np.maximum(
        homeo.weights[homeo.exc_mask] * alpha, 0.01)

    # SetStatus on the full collection — non-excitatory rows write same value
    nest.SetStatus(homeo.all_conns, "weight", homeo.weights.tolist())
    print(f"  [homeo] SetStatus complete in {time.perf_counter()-t0:.1f}s")

    stats = {
        "alpha":              alpha,
        "ca3_w_pre_mean":     float(homeo.weights_pre[homeo.exc_mask].mean()),
        "ca3_w_post_mean":    float(homeo.weights[homeo.exc_mask].mean()),
        "ca3_w_pre_std":      float(homeo.weights_pre[homeo.exc_mask].std()),
        "ca3_w_post_std":     float(homeo.weights[homeo.exc_mask].std()),
        "n_ca3_exc_synapses": homeo.n_rec_exc,
    }
    print(f"  [homeo] CA3 exc:  mean {stats['ca3_w_pre_mean']:.4f} → "
          f"{stats['ca3_w_post_mean']:.4f}  (×{alpha:.2f})  "
          f"[{stats['n_ca3_exc_synapses']:,} synapses]")
    return stats

# ============================================================================
# Replay quality metric
# ============================================================================

def replay_score(spk_times, spk_senders, seq_groups, window_start, window_stop):
    """Spearman rho between sequence-group index and mean spike time.

    IMPORTANT -- score the SWR event window itself, not a padded window.
    Call sites previously used (start-5, stop+30). That +30 ms tail reaches
    past the SWR into the post-event rebound, where the strong forward chain
    (seq_fwd K=20) re-ignites a FORWARD-propagating burst across all groups.
    That rebound is harmless for forward replay (it shares the forward
    ordering) but directly opposes reverse replay, cancelling it: measured on
    the 12% + DG run, rho_rev reads -0.094 with the pad and -0.789 without it,
    while rho_fwd is unchanged (+0.622 vs +0.613). Every archived consolidating
    run shows the same one-sided distortion (e.g. 25%: -0.033 -> -0.512), which
    is why bidirectional replay appeared to be incompatible with consolidation
    when in fact both were present all along.
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return None, None
    mask = (spk_times >= window_start) & (spk_times <= window_stop)
    t_win = spk_times[mask];  s_win = spk_senders[mask]
    gidx, gmean = [], []
    for k, grp in enumerate(seq_groups):
        t_g = t_win[np.isin(s_win, np.array(grp))]
        if len(t_g) > 0:
            gidx.append(k);  gmean.append(float(np.mean(t_g)))
    if len(gidx) < 3:
        return np.nan, np.nan
    from scipy.stats import spearmanr
    return spearmanr(gidx, gmean)


# ============================================================================
# Visualisation
# ============================================================================

def _get_spikes(spk_rec):
    ev = nest.GetStatus(spk_rec, "events")[0]
    return np.array(ev["times"], dtype=float), np.array(ev["senders"], dtype=int)


def _binned_rate(times_ms, n_cells, t_stop, bin_ms):
    edges     = np.arange(0.0, t_stop + bin_ms, bin_ms)
    counts, _ = np.histogram(times_ms, bins=edges)
    return edges[:-1] + bin_ms/2.0, counts / (bin_ms/1e3) / max(int(n_cells), 1)


def plot_bidirectional_replay(net, sim_ms=1000.0, save_prefix="replay"):
    t_sup,  s_sup  = _get_spikes(net["spk_ca3_sup"])
    t_deep, s_deep = _get_spikes(net["spk_ca3_deep"])
    t_ca3i_sup,  _ = _get_spikes(net["spk_ca3_int_sup"])
    t_ca3i_deep, _ = _get_spikes(net["spk_ca3_int_deep"])
    t_ca1p, s_ca1p = _get_spikes(net["spk_pyr"])
    t_ca1b, s_ca1b = _get_spikes(net["spk_ba"])

    seq_groups   = net["ca3_seq_groups"]
    n_groups     = net["n_seq_groups"]
    swr_fwd      = net["swr_fwd"]
    swr_rev      = net["swr_rev"]
    cmap_seq     = plt.cm.viridis
    group_colors = [cmap_seq(k / max(n_groups-1, 1)) for k in range(n_groups)]

    def shade(ax, alpha=0.18):
        ax.axvspan(*swr_fwd, color="steelblue", alpha=alpha, label="SWR-1 fwd")
        ax.axvspan(*swr_rev, color="tomato",    alpha=alpha, label="SWR-2 rev")

    out_paths = []

    # Fig 1: Overview
    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    fig.suptitle("Bidirectional Replay — Watson et al. 2025 Two-Layer CA3",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    for k, grp in enumerate(seq_groups):
        m = np.isin(s_sup, np.array(grp))
        ax.scatter(t_sup[m], s_sup[m], s=1.0, color=group_colors[k], rasterized=True)
    shade(ax);  ax.set_ylabel("CA3 SUP ID", fontsize=9)
    ax.set_title("A  CA3 SUPERFICIAL raster  [colour = seq group]", fontsize=9, loc="left")
    sm = ScalarMappable(cmap=cmap_seq, norm=Normalize(0, n_groups-1));  sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.01).set_label("Group #", fontsize=8)

    ax = axes[1]
    for k, grp in enumerate(net["ca3_deep_groups"]):
        m = np.isin(s_deep, np.array(grp))
        ax.scatter(t_deep[m], s_deep[m], s=1.5, color=group_colors[k],
                   marker="^", alpha=0.7, rasterized=True)
    shade(ax);  ax.set_ylabel("CA3 DEEP ID", fontsize=9)
    ax.set_title("B  CA3 DEEP raster  [burst-firing, tetrasynaptic output]", fontsize=9, loc="left")

    ax = axes[2]
    ax.scatter(t_ca1p, s_ca1p, s=0.8, color="slategray", rasterized=True)
    shade(ax);  ax.set_ylabel("CA1 PYR ID", fontsize=9)
    ax.set_title("C  CA1 PYR raster", fontsize=9, loc="left")

    ax = axes[3]
    tc, rc_sup  = _binned_rate(t_sup,  len(net["CA3_SUP"]),  sim_ms, 10.0)
    tc, rc_deep = _binned_rate(t_deep, len(net["CA3_DEEP"]), sim_ms, 10.0)
    tc, rc1     = _binned_rate(t_ca1p, len(net["PYR"]),      sim_ms, 10.0)
    ax.plot(tc, rc_sup,  color="darkorange", lw=1.2, label="CA3 SUP")
    ax.plot(tc, rc_deep, color="royalblue",  lw=1.2, label="CA3 DEEP")
    ax.plot(tc, rc1,     color="steelblue",  lw=1.2, alpha=0.7, label="CA1 PYR")
    shade(ax);  ax.legend(fontsize=7, ncol=3);  ax.set_ylabel("Rate (Hz)", fontsize=9)
    ax.set_title("D  Population rates  [10 ms bins]", fontsize=9, loc="left")

    ax = axes[4]
    tf, ri_sup  = _binned_rate(t_ca3i_sup,  len(net["CA3_INT_SUP"]),  sim_ms, 2.0)
    tf, ri_deep = _binned_rate(t_ca3i_deep, len(net["CA3_INT_DEEP"]), sim_ms, 2.0)
    tf, rb      = _binned_rate(t_ca1b,      len(net["BASKET"]),       sim_ms, 2.0)
    ax.plot(tf, ri_sup,  color="firebrick",    lw=0.8, alpha=0.85, label="CA3 INT_SUP")
    ax.plot(tf, ri_deep, color="salmon",       lw=0.8, alpha=0.85, label="CA3 INT_DEEP")
    ax.plot(tf, rb,      color="mediumorchid", lw=0.8, alpha=0.85, label="CA1 Basket")
    shade(ax);  ax.legend(fontsize=7, ncol=3)
    ax.set_xlabel("Time (ms)", fontsize=9);  ax.set_ylabel("Rate (Hz)", fontsize=9)
    ax.set_title("E  Inhibitory rates  [2 ms bins]", fontsize=9, loc="left")
    fig.tight_layout()
    p_path = f"{save_prefix}_fig1_overview.png"
    fig.savefig(p_path, dpi=150);  plt.close(fig);  out_paths.append(p_path)
    print(f"  saved {p_path}")

    # Fig 2: Heatmap
    bin_ms = 5.0
    edges  = np.arange(0.0, sim_ms + bin_ms, bin_ms)
    gs_per = len(seq_groups[0])
    heat   = np.zeros((n_groups, len(edges)-1))
    for k, grp in enumerate(seq_groups):
        m = np.isin(s_sup, np.array(grp))
        counts, _ = np.histogram(t_sup[m], bins=edges)
        heat[k]   = counts / (bin_ms/1e3) / max(gs_per, 1)
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(heat, aspect="auto", origin="lower",
                   extent=[0, sim_ms, -0.5, n_groups-0.5],
                   cmap="inferno", interpolation="nearest")
    fig.colorbar(im, ax=ax, pad=0.02).set_label("Rate (Hz)", fontsize=9)
    ax.axvspan(*swr_fwd, color="white", alpha=0.20, label="SWR-1 fwd")
    ax.axvspan(*swr_rev, color="cyan",  alpha=0.15, label="SWR-2 rev")
    ax.plot([swr_fwd[0], swr_fwd[0]+(swr_fwd[1]-swr_fwd[0])*0.75], [0, n_groups-1],
            "--w", lw=1.8, alpha=0.9, label="Fwd slope")
    ax.plot([swr_rev[0], swr_rev[0]+(swr_rev[1]-swr_rev[0])*0.75], [n_groups-1, 0],
            "--c", lw=1.8, alpha=0.9, label="Rev slope")
    ax.set_xlabel("Time (ms)", fontsize=10);  ax.set_ylabel("Sequence group #", fontsize=10)
    ax.set_title("CA3 SUP Sequence Group Heatmap", fontsize=11)
    ax.legend(fontsize=8, loc="upper right");  fig.tight_layout()
    p_path = f"{save_prefix}_fig2_heatmap.png"
    fig.savefig(p_path, dpi=150);  plt.close(fig);  out_paths.append(p_path)
    print(f"  saved {p_path}")

    return out_paths


# ============================================================================
# Console report
# ============================================================================

def print_report(net, sim_ms, scale_label, ec_module=None, dg_module=None,
                 eclv_module=None, mpfc_module=None):
    print(f"\n{'='*72}")
    print(f"SIMULATION REPORT  [{scale_label}]")
    print(f"{'='*72}")
    for label, pop, spk in [
        ("CA1 PYR",      net["PYR"],         net["spk_pyr"]),
        ("CA1 BASKET",   net["BASKET"],       net["spk_ba"]),
        ("CA1 OLM",      net["OLM"],          net["spk_olm"]),
        ("CA3 SUP",      net["CA3_SUP"],      net["spk_ca3_sup"]),
        ("CA3 DEEP",     net["CA3_DEEP"],     net["spk_ca3_deep"]),
        ("CA3 INT_SUP",  net["CA3_INT_SUP"],  net["spk_ca3_int_sup"]),
        ("CA3 INT_DEEP", net["CA3_INT_DEEP"], net["spk_ca3_int_deep"]),
    ]:
        ev   = nest.GetStatus(spk, "events")[0]
        rate = mean_rate(pop, spk, sim_ms)
        print(f"  {label:20s}: N={len(pop):8,} | {len(ev['times']):10,} spikes | {rate:6.2f} Hz")

    print("\n--- Replay quality (Spearman rho, CA3 SUP) ---")
    t_sup, s_sup = _get_spikes(net["spk_ca3_sup"])
    for label, win, expected_sign in [
        ("SWR-1 forward", net["swr_fwd"], +1),
        ("SWR-2 reverse", net["swr_rev"], -1),
    ]:
        # Score the SWR event window itself — see replay_score docstring for why
        # the old (-5, +30) padding one-sidedly cancelled reverse replay.
        rho, pval = replay_score(t_sup, s_sup, net["ca3_seq_groups"],
                                 win[0], win[1])
        if rho is not None and not np.isnan(rho):
            ok      = (expected_sign > 0 and rho > 0.5) or (expected_sign < 0 and rho < -0.5)
            verdict = "PASS" if ok else "WEAK"
            print(f"  {label:20s}: rho={rho:+.3f}  p={pval:.3f}  [{verdict}]")
        else:
            print(f"  {label:20s}: insufficient spikes or scipy missing")

    print("\n--- DEEP layer following ---")
    t_deep, _ = _get_spikes(net["spk_ca3_deep"])
    for label, (ws, we) in [("SWR-1 fwd", net["swr_fwd"]), ("SWR-2 rev", net["swr_rev"])]:
        n_s = np.sum((t_sup  >= ws) & (t_sup  <= we))
        n_d = np.sum((t_deep >= ws) & (t_deep <= we))
        print(f"  {label}: SUP={n_s:,}  DEEP={n_d:,}  DEEP/SUP ratio={n_d/max(n_s,1):.2f}")

    if dg_module is not None:
        print("\n--- Dentate gyrus (Phase 6.2 — input / pattern separation) ---")
        print("  perforant path : " + ("EC LII -> DG  [LOOP CLOSED: "
              "EC LII->DG->CA3->CA1->EC]" if getattr(dg_module, "ec_driven", False)
              else "Poisson stand-in  [loop OPEN — add --ec-lii to close]"))
        for label, pop_n, spk in [
            ("DG GC",      dg_module.N_gc,      dg_module.spk_gc),
            ("DG MC_LOW",  dg_module.N_mc_low,  dg_module.spk_mc_low),
            ("DG MC_HIGH", dg_module.N_mc_high, dg_module.spk_mc_high),
            ("DG BASKET",  dg_module.N_basket,  dg_module.spk_basket),
        ]:
            ev   = nest.GetStatus(spk, "events")[0]
            rate = len(ev["senders"]) / (max(pop_n, 1) * sim_ms / 1000.0)
            print(f"  {label:20s}: N={pop_n:8,} | {len(ev['times']):10,} spikes | {rate:6.2f} Hz")

        # MC_HIGH should fire less than MC_LOW under equal drive (higher rheobase)
        r_low  = len(nest.GetStatus(dg_module.spk_mc_low,  "events")[0]["senders"]) / max(dg_module.N_mc_low, 1)
        r_high = len(nest.GetStatus(dg_module.spk_mc_high, "events")[0]["senders"]) / max(dg_module.N_mc_high, 1)
        if r_high > 0:
            print(f"  MC_LOW/MC_HIGH spikes-per-cell ratio = {r_low/r_high:.2f} "
                  f"(expect >1: MC_HIGH has the higher rheobase)")
        else:
            print(f"  MC_LOW/MC_HIGH: MC_HIGH silent (0 spikes) vs MC_LOW "
                  f"{r_low:.3f} spikes/cell — consistent with its 4.97x higher "
                  f"rheobase under equal drive")

        print("  Pattern separation (granule sparse coding):")
        overall = dg_pattern_separation_stats(dg_module, sim_ms)
        print(f"    whole run : active {overall['active_fraction']*100:5.2f}% "
              f"({overall['n_active']:,}/{overall['n_gc']:,})  "
              f"mean {overall['mean_rate_hz']:.2f} Hz  [{overall['verdict']}]")
        for label, win in [("SWR-1 fwd", net["swr_fwd"]), ("SWR-2 rev", net["swr_rev"])]:
            w = dg_pattern_separation_stats(dg_module, sim_ms, window=win)
            print(f"    {label:9s} : active {w['active_fraction']*100:5.2f}% "
                  f"({w['n_active']:,} cells)  [{w['verdict']}]")
        print("    (DG hallmark: 2-4% active per pattern; FLAG = feedback "
              "inhibition too weak, needs MN5 tuning of w_basket_gc/pp_weight)")

    if ec_module is not None:
        print("\n--- EC LII/III (cortical target) ---")
        t_ec, _ = _get_spikes(ec_module.spike_rec)
        rate_ec  = len(t_ec) / (ec_module.N * sim_ms / 1000.0)
        print(f"  {'EC LII/III':20s}: N={ec_module.N:8,} | {len(t_ec):10,} spikes | {rate_ec:6.2f} Hz")
        for label, (ws, we) in [("SWR-1 fwd", net["swr_fwd"]), ("SWR-2 rev", net["swr_rev"])]:
            n_ec = np.sum((t_ec >= ws) & (t_ec <= we))
            print(f"  {label}: EC spikes in window = {n_ec:,}")
        print(f"  CA1->EC weights  : deferred — use STC hook (Phase 2)")

    if 'eclv_module' in dir() and eclv_module is not None:
        print("\n--- EC Layer V (Phase 3 — loop closure) ---")
        t_lv, _ = _get_spikes(eclv_module.spike_rec)
        rate_lv  = len(t_lv) / (eclv_module.N * sim_ms / 1000.0)
        print(f"  {'EC LV':20s}: N={eclv_module.N:8,} | {len(t_lv):10,} spikes | {rate_lv:6.2f} Hz")
        for label, (ws, we) in [("SWR-1 fwd", net["swr_fwd"]), ("SWR-2 rev", net["swr_rev"])]:
            n_lv = np.sum((t_lv >= ws) & (t_lv <= we))
            print(f"  {label}: EC LV spikes in window = {n_lv:,}")

    if 'mpfc_module' in dir() and mpfc_module is not None:
        print("\n--- mPFC (Phase 3 — cortical engram endpoint) ---")
        t_pfc, _ = _get_spikes(mpfc_module.spike_rec)
        rate_pfc  = len(t_pfc) / (mpfc_module.N * sim_ms / 1000.0)
        print(f"  {'mPFC':20s}: N={mpfc_module.N:8,} | {len(t_pfc):10,} spikes | {rate_pfc:6.2f} Hz")

    print(f"{'='*72}")


# ============================================================================
# HDF5 export  (offline plotting on any machine — no NEST required)
# ============================================================================

def save_replay_hdf5(net, sim_ms, scale_label, outpath, bin_ms=10.0,
                     ec_module=None, stc_hook=None,
                     eclv_module=None, mpfc_module=None,
                     homeo_stats=None, homeo_results=None, dg_module=None,
                     mpfc_assoc_hook=None, schaffer_hook=None,
                     mpfc_rec_hook=None):
    """
    Save all simulation results to an HDF5 file for offline plotting.

    Schema
    ------
    /                       — root attrs: metadata
    /times_ms               — bin-centre times [n_bins]
    /stats                  — attrs: spearman rho/pval, mean firing rates
    /ca3_sup/
        spk_times           — raw spike times   (float32, compressed)
        spk_senders         — raw spike senders (int32,   compressed)
        rate                — population-mean rate [n_bins]  (Hz)
        group_ids           — seq-group membership [n_groups, gs]  (int32)
        heatmap             — per-group rate  [n_groups, n_bins]   (float32)
    /ca3_deep/              — same structure (no heatmap)
    /ca3_int_sup/           — spk_times, spk_senders, rate
    /ca3_int_deep/          — spk_times, spk_senders, rate
    /ca1_pyr/               — spk_times, spk_senders, rate
    /ca1_basket/            — spk_times, spk_senders, rate
    /ca1_olm/               — spk_times, spk_senders, rate
    """
    if not _HDF5_AVAILABLE:
        print(">>> [WARNING] h5py not installed — skipping HDF5 export.")
        return

    import h5py
    import datetime

    edges    = np.arange(0.0, sim_ms + bin_ms, bin_ms)
    times_ms = (edges[:-1] + bin_ms / 2.0).astype(np.float32)

    # (h5_group_name, net_population_key, net_spike_recorder_key)
    pop_map = [
        ("ca3_sup",      "CA3_SUP",      "spk_ca3_sup"),
        ("ca3_deep",     "CA3_DEEP",     "spk_ca3_deep"),
        ("ca3_int_sup",  "CA3_INT_SUP",  "spk_ca3_int_sup"),
        ("ca3_int_deep", "CA3_INT_DEEP", "spk_ca3_int_deep"),
        ("ca1_pyr",      "PYR",          "spk_pyr"),
        ("ca1_basket",   "BASKET",       "spk_ba"),
        ("ca1_olm",      "OLM",          "spk_olm"),
    ]

    # -------------------------------------------------------------------------
    # Gather spikes from all MPI ranks to rank 0.
    #
    # In an MPI run each rank owns a disjoint subset of neurons, so
    # nest.GetStatus(spike_recorder, "events") returns ONLY the locally-owned
    # spikes.  Without gathering, every rank would open the same HDF5 file in
    # "w" (truncate) mode and write partial data, corrupting gzip chunks.
    # -------------------------------------------------------------------------
    spk_cache = {}
    for h5_key, pop_key, spk_key in pop_map:
        t_local, s_local = _get_spikes(net[spk_key])
        # _gather_spikes handles the MPI barrier + gather + sort internally.
        # On ranks > 0 it returns empty arrays; those ranks skip file I/O below.
        spk_cache[h5_key] = _gather_spikes(t_local, s_local)

    # Gather EC LII spikes if the module is present
    if ec_module is not None:
        t_local, s_local = _get_spikes(ec_module.spike_rec)
        spk_cache["ec_lii"] = _gather_spikes(t_local, s_local)

    # Gather DG spikes if the module is present (Phase 6.2)
    if dg_module is not None:
        for cache_key, spk_rec in [
            ("dg_gc",      dg_module.spk_gc),
            ("dg_mc_low",  dg_module.spk_mc_low),
            ("dg_mc_high", dg_module.spk_mc_high),
            ("dg_basket",  dg_module.spk_basket),
        ]:
            spk_cache[cache_key] = _gather_spikes(*_get_spikes(spk_rec))
    # Phase 3 spike gathering handled inline (modules may be None)

    # Only rank 0 writes the file — all other ranks are done here.
    if _mpi_rank() != 0:
        return

    compress = dict(compression="gzip", compression_opts=4)

    with h5py.File(outpath, "w") as h5:
        # --- root metadata ---------------------------------------------------
        h5.attrs["created_utc"]   = datetime.datetime.utcnow().isoformat()
        h5.attrs["sim_ms"]        = float(sim_ms)
        h5.attrs["dt_ms"]         = float(bin_ms)
        h5.attrs["scale"]         = scale_label
        h5.attrs["n_groups"]      = int(net["n_seq_groups"])
        h5.attrs["swr_fwd_start"] = float(net["swr_fwd"][0])
        h5.attrs["swr_fwd_stop"]  = float(net["swr_fwd"][1])
        h5.attrs["swr_rev_start"] = float(net["swr_rev"][0])
        h5.attrs["swr_rev_stop"]  = float(net["swr_rev"][1])
        h5.attrs["ec_lii_present"] = ec_module is not None
        if ec_module is not None:
            h5.attrs["ec_lii_N"]       = ec_module.N
            h5.attrs["ec_lii_K_ca1"]   = ec_module.K_ca1_ec
            h5.attrs["ec_lii_w_init"]  = ec_module.w_init
        try:
            import nest as _nest
            h5.attrs["nest_version"] = _nest.__version__
        except Exception:
            pass

        h5.create_dataset("times_ms", data=times_ms)

        # --- per-population groups -------------------------------------------
        for h5_key, pop_key, spk_key in pop_map:
            t_spk, s_spk = spk_cache[h5_key]
            n_cells       = int(len(net[pop_key]))

            g = h5.create_group(h5_key)
            g.attrs["n_cells"] = n_cells
            g.create_dataset("spk_times",   data=t_spk.astype(np.float32), **compress)
            g.create_dataset("spk_senders", data=s_spk.astype(np.int32),   **compress)

            counts, _ = np.histogram(t_spk, bins=edges)
            rate = (counts / (bin_ms / 1e3) / max(n_cells, 1)).astype(np.float32)
            g.create_dataset("rate", data=rate)

        # --- EC LII/III group (optional) -------------------------------------
        if ec_module is not None:
            t_spk, s_spk = spk_cache["ec_lii"]
            g_ec = h5.create_group("ec_lii")
            g_ec.attrs["n_cells"]   = ec_module.N
            g_ec.attrs["K_ca1_ec"]  = ec_module.K_ca1_ec
            g_ec.attrs["w_init"]    = ec_module.w_init
            g_ec.create_dataset("spk_times",   data=t_spk.astype(np.float32), **compress)
            g_ec.create_dataset("spk_senders", data=s_spk.astype(np.int32),   **compress)
            counts, _ = np.histogram(t_spk, bins=edges)
            rate_ec = (counts / (bin_ms / 1e3) / max(ec_module.N, 1)).astype(np.float32)
            g_ec.create_dataset("rate", data=rate_ec)
            g_ec.attrs["w_ca1_ec_note"] = "deferred to Phase 2 STC hook"

        # Phase 3: EC LV
        if eclv_module is not None:
            t_lv, s_lv = _gather_spikes(*_get_spikes(eclv_module.spike_rec))
            if _mpi_rank() == 0:
                g_lv = h5.create_group("ec_lv")
                g_lv.attrs["n_cells"]    = eclv_module.N
                g_lv.attrs["K_ca1_lv"]   = eclv_module.K_ca1_lv
                g_lv.attrs["K_eclii_lv"] = eclv_module.K_eclii_lv
                g_lv.attrs["w_init"]     = eclv_module.w_init
                g_lv.create_dataset("spk_times",   data=t_lv.astype(np.float32), **compress)
                g_lv.create_dataset("spk_senders", data=s_lv.astype(np.int32),   **compress)
                counts_lv, _ = np.histogram(t_lv, bins=edges)
                g_lv.create_dataset("rate",
                    data=(counts_lv/(bin_ms/1e3)/max(eclv_module.N,1)).astype(np.float32))
                h5.attrs["ec_lv_present"] = True

        # Phase 3: mPFC
        if mpfc_module is not None:
            t_pfc, s_pfc = _gather_spikes(*_get_spikes(mpfc_module.spike_rec))
            if _mpi_rank() == 0:
                g_pfc = h5.create_group("mpfc")
                g_pfc.attrs["n_cells"]     = mpfc_module.N
                g_pfc.attrs["K_eclv_mpfc"] = mpfc_module.K_eclv_mpfc
                g_pfc.attrs["w_init"]      = mpfc_module.w_init
                g_pfc.create_dataset("spk_times",   data=t_pfc.astype(np.float32), **compress)
                g_pfc.create_dataset("spk_senders", data=s_pfc.astype(np.int32),   **compress)
                counts_pfc, _ = np.histogram(t_pfc, bins=edges)
                g_pfc.create_dataset("rate",
                    data=(counts_pfc/(bin_ms/1e3)/max(mpfc_module.N,1)).astype(np.float32))
                h5.attrs["mpfc_present"] = True
                # mPFC interneurons: without these the lateral-inhibition
                # loop is invisible in the output, which is what made the
                # 12% mPFC saturation hard to diagnose from the file alone.
                if getattr(mpfc_module, "INT", None) is not None:
                    t_i, s_i = _gather_spikes(*_get_spikes(mpfc_module.spk_int))
                    g_i = h5.create_group("mpfc_int")
                    g_i.attrs["n_cells"] = mpfc_module.N_int
                    g_i.create_dataset("spk_times",   data=t_i.astype(np.float32), **compress)
                    g_i.create_dataset("spk_senders", data=s_i.astype(np.int32),   **compress)
                    c_i, _ = np.histogram(t_i, bins=edges)
                    g_i.create_dataset("rate",
                        data=(c_i/(bin_ms/1e3)/max(mpfc_module.N_int,1)).astype(np.float32))

        # --- Schaffer STDP weights -------------------------------------------
        # Exported so the CONSOLIDATED TRACE can be compared across runs: train
        # pattern A only vs pattern B only on the same seed, then ask whether the
        # potentiated synapses trace back to that pattern's CA3 cells. Source
        # gids are needed for that attribution, not just the weights.
        if schaffer_hook is not None:
            sg2 = h5.create_group("schaffer_stdp")
            m = schaffer_hook.mask
            sg2.attrs["n_all"]    = int(len(schaffer_hook.w))
            sg2.attrs["n_ca3_ca1"]= int(m.sum())
            sg2.attrs["w_init"]   = float(schaffer_hook.w_init)
            sg2.create_dataset("w_final", data=schaffer_hook.w[m].astype(np.float32), **compress)
            sg2.create_dataset("pre_gid", data=schaffer_hook.pre_g[m].astype(np.int64), **compress)
            sg2.create_dataset("post_gid",data=schaffer_hook.post_g[m].astype(np.int64), **compress)
            sg2.create_dataset("delay",  data=schaffer_hook.delay[m].astype(np.float32), **compress)

        # --- cortical association build-up (EC LV -> mPFC) -------------------
        if mpfc_assoc_hook is not None and mpfc_assoc_hook.history:
            hist = mpfc_assoc_hook.history
            ag = h5.create_group("mpfc_assoc")
            ag.attrs["description"] = (
                "Replay-gated Hebbian build-up of the EC LV->mPFC projection: "
                "co-activation during an SWR potentiates, post-without-pre "
                "weakly depresses. frac_associated is the share of synapses "
                "above the engram threshold.")
            ag.attrs["n_events"] = len(hist)
            ag.attrs["w_init"]   = mpfc_assoc_hook.w_init
            for key, dt in (("n_coactive", np.int32), ("n_hetero", np.int32),
                            ("w_mean", np.float32), ("n_associated", np.int32),
                            ("frac_associated", np.float32),
                            ("w_cv", np.float32),
                            ("t_swr_start", np.float32)):
                ag.create_dataset(key, data=np.array([r[key] for r in hist], dtype=dt))
            ag.create_dataset("w_final", data=mpfc_assoc_hook.w.astype(np.float32),
                              **compress)

        # --- recurrent mPFC->mPFC (what a lesioned cortex must recall ON) ----
        # Kept separate from mpfc_assoc: that group is the FEEDFORWARD
        # EC LV->mPFC projection. Conflating the two led to reading the
        # feedforward weights as evidence about the cortical attractor.
        if mpfc_rec_hook is not None and mpfc_rec_hook.history:
            hist = mpfc_rec_hook.history
            rg = h5.create_group("mpfc_recurrent")
            rg.attrs["description"] = (
                "Replay-gated Hebbian build-up of the RECURRENT mPFC->mPFC "
                "collaterals -- the only substrate left to reactivate a "
                "pattern once the hippocampus is lesioned.")
            rg.attrs["n_events"] = len(hist)
            rg.attrs["w_init"]   = mpfc_rec_hook.w_init
            for key, dt in (("n_coactive", np.int32), ("n_hetero", np.int32),
                            ("w_mean", np.float32), ("n_associated", np.int32),
                            ("frac_associated", np.float32),
                            ("w_cv", np.float32),
                            ("t_swr_start", np.float32)):
                rg.create_dataset(key, data=np.array([r[key] for r in hist], dtype=dt))
            rg.create_dataset("w_final", data=mpfc_rec_hook.w.astype(np.float32),
                              **compress)
            # pre/post indices make per-cell convergence measurable
            if getattr(mpfc_rec_hook, "pre_idx", None) is not None:
                rg.create_dataset("pre_idx", data=mpfc_rec_hook.pre_idx.astype(np.int32), **compress)
                rg.create_dataset("post_idx", data=mpfc_rec_hook.post_idx.astype(np.int32), **compress)

        # --- Dentate gyrus groups (optional, Phase 6.2) ----------------------
        h5.attrs["dg_present"] = dg_module is not None
        if dg_module is not None:
            dg_pops = [
                ("dg_gc",      dg_module.N_gc),
                ("dg_mc_low",  dg_module.N_mc_low),
                ("dg_mc_high", dg_module.N_mc_high),
                ("dg_basket",  dg_module.N_basket),
            ]
            for cache_key, n_cells in dg_pops:
                t_spk, s_spk = spk_cache[cache_key]
                g_dg = h5.create_group(cache_key)
                g_dg.attrs["n_cells"] = int(n_cells)
                g_dg.create_dataset("spk_times",   data=t_spk.astype(np.float32), **compress)
                g_dg.create_dataset("spk_senders", data=s_spk.astype(np.int32),   **compress)
                counts, _ = np.histogram(t_spk, bins=edges)
                g_dg.create_dataset("rate",
                    data=(counts/(bin_ms/1e3)/max(n_cells,1)).astype(np.float32))
            # Pattern-separation summary — the DG validation metric, so it can
            # be read offline without re-deriving it from the granule rasters.
            h5["dg_gc"].attrs["K_mf_ca3_sup"]  = dg_module.K_mf_sup
            h5["dg_gc"].attrs["K_mf_ca3_deep"] = dg_module.K_mf_deep
            overall = dg_pattern_separation_stats(dg_module, sim_ms)
            h5["dg_gc"].attrs["active_fraction"] = overall["active_fraction"]
            h5["dg_gc"].attrs["n_active"]        = overall["n_active"]
            h5["dg_gc"].attrs["sparse_verdict"]  = overall["verdict"]
            for name, win in [("fwd", net["swr_fwd"]), ("rev", net["swr_rev"])]:
                w = dg_pattern_separation_stats(dg_module, sim_ms, window=win)
                h5["dg_gc"].attrs[f"active_fraction_{name}"] = w["active_fraction"]

        # --- sequence group membership (CA3 SUP + DEEP) ----------------------
        sup_groups  = net["ca3_sup_groups"]
        deep_groups = net["ca3_deep_groups"]
        n_groups    = int(net["n_seq_groups"])
        gs_sup      = len(sup_groups[0])
        gs_deep     = len(deep_groups[0])

        sup_ids_arr  = np.array(sup_groups,  dtype=np.int32)   # [n_groups, gs_sup]
        deep_ids_arr = np.array(deep_groups, dtype=np.int32)   # [n_groups, gs_deep]
        h5["ca3_sup"].create_dataset("group_ids",  data=sup_ids_arr,  **compress)
        h5["ca3_deep"].create_dataset("group_ids", data=deep_ids_arr, **compress)

        # --- CA3 SUP sequence heatmap  [n_groups × n_bins] -------------------
        t_sup, s_sup = spk_cache["ca3_sup"]
        heat = np.zeros((n_groups, len(times_ms)), dtype=np.float32)
        for k, grp in enumerate(sup_groups):
            m = np.isin(s_sup, np.asarray(grp, dtype=np.int64))
            counts, _ = np.histogram(t_sup[m], bins=edges)
            heat[k] = counts / (bin_ms / 1e3) / max(gs_sup, 1)
        h5["ca3_sup"].create_dataset("heatmap", data=heat, **compress)

        # --- CA3 DEEP sequence heatmap ---------------------------------------
        t_deep, s_deep = spk_cache["ca3_deep"]
        heat_d = np.zeros((n_groups, len(times_ms)), dtype=np.float32)
        for k, grp in enumerate(deep_groups):
            m = np.isin(s_deep, np.asarray(grp, dtype=np.int64))
            counts, _ = np.histogram(t_deep[m], bins=edges)
            heat_d[k] = counts / (bin_ms / 1e3) / max(gs_deep, 1)
        h5["ca3_deep"].create_dataset("heatmap", data=heat_d, **compress)

        # --- replay quality stats --------------------------------------------
        sg = h5.create_group("stats")
        for label_key, win in [("fwd", net["swr_fwd"]), ("rev", net["swr_rev"])]:
            rho, pval = replay_score(t_sup, s_sup, sup_groups, win[0], win[1])
            sg.attrs[f"rho_{label_key}"]  = float(rho)  if (rho  is not None and not np.isnan(rho))  else float("nan")
            sg.attrs[f"pval_{label_key}"] = float(pval) if (pval is not None and not np.isnan(pval)) else float("nan")

        # mean firing rates (scalar per population)
        for h5_key, pop_key, _ in pop_map:
            t_spk, _ = spk_cache[h5_key]
            n_cells   = int(len(net[pop_key]))
            sg.attrs[f"mean_rate_{h5_key}"] = float(len(t_spk) / (n_cells * sim_ms / 1000.0))

        if ec_module is not None:
            t_ec, _ = spk_cache["ec_lii"]
            sg.attrs["mean_rate_ec_lii"] = float(len(t_ec) / (ec_module.N * sim_ms / 1000.0))

        # --- STC consolidation history (Phase 2) -----------------------------
        if stc_hook is not None and stc_hook.history:
            stc_grp = h5.create_group("stc")
            hist    = stc_hook.history
            stc_grp.create_dataset("event",        data=np.array([h["event"]        for h in hist], dtype=np.int32))
            stc_grp.create_dataset("t_swr_start",  data=np.array([h["t_swr_start"]  for h in hist], dtype=np.float32))
            stc_grp.create_dataset("t_swr_end",    data=np.array([h["t_swr_end"]    for h in hist], dtype=np.float32))
            stc_grp.create_dataset("n_active_syn", data=np.array([h["n_active_syn"] for h in hist], dtype=np.int32))
            stc_grp.create_dataset("n_tagged_syn", data=np.array([h.get("n_tagged_syn", h["n_active_syn"]) for h in hist], dtype=np.int32))
            stc_grp.create_dataset("n_ec_fired",   data=np.array([h.get("n_ec_fired", 0)  for h in hist], dtype=np.int32))
            stc_grp.create_dataset("prp_mean",     data=np.array([h.get("prp_mean", 0.0)  for h in hist], dtype=np.float32))
            stc_grp.create_dataset("prp_max",      data=np.array([h.get("prp_max",  0.0)  for h in hist], dtype=np.float32))
            stc_grp.create_dataset("n_ltp_new",      data=np.array([h["n_ltp_new"]              for h in hist], dtype=np.int32))
            stc_grp.create_dataset("n_ltp_total",    data=np.array([h["n_ltp_total"]            for h in hist], dtype=np.int32))
            stc_grp.create_dataset("n_struct_new",   data=np.array([h.get("n_struct_new",   0)  for h in hist], dtype=np.int32))
            stc_grp.create_dataset("n_struct_total", data=np.array([h.get("n_struct_total", 0)  for h in hist], dtype=np.int32))
            stc_grp.create_dataset("w_mean",       data=np.array([h["w_mean"]       for h in hist], dtype=np.float32))
            stc_grp.create_dataset("w_ltp_mean",   data=np.array([h["w_ltp_mean"]   for h in hist], dtype=np.float32))
            # Final weight distribution + L-LTP mask
            stc_grp.create_dataset("w_final",      data=stc_hook.w.astype(np.float32),           **compress)
            stc_grp.create_dataset("ltp_mask",     data=stc_hook.ltp_done.astype(np.uint8),      **compress)
            # PRP pool + tag snapshots (for tag occupancy map figure)
            stc_grp.create_dataset("prp_pool_final", data=stc_hook.prp_pool.astype(np.float32),  **compress)
            stc_grp.create_dataset("tag_final",    data=stc_hook.tag.astype(np.float32),          **compress)
            # Per-synapse EC neuron index (for tag occupancy grouping)
            stc_grp.create_dataset("post_idx",       data=stc_hook.post_idx.astype(np.int32),         **compress)
            # Structural plasticity snapshots
            if stc_hook.struct_count is not None:
                stc_grp.create_dataset("struct_count", data=stc_hook.struct_count.astype(np.int16),  **compress)
                stc_grp.create_dataset("struct_done",  data=stc_hook.struct_done.astype(np.uint8),   **compress)
            stc_grp.attrs["n_swr_events"]  = stc_hook.n_calls
            stc_grp.attrs["w_init"]        = ec_module.w_init
            stc_grp.attrs["n_ec_neurons"]  = ec_module.N
            stc_grp.attrs["n_synapses"]    = len(stc_hook.w)
            print(f"  [HDF5] STC history: {stc_hook.n_calls} events, "
                  f"{int(stc_hook.ltp_done.sum()):,} L-LTP synapses")

        # ---- Phase 4: Homeostasis group ------------------------------------
        # Multi-alpha aware: each alpha gets its own subgroup
        # /homeostasis/alpha_050/, /homeostasis/alpha_075/, etc.
        # If single alpha, also flatten into /homeostasis/ root attrs for
        # backward compatibility with existing analysis scripts.
        if homeo_results is None:
            homeo_results = {}
        if homeo_results:
            hg = h5.create_group("homeostasis")
            hg.attrs["description"] = (
                "Phase 4 synaptic homeostasis: multiplicative downscaling of "
                "CA3 recurrent excitatory weights after consolidation. "
                "L-LTP synapses in CA1->EC are NOT modified. "
                "rho_fwd/rev_post_homeo = replay quality during the "
                "verification epoch run immediately after downscaling.")
            hg.attrs["n_alphas"]    = len(homeo_results)
            hg.attrs["alpha_list"]  = np.array(sorted(homeo_results.keys()),
                                                dtype=float)
            hg.attrs["sweep_mode"]  = bool(len(homeo_results) > 1)

            for alpha_val, stats in sorted(homeo_results.items()):
                tag = f"alpha_{int(round(alpha_val * 100)):03d}"   # e.g. "alpha_075"
                sg  = hg.create_group(tag)
                for k, v in stats.items():
                    sg.attrs[k] = float(v)
                print(f"  [HDF5] /homeostasis/{tag}: "
                      f"alpha={stats['alpha']:.2f}  "
                      f"CA3 {stats['ca3_w_pre_mean']:.4f}->"
                      f"{stats['ca3_w_post_mean']:.4f}  "
                      f"rho_fwd={stats.get('rho_fwd_post_homeo', float('nan')):+.3f}")

            # Flatten single-alpha case to root attrs (legacy compatibility)
            if len(homeo_results) == 1:
                only_alpha = list(homeo_results.keys())[0]
                for k, v in homeo_results[only_alpha].items():
                    hg.attrs[k] = float(v)

    print(f">>> Saved HDF5: {outpath}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bidirectional hippocampal replay — Watson et al. 2025 (v4: EC LII module)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
--scale N  accepts any integer from 1 to 100 (percent of full rat hippocampus).

Approximate neuron counts and runtimes on MN5 (single node, 50 OpenMP threads):
  --scale  1   ~   7,700 neurons   ~2-3 min
  --scale  5   ~  38,500 neurons   ~8-12 min
  --scale 12   ~  93,500 neurons   ~25-40 min
  --scale 25   ~ 195,000 neurons   ~1-2 h
  --scale 50   ~ 390,000 neurons   ~3-5 h
  --scale 100  ~ 781,000 neurons   ~8-12 h

n_seq_groups = max(10, round(10 * sqrt(scale))):
  scale 1→10 groups,  4→20,  9→30,  25→50,  100→100

Cortical module:
  --ec-lii   adds EC LII/III (Phase 1 consolidation target).
             CA1→EC projection uses stdp_synapse.  At 10% scale this
             adds ~10k neurons and ~5M STDP synapses (~0.6 GB RAM).

Dentate gyrus (Phase 6.2):
  --dg       replaces the Poisson DG proxy with a real granule/mossy/basket
             circuit driving CA3 via mossy-fibre detonator synapses. Neuron
             params come from the MN5-confirmed f-I calibration. Granule cells
             are large (1.2M ref); use --dg-scale to size DG independently of
             --scale on smaller test runs.
        """,
    )
    parser.add_argument(
        "--scale", type=int, default=1, metavar="PCT",
        help="Network scale as integer percent of full rat hippocampus (1–100, default: 1)")
    parser.add_argument(
        "--sim-ms", type=float, default=1000.0,
        help="Simulation duration in ms (default: 1000)")
    parser.add_argument(
        "--threads", type=int, default=None,
        help="OpenMP threads per MPI rank (overrides auto-selected default)")
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip figure generation — recommended for scale>=12 on HPC")
    parser.add_argument(
        "--out-hdf5", type=str, default=None, metavar="FILE",
        help="Path for the output HDF5 file. "
             "If omitted, written to replay_output_<N>pct/replay_<N>pct.h5")
    # ---- Phase 6.2 dentate gyrus flags --------------------------------------
    parser.add_argument(
        "--dg", action="store_true",
        help="Add a real dentate gyrus (Phase 6.2): granule + two mossy-cell "
             "classes + basket feedback, driving CA3 via mossy-fibre detonator "
             "synapses. Replaces the Poisson DG proxy on CA3 (suppress_dg_drive). "
             "Params from the confirmed f-I calibration.")
    parser.add_argument(
        "--dg-scale", type=int, default=None, metavar="PCT",
        help="DG scale as independent percent of the rat DG reference counts "
             "(1.2M granule cells etc.). Defaults to --scale if omitted.")
    parser.add_argument(
        "--delay-jitter", type=float, default=0.0, metavar="MS",
        help="Per-synapse axonal delay jitter (ms) on the feedforward readout "
             "projections: Schaffer, CA1->EC LII/LV, EC LII->LV, EC LV->mPFC. "
             "0 (default) keeps the original single scalar delay per projection. "
             "Heterogeneous delays are the substrate for a temporal/polychronous "
             "code; the CA3 sequence chain and EC LV->CA3 feedback stay scalar "
             "because those delays are load-bearing for replay itself.")
    parser.add_argument(
        "--delay-jitter-wcomp", type=float, default=1.0, metavar="X",
        help="Weight scale applied to the jittered projections when "
             "--delay-jitter > 0. Spreading arrival times reduces coincident "
             "summation and lowers downstream rates, which confounds any "
             "comparison against the unjittered condition; this restores drive "
             "so the two can be compared at matched firing rates.")
    parser.add_argument(
        "--schaffer-k", type=int, default=None, metavar="K",
        help="Override the CA3->CA1 Schaffer in-degree. Weights are scaled by "
             "K_default/K so mean CA1 drive is preserved. The default (3000, "
             "clamped to CA3 size) gives density 1.0 at small scales -- every "
             "CA1 cell sees all of CA3 -- and makes a plasticity hook on 12M "
             "synapses impractical.")
    parser.add_argument(
        "--schaffer-group-frac", type=float, default=0.0, metavar="FRAC",
        help="Phase 12 (RESULTS.md SS20+): bias each CA1 PYR cell's (reduced, "
             "--schaffer-k) Schaffer sources toward ONE CA3 sequence group "
             "instead of sampling uniformly across all groups. FRAC in [0,1] "
             "is the expected fraction of a cell's in-degree drawn from its "
             "assigned home group (creation index modulo n_seq_groups); the "
             "rest is drawn uniformly from the other groups. Unlike EC LII->"
             "GC's --dg-ec-cluster-sigma, this clusters by GROUP membership, "
             "not neuron index -- CA3's groups are deliberately interleaved "
             "by index (see build_replay_network's `patterns`), so index-"
             "based clustering would not track pattern identity here. "
             "Requires --schaffer-k (meaningless at the default 100%% dense "
             "Schaffer, where every source is included regardless of bias).")
    parser.add_argument(
        "--cortical-recall", action="store_true",
        help="Test 3: consolidate, then LESION the hippocampus (zero CA1->EC), "
             "cue part of the cortical assembly and measure how much of the rest "
             "revives. This is the systems-consolidation test -- remote memory "
             "should survive hippocampal removal. Requires --mpfc; forces "
             "recurrent mPFC collaterals, without which cortex has nothing to "
             "complete a pattern with and the test is a guaranteed null.")
    parser.add_argument(
        "--cr-prime-rate", type=float, default=250.0, metavar="HZ",
        help="Tonic Poisson drive to mPFC after the lesion, holding cells near "
             "threshold so recurrent collaterals can propagate. Without it the "
             "test asks cortex to self-ignite from silence. Keep it "
             "subthreshold: the pre-cue baseline must stay ~0.")
    parser.add_argument(
        "--cr-prime-weight", type=float, default=1.5, metavar="W",
        help="Weight of the post-lesion mPFC priming drive.")
    parser.add_argument(
        "--het", type=float, default=0.0, metavar="CV",
        help="Model-wide heterogeneity: sets weight, delay AND neuron-parameter "
             "CV in one go, on EVERY cell and EVERY synapse. Neurons within a "
             "population are otherwise exact copies (only V_m varies, and that "
             "washes out at rest), and every synapse of a projection carries an "
             "identical weight and delay -- which is what creates the synchrony "
             "ceiling of §13. Try 0.2-0.4. Overridden per-channel by the three "
             "flags below.")
    parser.add_argument(
        "--het-wcomp", type=float, default=1.0, metavar="X",
        help="Scale applied to every EXCITATORY weight alongside --het, "
             "restoring the operating point that heterogeneity lowers. Without "
             "it CA1 PYR falls ~4x at cv 0.15. Inhibition is deliberately left "
             "alone: scaling both leaves E/I unchanged and only raises loop "
             "gain, which made CA3 SUP fall further (4.54 -> 3.57 Hz).")
    parser.add_argument(
        "--het-w-cv", type=float, default=None, metavar="CV",
        help="Weight CV on every projection (default: --het).")
    parser.add_argument(
        "--het-delay-cv", type=float, default=None, metavar="CV",
        help="Delay CV on every projection (default: --het).")
    parser.add_argument(
        "--het-neuron-cv", type=float, default=None, metavar="CV",
        help="Per-cell CV on Izhikevich a/b/c/d/I_e (default: --het). The "
             "calibrated value is kept as the mean, so the f-I calibration "
             "(e.g. the 4.97x MC_HIGH/MC_LOW rheobase ratio) is preserved in "
             "expectation.")
    parser.add_argument(
        "--dg-delay-jitter", type=float, default=0.0, metavar="MS",
        help="Per-synapse delay jitter (ms) on the DG pathway incl. the "
             "perforant path and mossy fibres. Spreads the EC volley in time, "
             "lifting the synchrony ceiling that caps w_ec_dg (see §13).")
    parser.add_argument(
        "--dg-neurogenesis", action="store_true",
        help="Phase 8: a new, hyperexcitable, under-inhibited cohort of "
             "granule cells is born each epoch and matures over "
             "--neurogenesis-maturation-epochs. Requires --dg. Population "
             "size only grows (no cell removal -- NEST has no clean mid-run "
             "node deletion).")
    parser.add_argument(
        "--neurogenesis-rate", type=float, default=0.01, metavar="F",
        help="Fraction of the ORIGINAL granule-cell count born as a new "
             "cohort each epoch (default 0.01 = 1%%).")
    parser.add_argument(
        "--neurogenesis-maturation-epochs", type=int, default=4, metavar="M",
        help="Epochs over which a cohort relaxes from young to mature "
             "(excitability, inhibition received, perforant-path gain).")
    parser.add_argument(
        "--neurogenesis-young-ie", type=float, default=4.0, metavar="PA",
        help="I_e offset (pA) at birth -- lowers rheobase, i.e. hyperexcitable. "
             "Relaxes to 0 (mature baseline) over the maturation window.")
    parser.add_argument(
        "--neurogenesis-young-inhib-scale", type=float, default=0.3, metavar="F",
        help="Fraction of the mature basket->GC weight a cohort receives at "
             "birth (0.3 = 70%% less inhibition than a mature cell). Rises "
             "to 1.0 over the maturation window.")
    parser.add_argument(
        "--neurogenesis-young-pp-scale", type=float, default=2.0, metavar="F",
        help="Phase 10: multiplier on the perforant-path (EC LII->GC) LEARNING "
             "RATE (--dg-assoc-a) a cohort has at birth -- enhanced LTP during "
             "the critical period, not a static weight boost. Decays to 1.0 "
             "(mature baseline rate) over the maturation window. Only has an "
             "effect together with --dg-perforant-stdp.")
    parser.add_argument(
        "--dg-perforant-stdp", action="store_true",
        help="Phase 10: Hebbian EC LII->GC learning (co-activation potentiates, "
             "post-without-pre weakly depresses -- same rule as the EC LV->mPFC "
             "association hook). Works standalone with just --dg (every "
             "granule cell learns at the mature baseline rate); combine with "
             "--dg-neurogenesis so young cohorts learn faster "
             "(--neurogenesis-young-pp-scale) while they are still young. "
             "Without this, young vs mature cells cannot differ in a way that "
             "depends on WHICH pattern they were exposed to -- only in "
             "age-indexed intrinsic properties. Requires --dg --ec-lii.")
    parser.add_argument(
        "--dg-assoc-a", type=float, default=0.01, metavar="A",
        help="Phase 10. Mature-baseline Hebbian potentiation per co-activation "
             "event (mpfc_assoc_hook's analogous default is 0.02). Needs "
             "empirical bracketing.")
    parser.add_argument(
        "--dg-assoc-a-hetero", type=float, default=0.002, metavar="A",
        help="Phase 10. Heterosynaptic depression per event when GC fires "
             "without its EC LII partner (mpfc_assoc_hook's analogous default "
             "is 0.004). NOT scaled by cohort age -- enhanced LTP in young "
             "cells is the specific literature claim, not enhanced LTD.")
    parser.add_argument(
        "--dg-assoc-w-max", type=float, default=1.5, metavar="W",
        help="Phase 10. Perforant-path weight ceiling from learning.")
    parser.add_argument(
        "--dg-assoc-w-min", type=float, default=0.05, metavar="W",
        help="Phase 10. Perforant-path weight floor from learning/depression.")
    parser.add_argument(
        "--w-cv", type=float, default=None, metavar="CV",
        help="Coefficient of variation for DG synaptic weights. Unset inherits "
             "--het; an explicit 0 forces scalar DG weights. This defaulted to "
             "0.0 and was passed through to fixed_connect, which shadowed the "
             "global setting -- JOBS H1/H2/H3 all printed cv=0.0 despite "
             "--het 0.30, so no DG synapse ever had weight heterogeneity.")
    parser.add_argument(
        "--w-ec-dg", type=float, default=0.15, metavar="W",
        help="Perforant path EC LII->GC weight. 0.15 is the synchrony-limited "
             "value; raise it together with --dg-delay-jitter.")
    parser.add_argument(
        "--pp-residual", type=float, default=0.95, metavar="F",
        help="Scale on the Poisson stand-in for unmodelled cortical drive to "
             "DG. Lowering it raises the pattern-carrying signal share.")
    parser.add_argument(
        "--dg-ec-cluster-sigma", type=float, default=0.0, metavar="SIGMA",
        help="Ring-distance width (same units as --place-field-sigma) of "
             "each granule cell's LOCAL EC LII input pool for the "
             "perforant path. 0 (default) keeps the original K=50 "
             "uniform-random sampling across all of EC LII, which measurably "
             "destroys pattern identity by K=50-sample averaging (RESULTS.md "
             "SS17). >0 biases each GC's 50 sources toward EC LII cells near "
             "its own topographic location (an even tiling by GC creation "
             "index stands in for a medial-lateral anatomical gradient) -- "
             "the bio-plausible assumption that closer axons make closer "
             "dendritic-tree synapses. Verified at 1%% scale, sigma=0.05: "
             "restores a real identity signal (DG mean ring-distance from "
             "the active pattern 0.273 -> 0.236, vs. EC LII's own 0.22 and "
             "a pure-chance null of 0.25) where reducing fan-in alone "
             "(K=1, no clustering) only reached 0.253. Requires --dg "
             "--ec-lii --pattern-source ec-lii (needs the real place-field "
             "centers, not a re-derived copy).")
    parser.add_argument(
        "--mpfc-k-rec", type=int, default=20, metavar="K",
        help="Recurrent mPFC->mPFC in-degree. The default 20 is a hard gate on "
             "Test 3: with N=1440 an uncued assembly cell then receives only "
             "~1.7 inputs from a 124-cell cue, so no weight can ignite it.")
    parser.add_argument(
        "--mpfc-w-rec", type=float, default=0.9, metavar="W",
        help="Recurrent mPFC->mPFC weight (initial; the recurrent hook learns "
             "from here).")
    parser.add_argument(
        "--cr-cue-frac", type=float, default=0.4, metavar="F",
        help="Fraction of the cortical assembly to cue in --cortical-recall.")
    parser.add_argument(
        "--train-pattern", type=int, default=None, metavar="P",
        help="Replay ONLY pattern P (keeping the --n-patterns partition) instead "
             "of alternating. Train A-only and B-only on the same seed to test "
             "whether the consolidated cortical trace is pattern-specific.")
    parser.add_argument(
        "--seed", type=int, default=None, metavar="S",
        help="Master RNG seed. Sets BOTH the NEST kernel RNG (Poisson drive, "
             "connectivity, delay jitter) and the numpy seed (V_m spread, DG "
             "wiring); varying only one leaves runs partially identical. Use to "
             "replicate an effect across independent networks.")
    parser.add_argument(
        "--schaffer-w-scale", type=float, default=1.0, metavar="X",
        help="Static multiplier on the Schaffer weights only. Control for "
             "--schaffer-stdp: STDP raises cortical rates, so a static run at "
             "matched drive separates path SELECTION from raw drive.")
    parser.add_argument(
        "--schaffer-stdp", action="store_true",
        help="Apply delay-aware pair STDP to CA3->CA1 between SWR events "
             "(Phase C step 2). dt = (t_post - t_pre) - delay_ij, so synapses "
             "whose own delay MATCHES the pre->post interval are potentiated. "
             "This is the mechanism that selects delay-matched paths; pair with "
             "--delay-jitter, which supplies the delay spread to select on.")
    parser.add_argument(
        "--n-patterns", type=int, default=1, metavar="P",
        help="Number of DISTINCT replay patterns (default 1). The CA3 sequence "
             "groups are split into P interleaved assemblies, each replayed in "
             "its own epochs. Needed for any engram/selectivity claim: with one "
             "pattern there is nothing to be selective ABOUT.")
    parser.add_argument(
        "--pattern-source", choices=["ca3", "ec-lii"], default="ca3", metavar="SRC",
        help="Where the pattern-defining drive attaches (Phase 7). 'ca3' "
             "(default) is the original replay/consolidation direction: the "
             "trigger/scaffold inject pattern identity straight into CA3 SUP, "
             "modelling an SWR reactivating an already-encoded memory -- every "
             "prior JOB H/Test-3 result used this. 'ec-lii' is the encoding "
             "direction (sensory/cortex -> EC -> DG -> CA3): CA3 gets NO direct "
             "injection at all, and a place-field-tuned drive (see "
             "--place-field-sigma) is wired onto EC LII instead, so CA3's "
             "assembly becomes a CONSEQUENCE of DG's own separation rather than "
             "an externally given label. Requires --ec-lii.")
    parser.add_argument(
        "--place-field-sigma", type=float, default=0.15, metavar="SIGMA",
        help="Only with --pattern-source ec-lii. Width (ring units, "
             "circumference 1) of each EC LII cell's place-field tuning curve. "
             "Controls how much adjacent patterns overlap -- smaller = more "
             "separated patterns, larger = more overlap. Needs empirical "
             "bracketing like every other DG/EC parameter here.")
    parser.add_argument(
        "--ec-pattern-base-rate", type=float, default=20.0, metavar="HZ",
        help="Only with --pattern-source ec-lii. Poisson floor rate for every "
             "EC LII cell during the pattern-drive window, regardless of "
             "tuning-curve gain. Needs empirical bracketing.")
    parser.add_argument(
        "--ec-pattern-peak-rate", type=float, default=200.0, metavar="HZ",
        help="Only with --pattern-source ec-lii. Additional Poisson rate at "
             "full tuning-curve gain (i.e. at a cell's own field centre). "
             "Needs empirical bracketing.")
    parser.add_argument(
        "--ec-pattern-weight", type=float, default=0.30, metavar="W",
        help="Only with --pattern-source ec-lii. Synaptic weight of the "
             "place-field drive onto EC LII. This is a K=1 (one dedicated "
             "generator per cell) pathway competing against the existing "
             "CA1->EC pathway (K=50, w=0.30) -- raising the rate alone "
             "plateaued (13-14%% active regardless of a 4x rate increase), "
             "so this needs to go well above 0.30 for the pattern signal to "
             "out-compete CA1's 50-fold convergence. Needs bracketing.")
    parser.add_argument(
        "--novel-pattern-onset", type=int, default=None, metavar="EPOCH",
        help="Phase 9 novelty-detection design. Only with --pattern-source "
             "ec-lii and --n-patterns >= 2. Holds back the LAST pattern "
             "index (n_patterns-1) -- it is never shown before this epoch, "
             "so its first-ever presentation can be compared against the "
             "already-familiar earlier patterns AT THE SAME EPOCH (an "
             "oddball/novelty-among-familiar design). After its debut it "
             "resumes cycling normally through all patterns. Unset (default) "
             "= no change, all patterns cycle evenly from epoch 0.")
    parser.add_argument(
        "--no-ec-dg-loop", action="store_true",
        help="Keep the EC LII->DG perforant path as a Poisson stand-in even when "
             "--ec-lii is present, leaving the EC->DG->CA3->CA1->EC loop OPEN. "
             "By default --dg + --ec-lii closes it. Use this to reproduce the "
             "pre-loop-closure behaviour or to isolate loop effects.")
    # ---- CA3 pattern-completion probe ---------------------------------------
    parser.add_argument(
        "--pattern-completion", action="store_true",
        help="Run the CA3 auto-association probe INSTEAD of the replay sim: cue "
             "a fraction of one assembly and measure how much of the rest the "
             "recurrent collaterals restore, intact vs sup_local-ablated control. "
             "Ignores STC/homeostasis/cortical flags.")
    parser.add_argument(
        "--pc-cue-fracs", type=str, default="0.1,0.2,0.3,0.5,0.7,1.0", metavar="F1,F2,...",
        help="Comma-separated cue fractions to sweep (default: 0.1..1.0). Each is "
             "cued on a distinct group; needs n_seq_groups >= count+2.")
    parser.add_argument(
        "--pc-cue-weight", type=float, default=2.5, metavar="W",
        help="Synaptic weight of the partial cue (default: 2.5).")
    # ---- Phase 1 cortical flag ----------------------------------------------
    parser.add_argument(
        "--ec-lii", action="store_true",
        help="Add EC LII/III population with static CA1→EC synapses (Phase 1). "
             "STDP weight updates applied by Python STC hook (Phase 2).")
    parser.add_argument(
        "--ec-lii-scale", type=int, default=None, metavar="PCT",
        help="EC LII scale as independent percent of 100k reference neurons. "
             "Defaults to --scale if omitted.")
    parser.add_argument(
        "--ec-lii-k", type=int, default=50, metavar="K",
        help="Target in-degree K for the CA1→EC projection (default: 50).")
    # ---- Phase 2 STC flags --------------------------------------------------
    parser.add_argument(
        "--stc", action="store_true",
        help="Enable Phase 2 STC hook: Python STDP + tag/PRP between SWR epochs. "
             "Requires --ec-lii.")
    parser.add_argument(
        "--n-swr", type=int, default=7, metavar="N",
        help="Number of SWR epochs for multi-epoch consolidation run (default: 7). "
             "Each epoch simulates one forward+reverse SWR pair. "
             "Total sim time = n_swr × epoch_ms.")
    parser.add_argument(
        "--epoch-ms", type=float, default=1000.0, metavar="MS",
        help="Duration of each SWR epoch in ms (default: 1000). "
             "The SWR events are placed at fixed offsets within each epoch.")
    # ---- Phase 3 cortical loop flags ----------------------------------
    parser.add_argument(
        "--ec-lv", action="store_true",
        help="Add EC Layer V population + CA1→LV + ECLII→LV + LV→CA3 feedback "
             "(Phase 3). Requires --ec-lii.")
    parser.add_argument(
        "--mpfc", action="store_true",
        help="Add mPFC population receiving EC LV input (Phase 3 endpoint). "
             "Requires --ec-lv.")
    parser.add_argument(
        "--no-mpfc-lateral-inh", action="store_true",
        help="Disable mPFC lateral inhibition. Without it every mPFC cell fires "
             "on every SWR, so the EC LV->mPFC association is non-selective and "
             "no engram subset forms.")
    parser.add_argument(
        "--no-mpfc-assoc", action="store_true",
        help="Disable the EC LV->mPFC associative (Hebbian) build-up that forms "
             "the cortical engram. By default --mpfc enables it; without it the "
             "LV->mPFC weights are static and no association ever forms.")
    parser.add_argument(
        "--prp-threshold", type=float, default=3.0, metavar="T",
        help="PRP pool threshold for L-LTP capture (default: 3.0 = 3 SWR events). "
             "Set to 999 for Phase 5 falsification experiment (blocks L-LTP while "
             "preserving E-LTP and tagging, isolating replay from consolidation).")
    # ---- Phase 4 homeostasis flags ------------------------------------------
    parser.add_argument(
        "--homeostasis", action="store_true",
        help="Enable Phase 4 synaptic homeostasis: after all SWR epochs, "
             "multiplicatively downscale Schaffer collateral and CA3 recurrent "
             "excitatory weights by --homeo-alpha, then run one verification "
             "epoch to confirm the cortical (EC) trace survives hippocampal "
             "downscaling. Requires --stc.")
    parser.add_argument(
        "--homeo-alpha", type=float, default=0.75, metavar="A",
        help="Downscaling factor for Phase 4 homeostasis (default: 0.75). "
             "Biology: ~0.75 per sleep night (Vyazovskiy et al. 2008). "
             "Applied multiplicatively to all Schaffer and CA3 exc synapses. "
             "L-LTP synapses in CA1->EC are exempt.")
    parser.add_argument(
        "--alpha-sweep", type=str, default=None, metavar="A1,A2,...",
        help="Phase 4 multi-alpha sweep: comma-separated list of alpha values, "
             "e.g. '0.50,0.75,0.90'. Runs ONE consolidation, then loops over "
             "alphas (restore checkpoint -> apply alpha -> run verification "
             "epoch -> record metrics). Saves ~10 hours of compute vs separate "
             "jobs. Overrides --homeo-alpha. Requires --homeostasis --stc.")
    args = parser.parse_args()

    # Model-wide heterogeneity must be set BEFORE any Create/Connect call.
    _pick = lambda v: args.het if v is None else v
    set_heterogeneity(w_cv=_pick(args.het_w_cv),
                      delay_cv=_pick(args.het_delay_cv),
                      neuron_cv=_pick(args.het_neuron_cv),
                      wcomp=args.het_wcomp)

    # Parse alpha sweep into a list of floats; empty list = single-alpha mode
    if args.alpha_sweep is not None:
        try:
            args.alpha_list = [float(a) for a in args.alpha_sweep.split(",") if a.strip()]
        except ValueError:
            parser.error("--alpha-sweep must be comma-separated floats, "
                         f"got '{args.alpha_sweep}'")
        if not args.alpha_list:
            parser.error("--alpha-sweep cannot be empty")
        if not args.homeostasis:
            parser.error("--alpha-sweep requires --homeostasis")
    else:
        args.alpha_list = [args.homeo_alpha] if args.homeostasis else []

    if args.stc and not args.ec_lii:
        parser.error("--stc requires --ec-lii")
    if args.ec_lv and not args.ec_lii:
        parser.error("--ec-lv requires --ec-lii")
    if args.mpfc and not args.ec_lv:
        parser.error("--mpfc requires --ec-lv")
    if args.homeostasis and not args.stc:
        parser.error("--homeostasis requires --stc (Phase 2 must be active)")
    if args.pattern_source == "ec-lii" and not args.ec_lii:
        parser.error("--pattern-source ec-lii requires --ec-lii")
    if args.dg_ec_cluster_sigma > 0:
        if not args.dg:
            parser.error("--dg-ec-cluster-sigma requires --dg")
        if not args.ec_lii:
            parser.error("--dg-ec-cluster-sigma requires --ec-lii")
        if args.pattern_source != "ec-lii":
            parser.error("--dg-ec-cluster-sigma requires --pattern-source "
                          "ec-lii (needs the real place-field centers)")
    if args.schaffer_group_frac > 0 and args.schaffer_k is None:
        parser.error("--schaffer-group-frac requires --schaffer-k (meaningless "
                      "at the default 100% dense Schaffer projection)")
    if args.dg_neurogenesis and not args.dg:
        parser.error("--dg-neurogenesis requires --dg")
    if args.dg_neurogenesis and not args.ec_lii:
        parser.error("--dg-neurogenesis requires --ec-lii (perforant-path source "
                      "for new cohorts)")
    if args.novel_pattern_onset is not None:
        if args.pattern_source != "ec-lii":
            parser.error("--novel-pattern-onset requires --pattern-source ec-lii")
        if args.n_patterns < 2:
            parser.error("--novel-pattern-onset requires --n-patterns >= 2 "
                          "(the last pattern index is the one held back)")
    if args.dg_perforant_stdp and not args.dg:
        parser.error("--dg-perforant-stdp requires --dg")
    if args.dg_perforant_stdp and not args.ec_lii:
        parser.error("--dg-perforant-stdp requires --ec-lii (perforant-path source)")
    if args.dg_perforant_stdp and not args.dg_neurogenesis:
        parser.error("--dg-perforant-stdp requires --dg-neurogenesis: plasticity "
                      "only applies to neurogenesis cohorts (the original GC "
                      "population's cache is computationally infeasible at MN5 "
                      "scale, see build_dg_neurogenesis_hook), so without "
                      "--dg-neurogenesis there is nothing for it to update")

    cfg = build_scale_config(args.scale)

    n_threads = (args.threads
                 if args.threads is not None
                 else int(os.environ.get("OMP_NUM_THREADS", cfg["n_threads_default"])))

    total_N  = sum(cfg[k] for k in _REF_100PCT)
    n_groups = cfg["n_seq_groups"]

    # ---- CA3 pattern-completion probe (self-contained; exits after) ---------
    if args.pattern_completion:
        try:
            cue_fracs = [float(x) for x in args.pc_cue_fracs.split(",") if x.strip()]
        except ValueError:
            parser.error(f"--pc-cue-fracs must be comma-separated floats, "
                         f"got '{args.pc_cue_fracs}'")
        if len(cue_fracs) + 2 > n_groups:
            parser.error(f"--pattern-completion needs n_seq_groups >= "
                         f"{len(cue_fracs)+2} for {len(cue_fracs)} cue fractions "
                         f"(scale {args.scale} gives {n_groups}); raise --scale "
                         f"or pass fewer --pc-cue-fracs.")
        print(f"\n{'='*72}")
        print(f"  CA3 PATTERN COMPLETION probe  [{cfg['label']}]  "
              f"cue_fracs={cue_fracs}  cue_weight={args.pc_cue_weight}")
        print(f"{'='*72}")
        print(">>> Intact network (sup_local recurrence ON)...")
        intact = run_pattern_completion(cfg, n_threads, cue_fracs,
                                        ablate=False, cue_weight=args.pc_cue_weight)
        print(">>> Ablated control (sup_local = 0)...")
        ablated = run_pattern_completion(cfg, n_threads, cue_fracs,
                                         ablate=True, cue_weight=args.pc_cue_weight)
        print_pattern_completion(intact, ablated)

        if _HDF5_AVAILABLE and _mpi_rank() == 0:
            import h5py, datetime
            out_dir = f"replay_output_{args.scale}pct"
            os.makedirs(out_dir, exist_ok=True)
            pc_path = args.out_hdf5 or os.path.join(out_dir, "pattern_completion.h5")
            with h5py.File(pc_path, "w") as h5:
                h5.attrs["created_utc"] = datetime.datetime.utcnow().isoformat()
                h5.attrs["scale"]       = cfg["label"]
                h5.attrs["cue_weight"]  = args.pc_cue_weight
                for name, res in [("intact", intact), ("ablated", ablated)]:
                    g = h5.create_group(name)
                    g.create_dataset("cue_frac",   data=np.array([r["cue_frac"] for r in res], dtype=np.float32))
                    g.create_dataset("completion", data=np.array([r["completion"] for r in res], dtype=np.float32))
                    g.create_dataset("completion_baseline", data=np.array([r["completion_baseline"] for r in res], dtype=np.float32))
                    g.create_dataset("cue_recall", data=np.array([r["cue_recall"] for r in res], dtype=np.float32))
                    g.create_dataset("group",      data=np.array([r["group"] for r in res], dtype=np.int32))
            print(f"\n>>> Saved pattern-completion data -> {pc_path}")
        sys.exit(0)

    ec_lii_pct = args.ec_lii_scale if args.ec_lii_scale is not None else args.scale
    N_ec_lii   = _round_to_multiple(
        _REF_CORTEX["N_ec_lii"] * ec_lii_pct / 100.0,
        n_groups,
    )

    # DG population sizes (Phase 6.2). Scaled by their own pct, rounded like the
    # cortex so hippocampal maths is untouched when --dg is absent.
    dg_pct = args.dg_scale if args.dg_scale is not None else args.scale
    N_dg = {k: _round_to_multiple(v * dg_pct / 100.0, n_groups)
            for k, v in _REF_DG.items()}

    # Total simulation time: either single sim_ms OR n_swr × epoch_ms
    SIM_MS       = args.sim_ms if not args.stc else args.epoch_ms
    n_epochs     = args.n_swr if args.stc else 1
    total_sim_ms = SIM_MS * n_epochs

    print(f"\n{'='*72}")
    print(f"  Watson et al. 2025 — Bidirectional Replay  [v10: Phase 3 EC-LV + mPFC loop]")
    print(f"  Scale    : {cfg['label']}")
    print(f"  CA3_SUP  : {cfg['N_ca3_sup']:>10,}  CA3_DEEP : {cfg['N_ca3_deep']:>8,}")
    print(f"  CA1_PYR  : {cfg['N_ca1_pyr']:>10,}  groups   : {n_groups:>8,}  "
          f"(CA3_SUP/group = {cfg['N_ca3_sup']//n_groups})")
    print(f"  Total N  : {total_N:>10,}")
    if args.dg:
        print(f"  DG       : GC={N_dg['N_dg_gc']:>9,}  MC_low={N_dg['N_dg_mc_low']:,}  "
              f"MC_high={N_dg['N_dg_mc_high']:,}  BSK={N_dg['N_dg_basket']:,}  "
              f"({dg_pct}% of rat DG)  [--dg]")
    if args.ec_lii:
        print(f"  EC LII   : {N_ec_lii:>10,}  ({ec_lii_pct}% of 100k ref)  "
              f"K={args.ec_lii_k}  [--ec-lii]")
    if args.stc:
        print(f"  STC hook : ENABLED  n_swr={n_epochs}  "
              f"epoch={SIM_MS:.0f} ms  total={total_sim_ms:.0f} ms  [--stc]  "
              f"PRP_threshold={args.prp_threshold}"
              + ("  [PHASE-5 FALSIFICATION: L-LTP blocked]" if args.prp_threshold > 100 else ""))
    if args.ec_lv:
        print(f"  EC LV    : ENABLED  [Phase 3 — closes hippocampo-cortical loop]")
    if args.mpfc:
        print(f"  mPFC     : ENABLED  [Phase 3 — cortical engram endpoint]")
    print(f"  Threads  : {n_threads}  |  Sim: {total_sim_ms:.0f} ms")
    print(f"  Connect  : fixed_indegree + pairwise_bernoulli (C++, OpenMP)")
    print(f"{'='*72}\n")

    t_wall = time.perf_counter()

    print(">>> Building hippocampal network...")
    net = build_replay_network(
        N_ca3_sup      = cfg["N_ca3_sup"],
        N_ca3_deep     = cfg["N_ca3_deep"],
        N_ca3_int_sup  = cfg["N_ca3_int_sup"],
        N_ca3_int_deep = cfg["N_ca3_int_deep"],
        N_ca1_pyr      = cfg["N_ca1_pyr"],
        N_ca1_basket   = cfg["N_ca1_basket"],
        N_ca1_olm      = cfg["N_ca1_olm"],
        n_seq_groups   = cfg["n_seq_groups"],
        n_threads      = n_threads,
        suppress_dg_drive = args.dg,   # real DG mossy fibres replace the proxy
        n_patterns     = args.n_patterns,
        train_pattern  = args.train_pattern,
        pattern_source = args.pattern_source,
        n_epochs       = n_epochs,
        epoch_ms       = SIM_MS,
        delay_jitter   = args.delay_jitter,
        delay_jitter_wcomp = args.delay_jitter_wcomp,
        schaffer_k     = args.schaffer_k,
        schaffer_w_scale = args.schaffer_w_scale,
        schaffer_group_frac = args.schaffer_group_frac,
        **({} if args.seed is None else
           dict(master_seed=args.seed, seed_connect=args.seed)),
    )

    # ---- Phase 9: novel-pattern-held-back schedule (--novel-pattern-onset) --
    # Overwrites net["epoch_pattern"] (NOT build_replay_network's internal
    # copy) so pattern_discrimination() -- which reads net["epoch_pattern"]
    # at call time, after the loop finishes -- groups epochs the same way
    # the actual drive was scheduled. Only touches --pattern-source ec-lii:
    # the last pattern index (n_patterns-1) is held back and never shown
    # before epoch args.novel_pattern_onset, so its first-ever presentation
    # can be compared against the already-familiar earlier patterns at that
    # SAME epoch (an oddball/novelty design, not a temporal-tagging one).
    if args.novel_pattern_onset is not None:
        onset  = args.novel_pattern_onset
        n_pat  = args.n_patterns
        novel_schedule = []
        for ep in range(max(1, n_epochs)):
            if ep < onset:
                novel_schedule.append(ep % (n_pat - 1))       # familiar only
            elif ep == onset:
                novel_schedule.append(n_pat - 1)               # novel pattern's debut
            else:
                novel_schedule.append((ep - onset - 1) % n_pat)  # resume full cycle
        net["epoch_pattern"] = novel_schedule
        print(f">>> Novel-pattern schedule: pattern {n_pat-1} held back until "
              f"epoch {onset}, then cycling normally: {novel_schedule}")

    # ---- Optional Phase 1: EC LII/III ----------------------------------------
    # Built BEFORE the DG so its population can supply the perforant path and
    # close the EC->DG->CA3->CA1->EC loop. EC LII itself only needs CA1, which
    # build_replay_network has already created.
    ec_module = None
    if args.ec_lii:
        print(">>> Building EC LII/III module...")
        ec_module = build_ec_lii(
            ca1_pyr       = net["PYR"],
            ca1_spike_rec = net["spk_pyr"],
            N_ec_lii      = N_ec_lii,
            K_ca1_ec      = args.ec_lii_k,
            delay_jitter  = args.delay_jitter,
            delay_jitter_wcomp = args.delay_jitter_wcomp,
        )

    # ---- Phase 7: encoding-direction pattern drive onto EC LII ---------------
    # Only runs with --pattern-source ec-lii (validated above to require
    # --ec-lii). This is the counterpart of the CA3 trigger/scaffold inside
    # build_replay_network, just attached to EC LII instead and issued here
    # because EC LII does not exist until build_ec_lii() has run. CA3 never
    # receives any direct pattern-specific injection in this mode -- see
    # build_replay_network's trigger_on/scaffold_on gating.
    ec_field_centers = None   # only set below; --dg-ec-cluster-sigma needs it
    if args.pattern_source == "ec-lii":
        print(">>> Wiring encoding-direction place-field drive onto EC LII...")
        _pf_rng = np.random.default_rng(
            (args.seed if args.seed is not None else 42) + 13)
        ec_field_centers = assign_place_fields(N_ec_lii, _pf_rng)
        n_pat = max(1, net["n_patterns"])
        for ep in range(max(1, n_epochs)):
            t0_ep = ep * SIM_MS
            pat_idx = (net["epoch_pattern"][ep] if ep < len(net["epoch_pattern"])
                       else ep % n_pat)
            pattern_center = pat_idx / n_pat
            f_s, r_s = t0_ep + net["swr_fwd"][0], t0_ep + net["swr_rev"][0]
            f_dur = net["swr_fwd"][1] - net["swr_fwd"][0]
            r_dur = net["swr_rev"][1] - net["swr_rev"][0]
            make_place_field_drive(
                ec_module.population, ec_field_centers, pattern_center,
                args.place_field_sigma, f_s, f_dur,
                args.ec_pattern_base_rate, args.ec_pattern_peak_rate,
                weight=args.ec_pattern_weight)
            make_place_field_drive(
                ec_module.population, ec_field_centers, pattern_center,
                args.place_field_sigma, r_s, r_dur,
                args.ec_pattern_base_rate, args.ec_pattern_peak_rate,
                weight=args.ec_pattern_weight)
        print(f"    {n_pat} pattern(s) as ring locations, sigma="
              f"{args.place_field_sigma}, base={args.ec_pattern_base_rate} Hz, "
              f"peak={args.ec_pattern_peak_rate} Hz  ** CA3 gets no direct "
              f"injection in this mode **")

    # ---- Phase C step 2: Schaffer STDP --------------------------------------
    schaffer_hook = None
    if args.schaffer_stdp:
        schaffer_hook = build_schaffer_stdp_hook(net)

    # ---- Optional Phase 6.2: dentate gyrus -----------------------------------
    dg_module = None
    if args.dg:
        print(">>> Building dentate gyrus module (Phase 6.2)...")
        dg_module = build_dg_module(
            ca3_sup   = net["CA3_SUP"],
            ca3_deep  = net["CA3_DEEP"],
            N_gc      = N_dg["N_dg_gc"],
            N_mc_low  = N_dg["N_dg_mc_low"],
            N_mc_high = N_dg["N_dg_mc_high"],
            N_basket  = N_dg["N_dg_basket"],
            ec_lii    = (ec_module.population
                         if (ec_module is not None and not args.no_ec_dg_loop)
                         else None),
            w_ec_dg      = args.w_ec_dg,
            pp_residual  = args.pp_residual,
            delay_jitter = args.dg_delay_jitter,
            w_cv         = args.w_cv,
            ec_field_centers    = ec_field_centers,
            dg_ec_cluster_sigma = args.dg_ec_cluster_sigma,
            # Was hardcoded to the default (42) regardless of --seed, so
            # every run drew the IDENTICAL granule-cell V_m/Poisson-residual
            # heterogeneity -- confirmed 2026-09-05 (jobs 45358807/45358815,
            # different --seed) via 66% overlap in DG's "always-on" cells
            # between the two runs, versus ~8 expected by chance. Threaded
            # through now so at least this bias varies by --seed; see
            # run_dg_residual_refresh_hook for the deeper per-epoch fix.
            seed_connect = (args.seed if args.seed is not None else 42),
        )

    # ---- Optional Phase 8: DG neurogenesis -----------------------------------
    # --dg-perforant-stdp requires --dg-neurogenesis (validated above): Phase
    # 10 plasticity only ever applies to neurogenesis cohorts, so the hook is
    # only needed when neurogenesis itself is on.
    neurogenesis_hook = None
    if args.dg_neurogenesis:
        print(">>> Initialising DG neurogenesis hook (Phase 8) / "
              "perforant-path plasticity (Phase 10)...")
        neurogenesis_hook = build_dg_neurogenesis_hook(
            dg_module, ec_module.population, net["CA3_SUP"], net["CA3_DEEP"],
            maturation_epochs = args.neurogenesis_maturation_epochs,
            young_ie          = args.neurogenesis_young_ie,
            young_inhib_scale = args.neurogenesis_young_inhib_scale,
            young_pp_scale    = args.neurogenesis_young_pp_scale,
            w_ec_dg           = args.w_ec_dg,
            **({} if args.seed is None else dict(seed=args.seed)),
        )

    # ---- Optional Phase 3: EC Layer V + mPFC ---------------------------------
    # Built BEFORE the plasticity hooks: nest.GetConnections() descriptors are
    # invalidated by any later Connect(), and NEST warns about exactly that.
    # (Previously build_stc_hook ran first and its cached CA1->EC collection was
    # invalidated by these two builds.)
    eclv_module  = None
    mpfc_module  = None
    if args.ec_lv and ec_module is not None:
        print(">>> Building EC Layer V module (Phase 3)...")
        eclv_module = build_ec_lv(
            ca1_pyr    = net["PYR"],
            ec_lii_pop = ec_module.population,
            ca3_sup    = net["CA3_SUP"],
            delay_jitter = args.delay_jitter,
            delay_jitter_wcomp = args.delay_jitter_wcomp,
        )
        if args.mpfc:
            print(">>> Building mPFC module (Phase 3)...")
            mpfc_module = build_mpfc(
                ec_lv_pop = eclv_module.population,
                lateral_inhibition = not args.no_mpfc_lateral_inh,
                recurrent = args.cortical_recall,
                K_rec = args.mpfc_k_rec,
                w_rec = args.mpfc_w_rec,
                delay_jitter = args.delay_jitter,
                delay_jitter_wcomp = args.delay_jitter_wcomp,
            )

    # ---- Plasticity hooks: after ALL populations are wired -------------------
    stc_hook = None
    if args.stc and ec_module is not None:
        print(">>> Initialising STC hook (Phase 2)...")
        stc_hook = build_stc_hook(ec_module)

    mpfc_assoc_hook = None
    mpfc_rec_hook   = None
    if mpfc_module is not None and eclv_module is not None and not args.no_mpfc_assoc:
        print(">>> Initialising mPFC association hook...")
        mpfc_assoc_hook = build_mpfc_assoc_hook(mpfc_module, eclv_module)
        if args.cortical_recall:
            # the recurrent collaterals are what a hippocampus-independent
            # memory is actually stored in, so they must learn too
            mpfc_rec_hook = build_mpfc_recurrent_hook(mpfc_module)

    # ---- Simulation: single epoch or multi-epoch STC loop -------------------
    swr_fwd = net["swr_fwd"]
    swr_rev = net["swr_rev"]

    print(f"\n>>> Simulating {n_epochs} epoch(s) × {SIM_MS:.0f} ms "
          f"= {total_sim_ms:.0f} ms total...")
    t_sim = time.perf_counter()

    dg_residual_rng = (np.random.default_rng((args.seed if args.seed is not None else 42) + 61)
                        if dg_module is not None else None)
    for epoch in range(n_epochs):
        epoch_t0 = epoch * SIM_MS
        if dg_module is not None:
            # Re-randomize the original population's Poisson-residual rates
            # every epoch -- see run_dg_residual_refresh_hook's docstring for
            # why a fixed per-cell draw was silently defeating every DG
            # pattern-separation mechanism tried so far.
            run_dg_residual_refresh_hook(dg_module, dg_residual_rng)
        if neurogenesis_hook is not None and args.dg_neurogenesis:
            run_dg_neurogenesis_hook(neurogenesis_hook, dg_module, epoch,
                                     args.neurogenesis_rate)
        nest.Simulate(SIM_MS)

        # Phase 10: DG perforant-path Hebbian learning, same two SWR windows
        # as every other plasticity hook. Runs whenever the hook exists (i.e.
        # --dg-neurogenesis + --dg-perforant-stdp were given) -- plasticity
        # applies only to neurogenesis cohorts, not the original population
        # (see build_dg_neurogenesis_hook's docstring for why), with young
        # cohorts getting a learning-rate boost while inside their
        # maturation window.
        if neurogenesis_hook is not None and args.dg_perforant_stdp:
            for _ws, _we in ((swr_fwd[0], swr_fwd[1]), (swr_rev[0], swr_rev[1])):
                run_dg_perforant_plasticity_hook(
                    neurogenesis_hook, dg_module, ec_module,
                    t_swr_start=epoch_t0 + _ws, t_swr_end=epoch_t0 + _we,
                    epoch=epoch,
                    A_assoc=args.dg_assoc_a, A_hetero=args.dg_assoc_a_hetero,
                    w_max=args.dg_assoc_w_max, w_min=args.dg_assoc_w_min,
                )

        if stc_hook is not None:
            current_t = epoch_t0 + SIM_MS
            # Run STC hook for the forward SWR event in this epoch
            run_stc_hook(
                stc_hook, ec_module,
                t_swr_start   = epoch_t0 + swr_fwd[0],
                t_swr_end     = epoch_t0 + swr_fwd[1],
                current_t_ms  = current_t,
                PRP_threshold = args.prp_threshold,
            )
            # Run STC hook for the reverse SWR event
            run_stc_hook(
                stc_hook, ec_module,
                t_swr_start   = epoch_t0 + swr_rev[0],
                t_swr_end     = epoch_t0 + swr_rev[1],
                current_t_ms  = current_t,
                PRP_threshold = args.prp_threshold,
            )

        if schaffer_hook is not None:
            for _ws, _we in ((swr_fwd[0], swr_fwd[1]), (swr_rev[0], swr_rev[1])):
                run_schaffer_stdp_hook(schaffer_hook, net,
                                       epoch_t0 + _ws, epoch_t0 + _we)

        if mpfc_rec_hook is not None:
            for _ws, _we in ((swr_fwd[0], swr_fwd[1]), (swr_rev[0], swr_rev[1])):
                run_mpfc_assoc_hook(mpfc_rec_hook, mpfc_module, mpfc_module,
                                    t_swr_start=epoch_t0 + _ws,
                                    t_swr_end=epoch_t0 + _we)

        # Cortical association build-up: same two SWR windows, EC LV -> mPFC
        if mpfc_assoc_hook is not None:
            for _ws, _we in ((swr_fwd[0], swr_fwd[1]), (swr_rev[0], swr_rev[1])):
                run_mpfc_assoc_hook(
                    mpfc_assoc_hook, eclv_module, mpfc_module,
                    t_swr_start = epoch_t0 + _ws,
                    t_swr_end   = epoch_t0 + _we,
                )

        if n_epochs > 1:
            print(f"    Epoch {epoch+1}/{n_epochs} done "
                  f"({time.perf_counter()-t_sim:.1f}s elapsed)")

    print(f"    Total simulation done in {time.perf_counter()-t_sim:.1f}s")

    # ---- Test 3: hippocampus-independent cortical recall --------------------
    if args.cortical_recall and mpfc_module is not None:
        print(f"\n{'='*72}")
        print("CORTICAL RECALL AFTER HIPPOCAMPAL LESION (Test 3)")
        print(f"{'='*72}")
        _t_all = total_sim_ms

        # 1. the cortical assembly = mPFC cells the consolidated pattern drives
        _ev = nest.GetStatus(mpfc_module.spike_rec, "events")[0]
        _t, _s = np.asarray(_ev["times"]), np.asarray(_ev["senders"])
        _m = np.zeros(len(_t), dtype=bool)
        for _e in range(n_epochs):
            _o = _e * SIM_MS
            _m |= (((_t >= _o + swr_fwd[0]) & (_t <= _o + swr_fwd[1])) |
                   ((_t >= _o + swr_rev[0]) & (_t <= _o + swr_rev[1])))
        _assembly = np.unique(_s[_m])
        print(f"  cortical assembly: {len(_assembly)}/{mpfc_module.N} mPFC cells "
              f"active during replay")
        if len(_assembly) < 8:
            print("  [SKIP] assembly too small to cue and score")
        else:
            # 2. lesion: hippocampal output to cortex is severed
            lesion_hippocampus(net, ec_module, eclv_module, stc=stc_hook)
            # 3. tonic priming. After the lesion mPFC has NO input at all -- EC
            # LV is silent, so the only remaining excitation is the recurrent
            # collaterals, which cannot start from nothing. The CA3 completion
            # probe needed exactly this (a sharp-wave-like depolarised state)
            # before it could complete anything; without it the recall test
            # measures whether cortex can self-ignite from silence, which no
            # memory would pass. Subthreshold on its own -- it holds cells near
            # threshold, it does not fire them (verified by the pre-cue
            # baseline, which must stay ~0).
            if args.cr_prime_rate > 0:
                _pg = nest.Create("poisson_generator", mpfc_module.N,
                                  params={"rate": float(args.cr_prime_rate)})
                nest.Connect(_pg, mpfc_module.population, conn_spec="one_to_one",
                             syn_spec={"weight": float(args.cr_prime_weight),
                                       "delay": 1.0})
                print(f"  [prime] mPFC tonic drive {args.cr_prime_rate:.0f} Hz "
                      f"@ w={args.cr_prime_weight} (subthreshold; baseline check below)")
            # 3. partial cue, well clear of the last replay
            _cue_t = _t_all + 200.0
            _cued, _un = cortical_recall_probe(
                mpfc_module, _assembly, args.cr_cue_frac, _cue_t,
                rng=np.random.default_rng(7))
            nest.Simulate(500.0)
            _ev2 = nest.GetStatus(mpfc_module.spike_rec, "events")[0]
            _t2, _s2 = np.asarray(_ev2["times"]), np.asarray(_ev2["senders"])
            _r = completion_index(_t2, _s2, _cued, _un, _cue_t, _cue_t + 80.0)
            # baseline: an equal window BEFORE the cue, post-lesion, no stimulus
            _b = completion_index(_t2, _s2, _cued, _un, _t_all + 100.0, _t_all + 180.0)
            print(f"  cue {args.cr_cue_frac*100:.0f}% of assembly "
                  f"({_r['n_cued']} cued, {_r['n_uncued']} to recover)")
            print(f"  cue_recall      {_r['cue_recall']:.3f}   (sanity: cued cells did fire)")
            print(f"  COMPLETION      {_r['completion']:.3f}")
            print(f"  pre-cue baseline{_b['completion']:8.3f}   "
                  f"(post-lesion spontaneous)")
            _net_c = _r['completion'] - _b['completion']
            print(f"  completion above baseline = {_net_c:+.3f}")
            if _net_c > 0.25:
                print("  [OK] cortex reactivates the pattern WITHOUT the hippocampus")
                print("       -> the memory has become hippocampus-independent.")
            else:
                print("  [FLAG] no cortical recall above baseline: the trace exists in")
                print("         weights but cannot reconstruct the pattern on its own.")
        print(f"{'='*72}")

    # ---- Phase 4: Synaptic Homeostasis (after all SWR epochs) ---------------
    # Supports both single-alpha (--homeo-alpha) and sweep (--alpha-sweep) modes.
    # In sweep mode: scan CA3 incoming connections ONCE, checkpoint post-
    # consolidation weights, then for each alpha: restore -> apply -> verify.
    # This avoids redundant build + 14-epoch consolidation for each alpha,
    # saving ~4-5 hours per additional alpha vs separate SLURM jobs.
    homeo_results = {}    # alpha (float) -> stats dict
    homeo         = None
    if args.homeostasis and stc_hook is not None:
        n_sweep = len(args.alpha_list)
        sweep_label = "alpha-sweep" if n_sweep > 1 else "single-alpha"
        print(f"\n>>> Phase 4: Synaptic homeostasis ({sweep_label}, "
              f"{n_sweep} alpha value(s): {args.alpha_list})")

        # Build hook ONCE: this is the expensive GetConnections(target=CA3_SUP) scan.
        # Use the first alpha as a placeholder; we override it inside the loop.
        homeo = build_homeostasis_hook(net, alpha=args.alpha_list[0])

        # Checkpoint: snapshot post-consolidation CA3 recurrent weights so we
        # can restore them between alpha iterations (otherwise alphas compound).
        ca3_weights_checkpoint = homeo.weights.copy()
        print(f"  [homeo] Checkpointed {len(ca3_weights_checkpoint):,} CA3 incoming "
              f"weights for {n_sweep} sweep iteration(s)")

        for sweep_idx, alpha_val in enumerate(args.alpha_list):
            print(f"\n  ---- alpha = {alpha_val:.3f}  "
                  f"({sweep_idx+1}/{n_sweep}) ----")

            # Restore checkpoint before each alpha (idempotent application)
            homeo.weights[:]    = ca3_weights_checkpoint
            homeo.weights_pre[:] = ca3_weights_checkpoint
            homeo.alpha         = alpha_val

            # Apply downscaling
            stats = run_homeostasis_hook(homeo)

            # Verification epoch: SIM_MS of additional simulation post-downscaling.
            # Each iteration extends total simulation by SIM_MS; SWR generators
            # already fired during epoch 1, so the verification window picks up
            # CA3 attractor dynamics under the downscaled weights.
            print(f"  [homeo] Running verification epoch ({SIM_MS:.0f} ms)...")
            t_verify = time.perf_counter()
            nest.Simulate(SIM_MS)

            # Verification epoch global start time = consolidation total + prior verifications
            epoch_t0_verify = (n_epochs + sweep_idx) * SIM_MS

            # Continue STC bookkeeping (writes are no-ops for already-LLTP synapses)
            run_stc_hook(
                stc_hook, ec_module,
                t_swr_start   = epoch_t0_verify + swr_fwd[0],
                t_swr_end     = epoch_t0_verify + swr_fwd[1],
                current_t_ms  = epoch_t0_verify + SIM_MS,
                PRP_threshold = args.prp_threshold,
            )
            run_stc_hook(
                stc_hook, ec_module,
                t_swr_start   = epoch_t0_verify + swr_rev[0],
                t_swr_end     = epoch_t0_verify + swr_rev[1],
                current_t_ms  = epoch_t0_verify + SIM_MS,
                PRP_threshold = args.prp_threshold,
            )
            print(f"    Verification epoch done in {time.perf_counter()-t_verify:.1f}s")

            # Measure replay quality on this verification epoch's SWR windows
            from nest import GetStatus as _gs
            _ev3 = _gs(net["spk_ca3_sup"], "events")[0]
            _t3  = np.array(_ev3["times"],   dtype=float)
            _s3  = np.array(_ev3["senders"], dtype=int)
            _rho_f, _pf = replay_score(_t3, _s3, net["ca3_seq_groups"],
                                        epoch_t0_verify + swr_fwd[0],
                                        epoch_t0_verify + swr_fwd[1])
            _rho_r, _pr = replay_score(_t3, _s3, net["ca3_seq_groups"],
                                        epoch_t0_verify + swr_rev[0],
                                        epoch_t0_verify + swr_rev[1])
            stats["rho_fwd_post_homeo"] = float(_rho_f) if _rho_f is not None else float("nan")
            stats["rho_rev_post_homeo"] = float(_rho_r) if _rho_r is not None else float("nan")
            stats["verification_t0_ms"] = float(epoch_t0_verify)
            stats["verification_t1_ms"] = float(epoch_t0_verify + SIM_MS)

            print(f"  [homeo] alpha={alpha_val:.2f}:  "
                  f"rho_fwd={stats['rho_fwd_post_homeo']:+.3f}  "
                  f"rho_rev={stats['rho_rev_post_homeo']:+.3f}  "
                  f"EC L-LTP w_mean={float(stc_hook.w.mean()):.4f}")

            homeo_results[alpha_val] = stats
            total_sim_ms += SIM_MS

        # Backwards-compatible alias for single-alpha downstream code
        homeo_stats        = homeo_results[args.alpha_list[0]] if n_sweep == 1 else None
        homeo_post_rho_fwd = (homeo_stats["rho_fwd_post_homeo"]
                              if homeo_stats is not None else None)
        homeo_post_rho_rev = (homeo_stats["rho_rev_post_homeo"]
                              if homeo_stats is not None else None)
    else:
        homeo_stats        = None
        homeo_post_rho_fwd = None
        homeo_post_rho_rev = None

    rank = _mpi_rank()

    # ---- HDF5 export ---------------------------------------------------------
    scale_tag = f"{args.scale}pct"
    out_dir   = os.path.join(_script_dir, f"replay_output_{scale_tag}")
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    hdf5_path = (args.out_hdf5
                 if args.out_hdf5
                 else os.path.join(out_dir, f"replay_{scale_tag}.h5"))
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(hdf5_path)), exist_ok=True)

    print(f"\n>>> [rank {rank}] Entering HDF5 export...")
    save_replay_hdf5(net, total_sim_ms, cfg["label"], hdf5_path,
                     ec_module=ec_module, stc_hook=stc_hook,
                     eclv_module=eclv_module, mpfc_module=mpfc_module,
                     homeo_stats=homeo_stats, homeo_results=homeo_results,
                     dg_module=dg_module, mpfc_assoc_hook=mpfc_assoc_hook,
                     mpfc_rec_hook=mpfc_rec_hook,
                     schaffer_hook=schaffer_hook)

    if rank != 0:
        print(f">>> [rank {rank}] Done (non-root rank exiting).")
        raise SystemExit(0)

    if schaffer_hook is not None and schaffer_hook.history:
        h = schaffer_hook.history
        print("\n--- Schaffer STDP (CA3->CA1, delay-aware) ---")
        print(f"  {'event':>6s} {'potentiated':>12s} {'depressed':>10s} "
              f"{'w_mean':>9s} {'w_CV':>8s}")
        for i, r in enumerate(h):
            if i < 2 or i >= len(h) - 2:
                print(f"  {i+1:6d} {r['n_pot']:12,d} {r['n_dep']:10,d} "
                      f"{r['w_mean']:9.4f} {r['w_cv']:8.4f}")
            elif i == 2:
                print(f"  {'...':>6s}")
        print(f"  weight CV {h[0]['w_cv']:.4f} -> {h[-1]['w_cv']:.4f}  "
              f"(rising CV = synapses differentiating, i.e. paths being selected)")

    if args.n_patterns > 1:
        _pops = [("CA3 SUP", net["CA3_SUP"], net["spk_ca3_sup"]),
                 ("CA1 PYR", net["PYR"],     net["spk_pyr"])]
        if dg_module is not None:
            # The engram question (RESULTS.md §13) is decided here, not
            # downstream: DG is the first stage with a sparse code, so it is
            # the first place "selective" can be measured at all. Previously
            # this table skipped DG and the within/between Jaccard had to be
            # recomputed offline from the exported raster every time.
            _pops.append(("DG GC", dg_module.GC, dg_module.spk_gc))
        if ec_module   is not None:
            _pops.append(("EC LII", ec_module.population,   ec_module.spike_rec))
        if eclv_module is not None:
            _pops.append(("EC LV",  eclv_module.population, eclv_module.spike_rec))
        if mpfc_module is not None:
            _pops.append(("mPFC",   mpfc_module.population, mpfc_module.spike_rec))
        _disc = pattern_discrimination(net, _pops, swr_fwd, swr_rev,
                                       SIM_MS, n_epochs)
        print_pattern_discrimination(_disc, args.n_patterns)

    if mpfc_assoc_hook is not None and mpfc_assoc_hook.history:
        h = mpfc_assoc_hook.history
        print(f"\n--- Cortical association build-up (EC LV -> mPFC) ---")
        print(f"  {'event':>6s} {'co-active':>10s} {'w_mean':>8s} {'CV':>8s}")
        for i, r in enumerate(h):
            if i < 3 or i >= len(h) - 3:
                print(f"  {i+1:6d} {r['n_coactive']:10,d} {r['w_mean']:8.4f} "
                      f"{r.get('w_cv', 0.0):8.4f}")
            elif i == 3:
                print(f"  {'...':>6s}")
        cv = h[-1].get('w_cv', 0.0)
        print(f"  weights {h[0]['w_mean']:.4f} -> {h[-1]['w_mean']:.4f} over "
              f"{len(h)} replay events")
        # Weight CV is the honest engram test: an engram potentiates a SUBSET,
        # so the distribution must SPREAD. frac_associated alone is misleading --
        # it compares uniform weights against a moving threshold and can report
        # an apparent fraction even when every synapse is identical.
        if cv < 0.02:
            print(f"  [FLAG] weight CV={cv:.4f} — association is NON-SELECTIVE: every")
            print(f"         LV->mPFC synapse moved together, so no engram subset exists.")
            print(f"         Cause is upstream, not the mPFC: EC LV fires all-or-nothing")
            print(f"         per SWR, so every mPFC cell receives identical drive and")
            print(f"         lateral inhibition has no differences to amplify.")
        else:
            print(f"  [OK] weight CV={cv:.4f} — a distinct subset is potentiated (engram).")

    print_report(net, total_sim_ms, cfg["label"],
                 ec_module=ec_module, dg_module=dg_module,
                 eclv_module=eclv_module, mpfc_module=mpfc_module)

    if not args.no_figures:
        print("\n>>> Generating figures...")
        prefix = os.path.join(out_dir, f"bidir_replay_{scale_tag}")
        paths  = plot_bidirectional_replay(net, sim_ms=SIM_MS, save_prefix=prefix)
        print(f">>> Figures saved to: {out_dir}/")
        for pp in paths:
            print(f"    {os.path.basename(pp)}")
    else:
        print("\n>>> Figure generation skipped (--no-figures).")
        print(f"    To plot locally, run:")
        print(f"    python replay_plot_from_hdf5.py --in {hdf5_path} --save-prefix replay_plots/run1")

    print(f"\n>>> Total wall time: {time.perf_counter()-t_wall:.1f}s")
    print(">>> Done.")
