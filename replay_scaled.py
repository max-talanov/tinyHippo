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


def fixed_connect(pre, post, indegree, weight, delay):
    """
    NEST native fixed_indegree — C++, fully MPI-parallel.
    Pre-flight checks cumulative static_synapse count vs NEST's 134M/VP limit.
    Each post neuron receives exactly `indegree` inputs drawn from pre.
    """
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
        syn_spec={"weight": float(weight), "delay": float(delay)},
    )


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
        t0  = swr_start_ms + step * inter_step_ms
        ids = [int(i) for i in seq_groups[k]]
        gens = nest.Create("poisson_generator", len(ids), params={
            "rate": float(drive_rate), "start": float(t0),
            "stop": float(t0 + drive_dur_ms),
        })
        nest.Connect(gens, nest.NodeCollection(ids), conn_spec="one_to_one",
                     syn_spec={"weight": float(weight), "delay": float(delay)})
        all_gens.append(gens)
    return all_gens


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
    scaffold_on=True, scaffold_step_ms=8.0,
    scaffold_rate=720.0, scaffold_weight=0.55,
    # Background drive rates [Hz]
    # Watson UPDATE-2: SUP receives ~3.4x stronger DG/EC input than DEEP
    rate_ec_ca1_pyr=580.0,
    rate_dg_ca3_sup=820.0,  rate_dg_ca3_deep=220.0,
    rate_ec_ca3_sup=530.0,  rate_ec_ca3_deep=150.0,
    rate_ca3_drive_sup=400.0, rate_ca3_drive_deep=120.0,
    rate_drive_ca1_basket=820.0,
    rate_drive_ca3_int_sup=820.0, rate_drive_ca3_int_deep=820.0,
    # Theta
    theta_on=True, theta_hz=8.0, theta_mean=1100.0, theta_amp=1000.0,
    # Synaptic weights (mV; fixed across scales — synapse count scales instead)
    w_seq_fwd=1.50,  w_seq_bwd=0.30,  w_sup_local=0.90,
    w_sup_to_deep=1.30, w_deep_local=0.85, w_deep_fwd=0.70, w_deep_to_sup=0.20,
    # Parallel
    n_threads=8,
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

    safe_set_seeds()

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
    # Populations
    # -------------------------------------------------------------------------
    print("  Creating populations...")
    CA1_PYR      = nest.Create("izhikevich", N_ca1_pyr,      params=ca1_pyr_params)
    CA1_BASKET   = nest.Create("izhikevich", N_ca1_basket,   params=basket_params)
    CA1_OLM      = nest.Create("izhikevich", N_ca1_olm,      params=olm_params)
    CA3_SUP      = nest.Create("izhikevich", N_ca3_sup,      params=ca3_sup_params)
    CA3_DEEP     = nest.Create("izhikevich", N_ca3_deep,     params=ca3_deep_params)
    CA3_INT_SUP  = nest.Create("izhikevich", N_ca3_int_sup,  params=basket_params)
    CA3_INT_DEEP = nest.Create("izhikevich", N_ca3_int_deep, params=basket_params)

    # -------------------------------------------------------------------------
    # Background inputs (one_to_one Poisson — cheap, no bottleneck)
    # UPDATE-2: SUP receives ~3.4x stronger DG/EC drive than DEEP
    # -------------------------------------------------------------------------
    print("  Connecting background inputs...")
    d_fast = 1.5;  d_slow = 3.0

    def _drive(n, rate, target, weight, delay):
        gen = nest.Create("poisson_generator", n, params={"rate": float(rate)})
        nest.Connect(gen, target, conn_spec="one_to_one",
                     syn_spec={"weight": float(weight), "delay": float(delay)})

    _drive(N_ca1_pyr,      rate_ec_ca1_pyr,         CA1_PYR,      2.0, d_slow)
    _drive(N_ca1_basket,   rate_drive_ca1_basket,   CA1_BASKET,   2.0, d_fast)
    _drive(N_ca3_sup,      rate_dg_ca3_sup,         CA3_SUP,      3.0, d_fast)
    _drive(N_ca3_deep,     rate_dg_ca3_deep,        CA3_DEEP,     1.0, d_fast)
    _drive(N_ca3_sup,      rate_ec_ca3_sup,         CA3_SUP,      2.0, d_slow)
    _drive(N_ca3_deep,     rate_ec_ca3_deep,        CA3_DEEP,     1.2, d_slow)
    _drive(N_ca3_sup,      rate_ca3_drive_sup,      CA3_SUP,      2.0, d_fast)
    _drive(N_ca3_deep,     rate_ca3_drive_deep,     CA3_DEEP,     1.5, d_fast)
    _drive(N_ca3_int_sup,  rate_drive_ca3_int_sup,  CA3_INT_SUP,  2.0, d_fast)
    _drive(N_ca3_int_deep, rate_drive_ca3_int_deep, CA3_INT_DEEP, 2.0, d_fast)

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
    for swr_s, swr_e in [(swr_fwd_start, swr_fwd_stop), (swr_rev_start, swr_rev_stop)]:
        sw_sh_sup, sw_rip_sup = make_swr_event_generators(
            n=N_ca3_sup, start_ms=swr_s, stop_ms=swr_e,
            sharpwave_rate=swr_sharpwave_rate,
            ripple_rate_mean=swr_ripple_mean, ripple_rate_amp=swr_ripple_amp,
            ripple_hz=swr_ripple_hz)
        nest.Connect(sw_sh_sup,  CA3_SUP, conn_spec="one_to_one", syn_spec={"weight": 0.35, "delay": 1.0})
        nest.Connect(sw_rip_sup, CA3_SUP, conn_spec="one_to_one", syn_spec={"weight": 0.15, "delay": 1.0})

        sw_sh_deep, sw_rip_deep = make_swr_event_generators(
            n=N_ca3_deep, start_ms=swr_s, stop_ms=swr_e,
            sharpwave_rate=swr_sharpwave_rate * 0.35,
            ripple_rate_mean=swr_ripple_mean * 0.35,
            ripple_rate_amp=swr_ripple_amp   * 0.35,
            ripple_hz=swr_ripple_hz)
        nest.Connect(sw_sh_deep,  CA3_DEEP, conn_spec="one_to_one", syn_spec={"weight": 0.35, "delay": 1.0})
        nest.Connect(sw_rip_deep, CA3_DEEP, conn_spec="one_to_one", syn_spec={"weight": 0.15, "delay": 1.0})

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
        nest.Connect(sw_sh_c1,  CA1_PYR, conn_spec="one_to_one", syn_spec={"weight": 0.20, "delay": 1.0})
        nest.Connect(sw_rip_c1, CA1_PYR, conn_spec="one_to_one", syn_spec={"weight": 0.10, "delay": 1.0})

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
        delay=d_fast,
    )
    print(f"    done in {time.perf_counter()-t_seq:.1f}s")

    # ---- CA3 E<->I  (fixed_indegree, UPDATE-5) ------------------------------
    print("  Wiring CA3 E<->I (fixed_indegree)...")
    t_ei = time.perf_counter()

    fixed_connect(CA3_SUP,      CA3_INT_SUP,  K("ca3_EI_sup",   N_ca3_sup),      0.5,  d_fast)
    fixed_connect(CA3_DEEP,     CA3_INT_DEEP, K("ca3_EI_deep",  N_ca3_deep),     0.5,  d_fast)
    fixed_connect(CA3_INT_SUP,  CA3_SUP,      K("ca3_IE_sup",   N_ca3_int_sup),  -2.0, d_fast)
    fixed_connect(CA3_INT_DEEP, CA3_DEEP,     K("ca3_IE_deep",  N_ca3_int_deep), -2.0, d_fast)
    fixed_connect(CA3_INT_SUP,  CA3_DEEP,     K("ca3_IE_cross", N_ca3_int_sup),  -0.2, d_fast)
    fixed_connect(CA3_INT_DEEP, CA3_SUP,      K("ca3_IE_cross", N_ca3_int_deep), -0.2, d_fast)
    fixed_connect(CA3_INT_SUP,  CA3_INT_SUP,  K("ca3_II",       N_ca3_int_sup),  -1.5, d_fast)
    fixed_connect(CA3_INT_DEEP, CA3_INT_DEEP, K("ca3_II",       N_ca3_int_deep), -1.5, d_fast)
    print(f"    done in {time.perf_counter()-t_ei:.1f}s")

    # ---- Schaffer collaterals (fixed_indegree, UPDATE-6) --------------------
    print("  Wiring Schaffer collaterals (fixed_indegree)...")
    t_sch = time.perf_counter()

    fixed_connect(CA3_SUP,  CA1_PYR,    K("schaffer_sup_pyr",     N_ca3_sup),  1.8, d_slow)
    fixed_connect(CA3_DEEP, CA1_PYR,    K("schaffer_deep_pyr",    N_ca3_deep), 2.2, d_slow)
    fixed_connect(CA3_SUP,  CA1_BASKET, K("schaffer_sup_basket",  N_ca3_sup),  1.5, d_fast)
    fixed_connect(CA3_DEEP, CA1_BASKET, K("schaffer_deep_basket", N_ca3_deep), 2.0, d_fast)
    print(f"    done in {time.perf_counter()-t_sch:.1f}s")

    # ---- CA1 local (fixed_indegree) -----------------------------------------
    print("  Wiring CA1 local (fixed_indegree)...")
    t_ca1 = time.perf_counter()

    fixed_connect(CA1_PYR,    CA1_PYR,    K("ca1_EE", N_ca1_pyr),     0.5,  d_slow)
    fixed_connect(CA1_PYR,    CA1_BASKET, K("ca1_EI", N_ca1_pyr),     0.5,  d_fast)
    fixed_connect(CA1_BASKET, CA1_PYR,    K("ca1_IE", N_ca1_basket),  -2.0, d_fast)
    fixed_connect(CA1_OLM,    CA1_PYR,    K("ca1_OE", N_ca1_olm),    -1.5, d_slow)
    print(f"    done in {time.perf_counter()-t_ca1:.1f}s")

    # ---- Replay triggers and scaffold ---------------------------------------
    print("  Connecting replay triggers and scaffold...")
    make_replay_trigger(ca3_sup_groups[0],  swr_fwd_start,
                        trigger_dur_ms=trigger_dur_ms,
                        trigger_rate=trigger_rate, weight=trigger_weight)
    make_replay_trigger(ca3_sup_groups[-1], swr_rev_start,
                        trigger_dur_ms=trigger_dur_ms,
                        trigger_rate=trigger_rate, weight=trigger_weight)
    if scaffold_on:
        make_staggered_replay_drive(
            ca3_sup_groups, swr_fwd_start, direction="forward",
            inter_step_ms=scaffold_step_ms, drive_rate=scaffold_rate,
            weight=scaffold_weight)
        make_staggered_replay_drive(
            ca3_sup_groups, swr_rev_start, direction="reverse",
            inter_step_ms=scaffold_step_ms, drive_rate=scaffold_rate,
            weight=scaffold_weight)

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
        ca3_seq_groups=ca3_sup_groups, ca3_sup_groups=ca3_sup_groups,
        ca3_deep_groups=ca3_deep_groups, n_seq_groups=n_seq_groups,
        swr_on=True,
        swr_fwd=(swr_fwd_start, swr_fwd_stop),
        swr_rev=(swr_rev_start, swr_rev_stop),
        swr_events=[(swr_fwd_start, swr_fwd_stop), (swr_rev_start, swr_rev_stop)],
        swr_ripple_hz=swr_ripple_hz, theta_on=theta_on, theta_hz=theta_hz,
    )


# ============================================================================
# EC LII/III module  (Phase 1 cortical addition)
# ============================================================================

@dataclass
class ECModule:
    """
    Entorhinal cortex layer II/III — minimal cortical consolidation target.

    Kept as a self-contained dataclass so it can be built optionally (via
    --ec-lii) without touching any existing hippocampal code.

    CA1→EC synapses use static_synapse (fast parallel connect on all NEST
    builds).  STDP weight updates are applied by the Python STC hook between
    SWR events via nest.GetStatus / nest.SetStatus — this is both faster and
    gives full control over the tag/PRP logic.

    Attributes
    ----------
    population    NEST NodeCollection   — EC LII/III stellate cells
    spike_rec     NEST NodeCollection   — spike recorder
    conns_ca1_ec  NEST SynapseCollection — CA1→EC static synapses (STC hook)
    N             neuron count
    K_ca1_ec      mean in-degree of the CA1→EC projection
    w_init        initial synaptic weight (all synapses start here)
    """
    population   : object   # nest.NodeCollection
    spike_rec    : object   # nest.NodeCollection  (spike_recorder)
    conns_ca1_ec : object   # nest.SynapseCollection
    N            : int
    K_ca1_ec     : int
    w_init       : float


def build_ec_lii(
    ca1_pyr,
    N_ec_lii     : int,
    K_ca1_ec     : int   = 50,
    w_ca1_ec     : float = 1.0,
    delay_ca1_ec : float = 3.0,   # axonal conduction delay CA1→EC [ms]
    rate_bg      : float = 200.0, # tonic background Poisson drive [Hz]
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
                         params=dict(a=0.02, b=0.2, c=-65.0, d=6.0,
                                     V_m=-65.0, U_m=-13.0, I_e=0.0))

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
                  "weight": float(w_ca1_ec),
                  "delay":  float(delay_ca1_ec)},
    )
    dt_conn = time.perf_counter() - t_conn

    # Store synapse handle immediately — cheap right after Connect().
    conns_ca1_ec  = nest.GetConnections(ca1_pyr, EC_LII)
    n_actual      = len(conns_ca1_ec)
    rate          = n_actual / max(dt_conn, 1e-6)

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
        population   = EC_LII,
        spike_rec    = spk_ec,
        conns_ca1_ec = conns_ca1_ec,
        N            = N_ec_lii,
        K_ca1_ec     = K,
        w_init       = w_ca1_ec,
    )


# ============================================================================
# Replay quality metric
# ============================================================================

def replay_score(spk_times, spk_senders, seq_groups, window_start, window_stop):
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

def print_report(net, sim_ms, scale_label, ec_module=None):
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
        rho, pval = replay_score(t_sup, s_sup, net["ca3_seq_groups"],
                                 win[0]-5, win[1]+30)
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

    if ec_module is not None:
        print("\n--- EC LII/III (cortical target) ---")
        t_ec, _ = _get_spikes(ec_module.spike_rec)
        rate_ec  = len(t_ec) / (ec_module.N * sim_ms / 1000.0)
        print(f"  {'EC LII/III':20s}: N={ec_module.N:8,} | {len(t_ec):10,} spikes | {rate_ec:6.2f} Hz")
        for label, (ws, we) in [("SWR-1 fwd", net["swr_fwd"]), ("SWR-2 rev", net["swr_rev"])]:
            n_ec = np.sum((t_ec >= ws) & (t_ec <= we))
            print(f"  {label}: EC spikes in window = {n_ec:,}")
        # Quick CA1->EC weight snapshot
        w_vals = np.array(nest.GetStatus(ec_module.conns_ca1_ec, "weight"))
        print(f"  CA1->EC weights: mean={w_vals.mean():.4f}  "
              f"min={w_vals.min():.4f}  max={w_vals.max():.4f}  "
              f"(init={ec_module.w_init:.4f})")

    print(f"{'='*72}")


# ============================================================================
# HDF5 export  (offline plotting on any machine — no NEST required)
# ============================================================================

def save_replay_hdf5(net, sim_ms, scale_label, outpath, bin_ms=10.0,
                     ec_module=None):
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
            # Initial CA1->EC weight snapshot (post-simulation, pre-STC hook)
            w_vals = np.array(nest.GetStatus(ec_module.conns_ca1_ec, "weight"),
                              dtype=np.float32)
            g_ec.create_dataset("w_ca1_ec_final", data=w_vals, **compress)
            g_ec.attrs["w_ca1_ec_mean"]  = float(w_vals.mean())
            g_ec.attrs["w_ca1_ec_std"]   = float(w_vals.std())

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
            rho, pval = replay_score(t_sup, s_sup, sup_groups, win[0] - 5, win[1] + 30)
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
        help="Target in-degree K for the CA1→EC projection (default: 50). "
             "static_synapse is fast at any K on MN5; K=50 is a safe start.")
    args = parser.parse_args()

    cfg = build_scale_config(args.scale)

    n_threads = (args.threads
                 if args.threads is not None
                 else int(os.environ.get("OMP_NUM_THREADS", cfg["n_threads_default"])))

    SIM_MS   = args.sim_ms
    total_N  = sum(cfg[k] for k in _REF_100PCT)
    n_groups = cfg["n_seq_groups"]

    # EC LII neuron count — independently scalable, defaults to hippocampal scale
    ec_lii_pct = args.ec_lii_scale if args.ec_lii_scale is not None else args.scale
    N_ec_lii   = _round_to_multiple(
        _REF_CORTEX["N_ec_lii"] * ec_lii_pct / 100.0,
        n_groups,
    )

    print(f"\n{'='*72}")
    print(f"  Watson et al. 2025 — Bidirectional Replay  [v4: EC LII module]")
    print(f"  Scale    : {cfg['label']}")
    print(f"  CA3_SUP  : {cfg['N_ca3_sup']:>10,}  CA3_DEEP : {cfg['N_ca3_deep']:>8,}")
    print(f"  CA1_PYR  : {cfg['N_ca1_pyr']:>10,}  groups   : {n_groups:>8,}  "
          f"(CA3_SUP/group = {cfg['N_ca3_sup']//n_groups})")
    print(f"  Total N  : {total_N:>10,}")
    if args.ec_lii:
        print(f"  EC LII   : {N_ec_lii:>10,}  ({ec_lii_pct}% of 100k ref)  "
              f"K={args.ec_lii_k}  [--ec-lii]")
    print(f"  Threads  : {n_threads}  |  Sim: {SIM_MS} ms")
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
    )

    # ---- Optional Phase 1: EC LII/III ----------------------------------------
    ec_module = None
    if args.ec_lii:
        print(">>> Building EC LII/III module...")
        ec_module = build_ec_lii(
            ca1_pyr  = net["PYR"],
            N_ec_lii = N_ec_lii,
            K_ca1_ec = args.ec_lii_k,
        )

    print(f"\n>>> Simulating {SIM_MS} ms...")
    t_sim = time.perf_counter()
    nest.Simulate(SIM_MS)
    print(f"    Simulation done in {time.perf_counter()-t_sim:.1f}s")

    rank = _mpi_rank()

    # ---- HDF5 export (gather+write on rank 0) --------------------------------
    scale_tag = f"{args.scale}pct"
    out_dir   = os.path.join(_script_dir, f"replay_output_{scale_tag}")
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    hdf5_path = (args.out_hdf5
                 if args.out_hdf5
                 else os.path.join(out_dir, f"replay_{scale_tag}.h5"))
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(hdf5_path)), exist_ok=True)

    print(f"\n>>> [rank {rank}] Entering HDF5 export (gather + write on rank 0)...")
    save_replay_hdf5(net, SIM_MS, cfg["label"], hdf5_path, ec_module=ec_module)

    if rank != 0:
        print(f">>> [rank {rank}] Done (non-root rank exiting).")
        raise SystemExit(0)

    print_report(net, SIM_MS, cfg["label"], ec_module=ec_module)

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
