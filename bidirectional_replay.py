#!/usr/bin/env python3
"""
bidirectional_replay_watson2025.py
====================================
Bidirectional sequence replay over SWR events.

ARCHITECTURE UPDATES (Watson, Vargas-Barroso & Jonas, Cell Reports 2025)
=========================================================================
This file extends the original bidirectional_replay.py with a biologically
accurate CA3 microcircuit based on the findings of Watson et al. 2025
("Cell-specific wiring routes information flow through hippocampal CA3").

Summary of changes vs original bidirectional_replay.py:
  UPDATE-1  CA3_PYR split into CA3_SUP + CA3_DEEP with distinct Izhikevich
            parameters reflecting subclass-specific intrinsic properties.
  UPDATE-2  DG (mossy-fiber) and EC drive differentiated: strong to SUP,
            ~3-4x weaker to DEEP (matching sEPSC ratio 3.2 vs 0.94 Hz).
  UPDATE-3  Four-way asymmetric CA3 recurrent connectivity replacing the
            uniform pool: S->S, S->D, D->D each ~3%, D->S ~0.18%.
  UPDATE-4  sequence_connect_ca3_layered() replaces sequence_connect_ca3().
            Sequence chain lives in CA3_SUP; CA3_DEEP receives unidirectional
            SUP output (no feedback to SUP), forming the tetrasynaptic loop.
  UPDATE-5  CA3_INT split into CA3_INT_SUP + CA3_INT_DEEP. Each pool wires
            preferentially within its own PN sublayer (coincident-input data,
            Fig 5B Watson et al.); cross-class inhibition ~10x weaker.
  UPDATE-6  Schaffer collaterals split: separate CA3_SUP->CA1 and
            CA3_DEEP->CA1 projections. CA3_DEEP is the primary output stage
            (tetrasynaptic: EC->GC->CA3_SUP->CA3_DEEP->CA1).
  UPDATE-7  Reporting, connectivity stats, and visualisation panels updated
            to reflect the two CA3 sublayers.

Usage:
    python bidirectional_replay_watson2025.py

Requirements:
    NEST >= 3.x, numpy, matplotlib, scipy (optional for replay score)
    tiny.py must be in the same directory.
"""

import sys
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import nest
from tiny import (
    safe_set_seeds,
    maybe_make_theta_generators,
    make_swr_event_generators,
)


# ============================================================================
# NEST-3.x compatibility layer  (unchanged from original)
# ============================================================================

def _to_nc(x):
    """Coerce int / list-of-ints / NodeCollection -> NodeCollection."""
    if isinstance(x, nest.NodeCollection):
        return x
    if isinstance(x, (int, np.integer)):
        return nest.NodeCollection([int(x)])
    return nest.NodeCollection([int(i) for i in x])


def conn_stats(label: str, pre, post):
    """Population connectivity summary; accepts NodeCollection or plain lists."""
    pre_nc  = _to_nc(pre)
    post_nc = _to_nc(post)
    conns   = nest.GetConnections(pre_nc, post_nc)
    n_pre, n_post = len(pre_nc), len(post_nc)
    try:
        n_conn = conns.get("source").size
    except Exception:
        n_conn = len(conns)
    density = n_conn / (n_pre * n_post) if n_pre * n_post > 0 else 0.0
    print(
        f"{label:22s}: {n_conn:7d} conns | density={density:.4f} | "
        f"out={n_conn/max(n_pre,1):.2f} | in={n_conn/max(n_post,1):.2f}"
    )


def bernoulli_connect(pre, post, p: float, weight: float, delay: float, rng):
    """
    NEST-3.x-safe Bernoulli connectivity.
    Converts pre/post to plain int lists before building NodeCollections.
    """
    if isinstance(pre, nest.NodeCollection):
        pre_list = list(pre.tolist())
    else:
        pre_list = [int(i) for i in pre]

    if isinstance(post, nest.NodeCollection):
        post_arr = np.array(post.tolist(), dtype=int)
    else:
        post_arr = np.array([int(i) for i in post], dtype=int)

    for src in pre_list:
        mask    = rng.random(post_arr.size) < p
        targets = post_arr[mask].tolist()
        if targets:
            nest.Connect(
                nest.NodeCollection([src]),
                nest.NodeCollection(targets),
                syn_spec={"weight": float(weight), "delay": float(delay)},
            )


def mean_rate(pop, spk, sim_ms: float) -> float:
    ev = nest.GetStatus(spk, "events")[0]
    return len(ev["senders"]) / (len(pop) * (sim_ms / 1000.0))


# ============================================================================
# UPDATE-4  sequence_connect_ca3_layered()
# ----------------------------------------------------------------------------
# Replaces the original sequence_connect_ca3() which wired a single
# homogeneous pool with symmetric forward/backward chains.
#
# NEW DESIGN (Watson et al. 2025, Fig 2D + Discussion):
#   - The temporal sequence (engram chain) is encoded in CA3_SUP only.
#     Superficial PNs form the classical autoassociative recurrent network.
#   - CA3_DEEP receives the output of completed SUP ensembles via the
#     asymmetric S->D projection (no feedback D->S at local scale).
#   - The deep layer can form its own local recurrent sub-network (D->D),
#     enabling "associations of associations" (higher-order processing).
#   - Reverse replay travels backward along the SUP chain only; deep cells
#     are not involved in the sequence-generation mechanism per se, but
#     fire in response to whichever SUP group drives them.
# ============================================================================

def sequence_connect_ca3_layered(
    ca3_sup,                    # UPDATE-4: takes separate SUP population
    ca3_deep,                   # UPDATE-4: takes separate DEEP population
    n_groups:    int   = 10,
    # -- Superficial recurrent sequence weights (Watson Fig 2D: S->S ~3.64%)
    p_sup_fwd:   float = 0.28,   # forward  within-SUP chain
    p_sup_bwd:   float = 0.06,   # backward within-SUP chain (for reverse replay)
    p_sup_local: float = 0.12,   # local within-group SUP recurrence
    w_sup_fwd:   float = 1.50,
    w_sup_bwd:   float = 0.30,
    w_sup_local: float = 0.90,
    # -- SUP -> DEEP feedforward (Watson Fig 2D: S->D ~3.03%)
    # UPDATE-3/4: S->D is a one-way, non-sequential connection from each
    # SUP group to the corresponding DEEP group. No D->S return at local level.
    p_sup_to_deep: float = 0.28,   # probability, mirroring S->S
    w_sup_to_deep: float = 1.30,   # slightly weaker than S->S
    # -- Deep local recurrence (Watson Fig 2D: D->D ~2.25%)
    p_deep_local: float = 0.10,   # within-group DEEP
    p_deep_fwd:   float = 0.08,   # DEEP to next DEEP group (secondary chain)
    w_deep_local: float = 0.85,
    w_deep_fwd:   float = 0.70,
    # -- Deep -> Superficial (Watson Fig 2D: D->S ~0.18% -- near zero)
    # UPDATE-3: This is the critical asymmetry. The original model had
    # p_bwd applying uniformly; here D->S is explicitly set to ~0.002
    # to match the ~18x lower connectivity measured in the paper.
    p_deep_to_sup: float = 0.002,  # Watson et al. 2025: order-of-magnitude lower
    w_deep_to_sup: float = 0.20,   # weak; minimal functional influence
    delay: float = 1.5,
    rng=None,
) -> tuple:
    """
    Wire CA3 SUP and DEEP populations with sublayer-specific connectivity.

    Returns (sup_groups, deep_groups): each a list[list[int]] of neuron IDs
    partitioned into n_groups sequence groups.

    Key asymmetry implemented (Watson et al. 2025, Fig 2D):
      S->S : ~3.64%   (sequence chain + local recurrence)
      S->D : ~3.03%   (unidirectional feedforward to deep layer)
      D->D : ~2.25%   (local deep recurrence, no strong sequence)
      D->S : ~0.18%   (near-absent; critical circuit constraint)
    """
    if rng is None:
        rng = np.random.default_rng(99)

    # Partition SUP population into sequence groups
    sup_ids  = list(ca3_sup.tolist())
    deep_ids = list(ca3_deep.tolist())
    n_sup    = len(sup_ids)
    n_deep   = len(deep_ids)

    assert n_sup  % n_groups == 0, f"N_ca3_sup ({n_sup}) must be divisible by n_groups"
    assert n_deep % n_groups == 0, f"N_ca3_deep ({n_deep}) must be divisible by n_groups"

    gs_sup  = n_sup  // n_groups
    gs_deep = n_deep // n_groups
    sup_groups  = [sup_ids [k * gs_sup  : (k+1) * gs_sup ] for k in range(n_groups)]
    deep_groups = [deep_ids[k * gs_deep : (k+1) * gs_deep] for k in range(n_groups)]

    for k in range(n_groups):
        grp_s = sup_groups[k]
        grp_d = deep_groups[k]

        # -- SUP local recurrence (Watson: S->S ~3.64%)
        bernoulli_connect(grp_s, grp_s, p_sup_local, w_sup_local, delay, rng)

        # -- SUP sequence chain (forward and backward for bidirectional replay)
        if k + 1 < n_groups:
            bernoulli_connect(grp_s, sup_groups[k+1], p_sup_fwd, w_sup_fwd, delay, rng)
        if k - 1 >= 0:
            bernoulli_connect(grp_s, sup_groups[k-1], p_sup_bwd, w_sup_bwd, delay, rng)

        # -- SUP -> DEEP unidirectional (Watson: S->D ~3.03%, NO D->S return)
        # UPDATE-4: Each SUP group projects to its corresponding DEEP group.
        # This is the feedforward arm of the tetrasynaptic loop. Importantly
        # there is NO return projection at this connectivity level, implementing
        # the near-absent D->S link (0.18%) measured by Watson et al.
        bernoulli_connect(grp_s, grp_d, p_sup_to_deep, w_sup_to_deep, delay, rng)

        # -- DEEP local recurrence (Watson: D->D ~2.25%)
        # UPDATE-4: DEEP forms its own secondary recurrent loop; no sequence
        # ordering is enforced here -- deep cells fire in response to which
        # SUP group activates them, so temporal order is inherited.
        bernoulli_connect(grp_d, grp_d, p_deep_local, w_deep_local, delay, rng)
        if k + 1 < n_groups:
            bernoulli_connect(grp_d, deep_groups[k+1], p_deep_fwd, w_deep_fwd, delay, rng)

        # -- DEEP -> SUP (Watson: D->S ~0.18% -- near-absent, crucial constraint)
        # UPDATE-3: This is explicitly set to p=0.002, reflecting the ~18x
        # reduction observed vs S->S. In the original model this pathway was
        # implicitly equal to p_bwd, a major biological inaccuracy.
        bernoulli_connect(grp_d, grp_s, p_deep_to_sup, w_deep_to_sup, delay, rng)

    return sup_groups, deep_groups


# ============================================================================
# Replay trigger  (unchanged API; now called per sublayer)
# ============================================================================

def make_replay_trigger(
    group_ids,
    trigger_start_ms: float,
    trigger_dur_ms:   float = 16.0,
    trigger_rate:     float = 2600.0,
    weight:           float = 0.95,
    delay:            float = 1.0,
):
    """
    Brief high-rate Poisson seed targeting one sequence group.
    Forward replay -> group 0 of CA3_SUP  (low index)
    Reverse replay -> group N-1 of CA3_SUP (high index)
    NOTE: Triggers target CA3_SUP only (replay is generated there);
    CA3_DEEP fires in response via the S->D projections.
    """
    ids = [int(i) for i in group_ids]
    n   = len(ids)
    gens = nest.Create(
        "poisson_generator", n,
        params={
            "rate":  float(trigger_rate),
            "start": float(trigger_start_ms),
            "stop":  float(trigger_start_ms + trigger_dur_ms),
        },
    )
    nest.Connect(
        gens,
        nest.NodeCollection(ids),
        conn_spec="one_to_one",
        syn_spec={"weight": float(weight), "delay": float(delay)},
    )
    return gens


# ============================================================================
# Staggered scaffold  (unchanged, applied to CA3_SUP groups only)
# ============================================================================

def make_staggered_replay_drive(
    seq_groups:      list,
    swr_start_ms:    float,
    direction:       str   = "forward",
    inter_step_ms:   float = 8.0,
    drive_dur_ms:    float = 10.0,
    drive_rate:      float = 750.0,
    weight:          float = 0.55,
    delay:           float = 1.0,
):
    """
    Per-group time-staggered Poisson bursts scaffolding replay propagation.
    Applied to CA3_SUP groups only (CA3_DEEP follows via S->D wiring).
    """
    n     = len(seq_groups)
    order = list(range(n)) if direction == "forward" else list(range(n-1, -1, -1))
    all_gens = []
    for step, k in enumerate(order):
        t_start = swr_start_ms + step * inter_step_ms
        ids     = [int(i) for i in seq_groups[k]]
        gens = nest.Create(
            "poisson_generator", len(ids),
            params={
                "rate":  float(drive_rate),
                "start": float(t_start),
                "stop":  float(t_start + drive_dur_ms),
            },
        )
        nest.Connect(
            gens,
            nest.NodeCollection(ids),
            conn_spec="one_to_one",
            syn_spec={"weight": float(weight), "delay": float(delay)},
        )
        all_gens.append(gens)
    return all_gens


# ============================================================================
# Network builder  -- main function with all Watson et al. 2025 updates
# ============================================================================

def build_replay_network(
    # CA1 sizes (unchanged)
    N_ca1_pyr:    int = 200,
    N_ca1_basket: int = 15,
    N_ca1_olm:    int = 12,
    # -- UPDATE-1: CA3 split into SUP + DEEP sublayers
    # Watson et al. estimate deep PNs are ~10-20% of the total population.
    # Original model had N_ca3_pyr=300 as a single pool.
    # New default: 240 SUP + 60 DEEP = 300 total (20% deep fraction).
    N_ca3_sup:    int = 240,   # UPDATE-1: superficial subclass (~80%)
    N_ca3_deep:   int = 60,    # UPDATE-1: deep subclass (~20%, Watson ~10-20%)
    # -- UPDATE-5: Two CA3 interneuron pools (one per sublayer)
    N_ca3_int_sup:  int = 30,  # UPDATE-5: interneurons targeting SUP PNs
    N_ca3_int_deep: int = 10,  # UPDATE-5: interneurons targeting DEEP PNs
    # Sequence structure
    n_seq_groups: int   = 10,
    p_seq_fwd:    float = 0.28,
    p_seq_bwd:    float = 0.06,
    w_seq_fwd:    float = 1.50,
    w_seq_bwd:    float = 0.30,
    # SUP->DEEP feedforward weight (UPDATE-4)
    p_sup_to_deep: float = 0.28,
    w_sup_to_deep: float = 1.30,
    # SWR windows [ms]
    swr_fwd_start: float = 300.0,
    swr_fwd_stop:  float = 420.0,
    swr_rev_start: float = 600.0,
    swr_rev_stop:  float = 720.0,
    # SWR generator params
    swr_sharpwave_rate: float = 280.0,
    swr_ripple_hz:      float = 180.0,
    swr_ripple_mean:    float = 1100.0,
    swr_ripple_amp:     float = 850.0,
    # Replay trigger
    trigger_dur_ms: float = 16.0,
    trigger_rate:   float = 2600.0,
    trigger_weight: float = 0.95,
    # Staggered scaffold
    scaffold_on:       bool  = True,
    scaffold_step_ms:  float = 8.0,
    scaffold_rate:     float = 720.0,
    scaffold_weight:   float = 0.55,
    # -- UPDATE-2: Background drive rates split by sublayer
    # SUP PNs receive strong DG/EC input; DEEP receive ~3-4x less
    # (Watson et al. sEPSC ratio: superficial 3.2 Hz vs deep 0.94 Hz)
    rate_ec_ca1_pyr:         float = 580.0,
    rate_dg_ca3_sup:         float = 820.0,   # UPDATE-2: high DG drive to SUP
    rate_dg_ca3_deep:        float = 220.0,   # UPDATE-2: ~3.7x lower (0.94/3.2)
    rate_ec_ca3_sup:         float = 530.0,   # UPDATE-2: EC drives SUP
    rate_ec_ca3_deep:        float = 150.0,   # UPDATE-2: minimal EC to DEEP
    rate_ca3_drive_sup:      float = 400.0,
    rate_ca3_drive_deep:     float = 120.0,   # UPDATE-2: reduced background to DEEP
    rate_drive_ca1_basket:   float = 820.0,
    rate_drive_ca3_int_sup:  float = 820.0,   # UPDATE-5: drive to SUP interneurons
    rate_drive_ca3_int_deep: float = 820.0,   # UPDATE-5: drive to DEEP interneurons
    # Theta drive
    theta_on:   bool  = True,
    theta_hz:   float = 8.0,
    theta_mean: float = 1100.0,
    theta_amp:  float = 1000.0,
    # CA1 local connectivity (unchanged)
    p_ca1_EE: float = 0.02,
    p_ca1_EI: float = 0.10,
    p_ca1_IE: float = 0.15,
    p_ca1_OE: float = 0.10,
    # -- UPDATE-5: CA3 E->I and I->E now per-sublayer
    p_ca3_EI_sup:  float = 0.12,   # SUP PYR -> SUP INT
    p_ca3_EI_deep: float = 0.12,   # DEEP PYR -> DEEP INT
    p_ca3_IE_sup:  float = 0.20,   # SUP INT -> SUP PYR
    p_ca3_IE_deep: float = 0.20,   # DEEP INT -> DEEP PYR
    # Cross-class inhibition (Watson et al.: minimal cross-class synchrony)
    p_ca3_IE_cross: float = 0.02,  # UPDATE-5: INT_SUP->DEEP or INT_DEEP->SUP
    p_ca3_II:       float = 0.10,
    # -- UPDATE-6: Schaffer collaterals per sublayer
    # Deep CA3 is the primary output stage (tetrasynaptic loop terminal)
    p_schaffer_sup_pyr:    float = 0.08,   # UPDATE-6: SUP->CA1 PYR
    p_schaffer_deep_pyr:   float = 0.12,   # UPDATE-6: DEEP->CA1 PYR (primary output)
    p_schaffer_sup_basket: float = 0.10,   # UPDATE-6: SUP->CA1 basket
    p_schaffer_deep_basket:float = 0.14,   # UPDATE-6: DEEP->CA1 basket (stronger)
    seed_connect: int = 42,
    n_threads:    int = 4,
):
    """
    Build full CA1+CA3 network with bidirectional replay.
    Updated for Watson, Vargas-Barroso & Jonas, Cell Reports 2025.
    """

    nest.ResetKernel()
    nest.SetKernelStatus({
        "resolution":        0.1,
        "local_num_threads": n_threads,
        "print_time":        True,
        "overwrite_files":   True,
    })
    safe_set_seeds()

    try:
        available = list(nest.node_models)
    except AttributeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            available = list(nest.Models("nodes"))
    if "izhikevich" not in available:
        raise RuntimeError("NEST model 'izhikevich' not found in this build.")

    # -------------------------------------------------------------------------
    # UPDATE-1  Izhikevich parameter sets -- subclass-specific
    # -------------------------------------------------------------------------
    # Original model used a single pyr_params for all CA3 cells.
    # Watson et al. report:
    #   - Superficial PNs: regular/adapting firing, Rn ~240 MOhm, rheobase ~128 pA
    #   - Deep PNs:        burst-firing, Rn ~261 MOhm, rheobase ~77 pA (p<0.0001)
    #   - Deep PNs have lower resting Vm (-64.7 vs -63.2 mV) and higher
    #     spike threshold (-41.0 vs -42.6 mV) -- Watson et al. Fig 4H.
    #
    # Izhikevich mapping:
    #   Regular spiking:    a=0.02, b=0.2, c=-65, d=8
    #   Intrinsic bursting: a=0.02, b=0.2, c=-55, d=4  (higher c = faster reset
    #                        enabling burst; lower d = less adaptation -> burst)
    # The lower rheobase of DEEP PNs is captured by a small positive I_e offset.

    # UPDATE-1a: Superficial CA3 PNs -- regular adapting (unchanged from original)
    ca3_sup_params = dict(a=0.02, b=0.2,  c=-65.0, d=8.0,
                          V_m=-63.2, U_m=-13.0, I_e=0.0)

    # UPDATE-1b: Deep CA3 PNs -- intrinsic bursting, lower rheobase
    # c=-55 gives a faster reset voltage enabling burst after-depolarisation.
    # d=4   reduces spike-frequency adaptation, sustaining burst output.
    # I_e=3.0 encodes the lower rheobase (~77 pA vs ~128 pA superficial).
    # V_m=-64.7 matches the slightly more hyperpolarised resting potential.
    ca3_deep_params = dict(a=0.02, b=0.2,  c=-55.0, d=4.0,
                           V_m=-64.7, U_m=-13.0, I_e=3.0)   # UPDATE-1b

    # CA1 and interneuron params (unchanged)
    ca1_pyr_params    = dict(a=0.02, b=0.2,  c=-65.0, d=8.0, V_m=-65.0, U_m=-13.0, I_e=0.0)
    basket_params     = dict(a=0.10, b=0.2,  c=-65.0, d=2.0, V_m=-65.0, U_m=-13.0, I_e=0.0)
    olm_params        = dict(a=0.02, b=0.25, c=-65.0, d=2.0, V_m=-65.0, U_m=-16.25, I_e=0.0)

    # -------------------------------------------------------------------------
    # Populations
    # -------------------------------------------------------------------------
    # CA1 (unchanged)
    CA1_PYR    = nest.Create("izhikevich", N_ca1_pyr,    params=ca1_pyr_params)
    CA1_BASKET = nest.Create("izhikevich", N_ca1_basket, params=basket_params)
    CA1_OLM    = nest.Create("izhikevich", N_ca1_olm,    params=olm_params)

    # UPDATE-1: CA3 split into two populations with distinct parameters
    CA3_SUP    = nest.Create("izhikevich", N_ca3_sup,       params=ca3_sup_params)   # UPDATE-1a
    CA3_DEEP   = nest.Create("izhikevich", N_ca3_deep,      params=ca3_deep_params)  # UPDATE-1b

    # UPDATE-5: Two CA3 interneuron pools -- one per PN sublayer
    CA3_INT_SUP  = nest.Create("izhikevich", N_ca3_int_sup,  params=basket_params)   # UPDATE-5a
    CA3_INT_DEEP = nest.Create("izhikevich", N_ca3_int_deep, params=basket_params)   # UPDATE-5b

    # -------------------------------------------------------------------------
    # UPDATE-2  Background / external inputs -- differentiated by sublayer
    # -------------------------------------------------------------------------
    # Original model connected DG_CA3 and EC_CA3 uniformly to all CA3_PYR.
    # Watson et al. (Fig 5A): sEPSC frequency ~3.2 Hz on superficial vs
    # 0.94 Hz on deep PNs, a ~3.4x difference. GABAzine blocked similar
    # absolute rates from both, confirming the difference is in excitatory
    # (not inhibitory) drive -- consistent with fewer mossy-fiber contacts
    # on deep PNs (sparse thorny excrescences).

    ECIII_CA1    = nest.Create("poisson_generator", N_ca1_pyr,
                               params={"rate": rate_ec_ca1_pyr})
    DRIVE_CA1B   = nest.Create("poisson_generator", N_ca1_basket,
                               params={"rate": rate_drive_ca1_basket})

    # UPDATE-2a: DG (mossy-fiber) input -- high to SUP, reduced to DEEP
    DG_CA3_SUP   = nest.Create("poisson_generator", N_ca3_sup,
                               params={"rate": rate_dg_ca3_sup})   # UPDATE-2a: strong
    DG_CA3_DEEP  = nest.Create("poisson_generator", N_ca3_deep,
                               params={"rate": rate_dg_ca3_deep})  # UPDATE-2a: ~3.7x weaker

    # UPDATE-2b: EC input -- present to both but reduced for DEEP
    EC_CA3_SUP   = nest.Create("poisson_generator", N_ca3_sup,
                               params={"rate": rate_ec_ca3_sup})   # UPDATE-2b
    EC_CA3_DEEP  = nest.Create("poisson_generator", N_ca3_deep,
                               params={"rate": rate_ec_ca3_deep})  # UPDATE-2b: weaker

    # UPDATE-2c: Background recurrent drive (split by sublayer)
    DRIV_CA3_SUP  = nest.Create("poisson_generator", N_ca3_sup,
                                params={"rate": rate_ca3_drive_sup})
    DRIV_CA3_DEEP = nest.Create("poisson_generator", N_ca3_deep,
                                params={"rate": rate_ca3_drive_deep})  # UPDATE-2c: reduced

    # UPDATE-5: Interneuron drives (now two separate pools)
    DRIV_INT_SUP  = nest.Create("poisson_generator", N_ca3_int_sup,
                                params={"rate": rate_drive_ca3_int_sup})   # UPDATE-5a
    DRIV_INT_DEEP = nest.Create("poisson_generator", N_ca3_int_deep,
                                params={"rate": rate_drive_ca3_int_deep})  # UPDATE-5b

    d_fast = 1.5;  d_slow = 3.0

    # CA1 inputs (unchanged)
    nest.Connect(ECIII_CA1,  CA1_PYR,    conn_spec="one_to_one", syn_spec={"weight": 2.0, "delay": d_slow})
    nest.Connect(DRIVE_CA1B, CA1_BASKET, conn_spec="one_to_one", syn_spec={"weight": 2.0, "delay": d_fast})

    # UPDATE-2: Connect differentiated CA3 inputs (replaces single DG_CA3 + EC_CA3)
    # SUP receives full DG mossy-fiber weight (major thorny excrescences)
    nest.Connect(DG_CA3_SUP,  CA3_SUP,  conn_spec="one_to_one", syn_spec={"weight": 3.0, "delay": d_fast})
    # DEEP receives weaker DG input (sparse/absent thorny excrescences, Watson Fig 1B/D)
    nest.Connect(DG_CA3_DEEP, CA3_DEEP, conn_spec="one_to_one", syn_spec={"weight": 1.0, "delay": d_fast})  # UPDATE-2a
    # EC inputs
    nest.Connect(EC_CA3_SUP,   CA3_SUP,  conn_spec="one_to_one", syn_spec={"weight": 2.0, "delay": d_slow})
    nest.Connect(EC_CA3_DEEP,  CA3_DEEP, conn_spec="one_to_one", syn_spec={"weight": 1.2, "delay": d_slow})  # UPDATE-2b
    # Background drives
    nest.Connect(DRIV_CA3_SUP,  CA3_SUP,  conn_spec="one_to_one", syn_spec={"weight": 2.0, "delay": d_fast})
    nest.Connect(DRIV_CA3_DEEP, CA3_DEEP, conn_spec="one_to_one", syn_spec={"weight": 1.5, "delay": d_fast})  # UPDATE-2c
    # UPDATE-5: Interneuron drives
    nest.Connect(DRIV_INT_SUP,  CA3_INT_SUP,  conn_spec="one_to_one", syn_spec={"weight": 2.0, "delay": d_fast})
    nest.Connect(DRIV_INT_DEEP, CA3_INT_DEEP, conn_spec="one_to_one", syn_spec={"weight": 2.0, "delay": d_fast})

    # -------------------------------------------------------------------------
    # Theta drive (unchanged logic, extended to new populations)
    # -------------------------------------------------------------------------
    if theta_on:
        for pop, w_th in [
            (CA3_SUP,       0.80),
            (CA3_DEEP,      0.60),   # UPDATE-1: DEEP already has I_e offset; lighter theta
            (CA3_INT_SUP,   1.80),   # UPDATE-5
            (CA3_INT_DEEP,  1.80),   # UPDATE-5
            (CA1_PYR,       1.00),
            (CA1_BASKET,    2.00),
            (CA1_OLM,       2.00),
        ]:
            th = maybe_make_theta_generators(len(pop), theta_mean, theta_amp, theta_hz)
            if th is not None:
                nest.Connect(th, pop, conn_spec="one_to_one",
                             syn_spec={"weight": float(w_th), "delay": 1.0})

    # -------------------------------------------------------------------------
    # SWR event generators (two events: forward + reverse)
    # Applied uniformly to SUP and DEEP during each event window.
    # -------------------------------------------------------------------------
    for swr_s, swr_e in [(swr_fwd_start, swr_fwd_stop), (swr_rev_start, swr_rev_stop)]:

        # SUP PYR receives sharp-wave + ripple
        sw_sh_sup, sw_rip_sup = make_swr_event_generators(
            n=N_ca3_sup, start_ms=swr_s, stop_ms=swr_e,
            sharpwave_rate=swr_sharpwave_rate,
            ripple_rate_mean=swr_ripple_mean, ripple_rate_amp=swr_ripple_amp,
            ripple_hz=swr_ripple_hz)
        nest.Connect(sw_sh_sup,  CA3_SUP, conn_spec="one_to_one", syn_spec={"weight": 0.35, "delay": 1.0})
        nest.Connect(sw_rip_sup, CA3_SUP, conn_spec="one_to_one", syn_spec={"weight": 0.15, "delay": 1.0})

        # DEEP PYR receives attenuated SWR (lower spontaneous input, Watson Fig 5A)
        sw_sh_deep, sw_rip_deep = make_swr_event_generators(
            n=N_ca3_deep, start_ms=swr_s, stop_ms=swr_e,
            sharpwave_rate=swr_sharpwave_rate * 0.35,   # UPDATE-2: attenuated for DEEP
            ripple_rate_mean=swr_ripple_mean * 0.35,
            ripple_rate_amp=swr_ripple_amp * 0.35,
            ripple_hz=swr_ripple_hz)
        nest.Connect(sw_sh_deep,  CA3_DEEP, conn_spec="one_to_one", syn_spec={"weight": 0.35, "delay": 1.0})
        nest.Connect(sw_rip_deep, CA3_DEEP, conn_spec="one_to_one", syn_spec={"weight": 0.15, "delay": 1.0})

        # UPDATE-5: Each INT pool receives ripple drive (no sharp-wave)
        _, sw_rip_int_sup = make_swr_event_generators(
            n=N_ca3_int_sup, start_ms=swr_s, stop_ms=swr_e,
            sharpwave_rate=0.0,
            ripple_rate_mean=swr_ripple_mean, ripple_rate_amp=swr_ripple_amp,
            ripple_hz=swr_ripple_hz)
        nest.Connect(sw_rip_int_sup, CA3_INT_SUP, conn_spec="one_to_one",
                     syn_spec={"weight": 0.60, "delay": 1.0})

        _, sw_rip_int_deep = make_swr_event_generators(
            n=N_ca3_int_deep, start_ms=swr_s, stop_ms=swr_e,
            sharpwave_rate=0.0,
            ripple_rate_mean=swr_ripple_mean, ripple_rate_amp=swr_ripple_amp,
            ripple_hz=swr_ripple_hz)
        nest.Connect(sw_rip_int_deep, CA3_INT_DEEP, conn_spec="one_to_one",
                     syn_spec={"weight": 0.60, "delay": 1.0})

        # CA1 SWR inputs (unchanged)
        sw_sh_c1, sw_rip_c1 = make_swr_event_generators(
            n=N_ca1_pyr, start_ms=swr_s, stop_ms=swr_e,
            sharpwave_rate=swr_sharpwave_rate * 0.6,
            ripple_rate_mean=swr_ripple_mean * 0.6, ripple_rate_amp=swr_ripple_amp * 0.6,
            ripple_hz=swr_ripple_hz)
        nest.Connect(sw_sh_c1,  CA1_PYR, conn_spec="one_to_one", syn_spec={"weight": 0.20, "delay": 1.0})
        nest.Connect(sw_rip_c1, CA1_PYR, conn_spec="one_to_one", syn_spec={"weight": 0.10, "delay": 1.0})

        _, sw_rip_c1b = make_swr_event_generators(
            n=N_ca1_basket, start_ms=swr_s, stop_ms=swr_e,
            sharpwave_rate=0.0,
            ripple_rate_mean=swr_ripple_mean * 0.8, ripple_rate_amp=swr_ripple_amp * 0.8,
            ripple_hz=swr_ripple_hz)
        nest.Connect(sw_rip_c1b, CA1_BASKET, conn_spec="one_to_one",
                     syn_spec={"weight": 0.55, "delay": 1.0})

    # -------------------------------------------------------------------------
    # UPDATE-3/4  Recurrent connectivity -- four-way asymmetric (Watson Fig 2D)
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(seed_connect)

    # Replace original sequence_connect_ca3() with layered version
    # FIX: use p_sup_fwd / p_sup_bwd (matching function signature)
    ca3_sup_groups, ca3_deep_groups = sequence_connect_ca3_layered(
        CA3_SUP, CA3_DEEP,
        n_groups=n_seq_groups,
        p_sup_fwd=p_seq_fwd,         w_sup_fwd=w_seq_fwd,
        p_sup_bwd=p_seq_bwd,         w_sup_bwd=w_seq_bwd,
        p_sup_to_deep=p_sup_to_deep, w_sup_to_deep=w_sup_to_deep,
        delay=d_fast, rng=rng,
    )

    # -------------------------------------------------------------------------
    # UPDATE-5  CA3 E<->I per sublayer, with minimal cross-layer inhibition
    # -------------------------------------------------------------------------
    # Original: single CA3_PYR <-> CA3_INT pool.
    # NEW: separate E->I and I->E for each sublayer; cross-layer ~10x weaker.

    # Within-sublayer E->I
    bernoulli_connect(CA3_SUP,  CA3_INT_SUP,  p_ca3_EI_sup,  0.5,  d_fast, rng)  # SUP E->I
    bernoulli_connect(CA3_DEEP, CA3_INT_DEEP, p_ca3_EI_deep, 0.5,  d_fast, rng)  # DEEP E->I

    # Within-sublayer I->E (main inhibition, negative weight)
    bernoulli_connect(CA3_INT_SUP,  CA3_SUP,  p_ca3_IE_sup,  -2.0, d_fast, rng)  # SUP I->E
    bernoulli_connect(CA3_INT_DEEP, CA3_DEEP, p_ca3_IE_deep, -2.0, d_fast, rng)  # DEEP I->E

    # Cross-layer inhibition (Watson: minimal, ~10x weaker than within-layer)
    bernoulli_connect(CA3_INT_SUP,  CA3_DEEP, p_ca3_IE_cross, -0.2, d_fast, rng)  # UPDATE-5
    bernoulli_connect(CA3_INT_DEEP, CA3_SUP,  p_ca3_IE_cross, -0.2, d_fast, rng)  # UPDATE-5

    # I->I within sublayer
    bernoulli_connect(CA3_INT_SUP,  CA3_INT_SUP,  p_ca3_II, -1.5, d_fast, rng)
    bernoulli_connect(CA3_INT_DEEP, CA3_INT_DEEP, p_ca3_II, -1.5, d_fast, rng)

    # -------------------------------------------------------------------------
    # UPDATE-6  Schaffer collaterals -- per sublayer
    # -------------------------------------------------------------------------
    # Original: single CA3_PYR -> CA1 projection.
    # NEW: CA3_SUP->CA1 and CA3_DEEP->CA1 with separate probabilities.
    # CA3_DEEP is the primary output (tetrasynaptic loop terminal).
    bernoulli_connect(CA3_SUP,  CA1_PYR,    p_schaffer_sup_pyr,    1.8, d_slow, rng)  # UPDATE-6
    bernoulli_connect(CA3_DEEP, CA1_PYR,    p_schaffer_deep_pyr,   2.2, d_slow, rng)  # UPDATE-6: primary
    bernoulli_connect(CA3_SUP,  CA1_BASKET, p_schaffer_sup_basket, 1.5, d_fast, rng)  # UPDATE-6
    bernoulli_connect(CA3_DEEP, CA1_BASKET, p_schaffer_deep_basket,2.0, d_fast, rng)  # UPDATE-6: stronger

    # -------------------------------------------------------------------------
    # CA1 local connectivity (unchanged)
    # -------------------------------------------------------------------------
    bernoulli_connect(CA1_PYR,    CA1_PYR,    p_ca1_EE, 0.5,  d_slow, rng)
    bernoulli_connect(CA1_PYR,    CA1_BASKET, p_ca1_EI, 0.5,  d_fast, rng)
    bernoulli_connect(CA1_BASKET, CA1_PYR,    p_ca1_IE, -2.0, d_fast, rng)
    bernoulli_connect(CA1_OLM,    CA1_PYR,    p_ca1_OE, -1.5, d_slow, rng)

    # -------------------------------------------------------------------------
    # Replay triggers (target CA3_SUP groups only; DEEP follows via S->D)
    # -------------------------------------------------------------------------
    make_replay_trigger(
        ca3_sup_groups[0], swr_fwd_start,
        trigger_dur_ms=trigger_dur_ms,
        trigger_rate=trigger_rate,
        weight=trigger_weight,
    )
    make_replay_trigger(
        ca3_sup_groups[-1], swr_rev_start,
        trigger_dur_ms=trigger_dur_ms,
        trigger_rate=trigger_rate,
        weight=trigger_weight,
    )

    # -------------------------------------------------------------------------
    # Staggered scaffold (CA3_SUP groups only)
    # -------------------------------------------------------------------------
    if scaffold_on:
        make_staggered_replay_drive(
            ca3_sup_groups, swr_fwd_start, direction="forward",
            inter_step_ms=scaffold_step_ms, drive_rate=scaffold_rate, weight=scaffold_weight)
        make_staggered_replay_drive(
            ca3_sup_groups, swr_rev_start, direction="reverse",
            inter_step_ms=scaffold_step_ms, drive_rate=scaffold_rate, weight=scaffold_weight)

    # -------------------------------------------------------------------------
    # Recorders
    # -------------------------------------------------------------------------
    spk_ca1_pyr = nest.Create("spike_recorder")
    spk_ca1_ba  = nest.Create("spike_recorder")
    spk_ca1_olm = nest.Create("spike_recorder")
    nest.Connect(CA1_PYR,    spk_ca1_pyr)
    nest.Connect(CA1_BASKET, spk_ca1_ba)
    nest.Connect(CA1_OLM,    spk_ca1_olm)

    # UPDATE-7: Separate spike recorders for SUP and DEEP sublayers
    spk_ca3_sup       = nest.Create("spike_recorder")  # UPDATE-7a
    spk_ca3_deep      = nest.Create("spike_recorder")  # UPDATE-7b
    spk_ca3_int_sup   = nest.Create("spike_recorder")  # UPDATE-7c
    spk_ca3_int_deep  = nest.Create("spike_recorder")  # UPDATE-7d
    nest.Connect(CA3_SUP,      spk_ca3_sup)
    nest.Connect(CA3_DEEP,     spk_ca3_deep)
    nest.Connect(CA3_INT_SUP,  spk_ca3_int_sup)
    nest.Connect(CA3_INT_DEEP, spk_ca3_int_deep)

    try:
        vm = nest.Create("multimeter", params={"record_from": ["V_m", "U_m"], "interval": 0.2})
    except Exception:
        vm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 0.2})
    nest.Connect(vm, CA1_PYR[:5])
    nest.Connect(vm, CA3_SUP[:5])
    nest.Connect(vm, CA3_DEEP[:min(3, N_ca3_deep)])  # UPDATE-7: also record DEEP Vm

    # -------------------------------------------------------------------------
    # UPDATE-7  Connectivity stats -- extended for two-sublayer architecture
    # -------------------------------------------------------------------------
    print("\n=== Connectivity stats (Watson et al. 2025 two-layer CA3) ===")
    # Four-way CA3 recurrent (compare to Watson Fig 2D data)
    conn_stats("CA3 SUP->SUP (S-S)", CA3_SUP,  CA3_SUP)   # expect ~3.64%
    conn_stats("CA3 SUP->DEEP(S-D)", CA3_SUP,  CA3_DEEP)  # expect ~3.03%
    conn_stats("CA3 DEEP->DEEP(D-D)", CA3_DEEP, CA3_DEEP) # expect ~2.25%
    conn_stats("CA3 DEEP->SUP (D-S)", CA3_DEEP, CA3_SUP)  # expect ~0.18%
    # Sequence chain sample
    conn_stats("Seq SUP g0->g1",
               nest.NodeCollection(ca3_sup_groups[0]),
               nest.NodeCollection(ca3_sup_groups[1]))
    # Inhibitory sublayer wiring
    conn_stats("CA3 SUP->INT_SUP",   CA3_SUP,      CA3_INT_SUP)
    conn_stats("CA3 INT_SUP->SUP",   CA3_INT_SUP,  CA3_SUP)
    conn_stats("CA3 DEEP->INT_DEEP", CA3_DEEP,     CA3_INT_DEEP)
    conn_stats("CA3 INT_DEEP->DEEP", CA3_INT_DEEP, CA3_DEEP)
    conn_stats("CA3 INT_SUP->DEEP",  CA3_INT_SUP,  CA3_DEEP)  # cross-layer: should be ~10x weaker
    # Schaffer collaterals
    conn_stats("Sch SUP->CA1E",      CA3_SUP,  CA1_PYR)
    conn_stats("Sch DEEP->CA1E",     CA3_DEEP, CA1_PYR)
    conn_stats("CA1E->CA1E",         CA1_PYR,  CA1_PYR)
    conn_stats("CA1I->CA1E",         CA1_BASKET, CA1_PYR)

    return dict(
        # CA1
        PYR=CA1_PYR, BASKET=CA1_BASKET, OLM=CA1_OLM,
        spk_pyr=spk_ca1_pyr, spk_ba=spk_ca1_ba, spk_olm=spk_ca1_olm,
        # UPDATE-7: CA3 sub-populations exposed separately
        CA3_SUP=CA3_SUP,   CA3_DEEP=CA3_DEEP,
        CA3_INT_SUP=CA3_INT_SUP, CA3_INT_DEEP=CA3_INT_DEEP,
        spk_ca3_sup=spk_ca3_sup, spk_ca3_deep=spk_ca3_deep,
        spk_ca3_int_sup=spk_ca3_int_sup, spk_ca3_int_deep=spk_ca3_int_deep,
        # Legacy keys for backward compatibility with plotting helpers
        CA3_PYR=CA3_SUP,   CA3_INT=CA3_INT_SUP,
        spk_ca3_pyr=spk_ca3_sup, spk_ca3_int=spk_ca3_int_sup,
        vm=vm,
        # Sequence groups (SUP carries the replay; DEEP follows)
        ca3_seq_groups=ca3_sup_groups,   # used by replay_score / triggers
        ca3_sup_groups=ca3_sup_groups,
        ca3_deep_groups=ca3_deep_groups,
        n_seq_groups=n_seq_groups,
        # SWR windows
        swr_on=True,
        swr_fwd=(swr_fwd_start, swr_fwd_stop),
        swr_rev=(swr_rev_start, swr_rev_stop),
        swr_events=[(swr_fwd_start, swr_fwd_stop), (swr_rev_start, swr_rev_stop)],
        swr_ripple_hz=swr_ripple_hz,
        theta_on=theta_on, theta_hz=theta_hz,
    )


# ============================================================================
# Replay quality metric  (unchanged logic; operates on CA3_SUP spikes)
# ============================================================================

def replay_score(spk_times, spk_senders, seq_groups, window_start, window_stop):
    """
    Spearman rho between sequence group index and mean spike time within window.
    rho ~ +1.0  -> clean forward replay
    rho ~ -1.0  -> clean reverse replay
    Applied to CA3_SUP spikes (sequence is encoded there, Watson et al.).
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("[replay_score] scipy not available -- skipping.")
        return None, None

    mask  = (spk_times >= window_start) & (spk_times <= window_stop)
    t_win = spk_times[mask]
    s_win = spk_senders[mask]

    gidx, gmean = [], []
    for k, grp in enumerate(seq_groups):
        t_g = t_win[np.isin(s_win, np.array(grp))]
        if len(t_g) > 0:
            gidx.append(k)
            gmean.append(float(np.mean(t_g)))

    if len(gidx) < 3:
        return np.nan, np.nan

    return spearmanr(gidx, gmean)


# ============================================================================
# Visualisation helpers
# ============================================================================

def _get_spikes(spk_rec):
    ev = nest.GetStatus(spk_rec, "events")[0]
    return np.array(ev["times"], dtype=float), np.array(ev["senders"], dtype=int)


def _binned_rate(times_ms, n_cells, t_stop, bin_ms):
    edges     = np.arange(0.0, t_stop + bin_ms, bin_ms)
    counts, _ = np.histogram(times_ms, bins=edges)
    centers   = edges[:-1] + bin_ms / 2.0
    return centers, counts / (bin_ms / 1e3) / max(int(n_cells), 1)


def plot_bidirectional_replay(net, sim_ms=1000.0, save_prefix="replay"):
    """
    Six-figure visualisation suite.
    UPDATE-7: Figs 1-2 now show CA3_SUP and CA3_DEEP in separate colours.
    Fig 5 shows INT_SUP and INT_DEEP rates separately.
    """
    # UPDATE-7: Retrieve spikes for both CA3 sublayers
    t_sup,  s_sup  = _get_spikes(net["spk_ca3_sup"])
    t_deep, s_deep = _get_spikes(net["spk_ca3_deep"])
    t_ca3i_sup,  _ = _get_spikes(net["spk_ca3_int_sup"])
    t_ca3i_deep, _ = _get_spikes(net["spk_ca3_int_deep"])
    t_ca1p, s_ca1p = _get_spikes(net["spk_pyr"])
    t_ca1b, s_ca1b = _get_spikes(net["spk_ba"])

    # Combine CA3 spikes for legacy plots
    t_ca3p = np.concatenate([t_sup, t_deep])
    s_ca3p = np.concatenate([s_sup, s_deep])
    t_ca3i = np.concatenate([t_ca3i_sup, t_ca3i_deep])

    seq_groups   = net["ca3_seq_groups"]
    n_groups     = net["n_seq_groups"]
    swr_fwd      = net["swr_fwd"]
    swr_rev      = net["swr_rev"]
    cmap_seq     = plt.cm.viridis
    group_colors = [cmap_seq(k / max(n_groups - 1, 1)) for k in range(n_groups)]

    def shade(ax, alpha=0.18):
        ax.axvspan(*swr_fwd, color="steelblue", alpha=alpha, label="SWR-1 fwd")
        ax.axvspan(*swr_rev, color="tomato",    alpha=alpha, label="SWR-2 rev")

    out_paths = []

    # -- Fig 1: Overview (UPDATE-7: SUP and DEEP shown in separate panels) ----
    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    fig.suptitle("Bidirectional Replay -- Watson et al. 2025 Two-Layer CA3",
                 fontsize=13, fontweight="bold")

    # Panel A: CA3 SUP raster (sequence-coloured)
    ax = axes[0]
    for k, grp in enumerate(seq_groups):
        m = np.isin(s_sup, np.array(grp))
        ax.scatter(t_sup[m], s_sup[m], s=1.5, color=group_colors[k], rasterized=True)
    shade(ax)
    ax.set_ylabel("CA3 SUP ID", fontsize=9)
    ax.set_title("A  CA3 SUPERFICIAL raster  [colour=seq group]  |  replay generator", fontsize=9, loc="left")
    sm = ScalarMappable(cmap=cmap_seq, norm=Normalize(0, n_groups - 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.01).set_label("Group #", fontsize=8)

    # UPDATE-7: Panel B -- CA3 DEEP raster (new; shows tetrasynaptic output layer)
    ax = axes[1]
    for k, grp in enumerate(net["ca3_deep_groups"]):
        m = np.isin(s_deep, np.array(grp))
        ax.scatter(t_deep[m], s_deep[m], s=2.0, color=group_colors[k],
                   marker="^", alpha=0.7, rasterized=True)
    shade(ax)
    ax.set_ylabel("CA3 DEEP ID", fontsize=9)
    ax.set_title("B  CA3 DEEP raster  [UPDATE-1: burst-firing, tetrasynaptic output]",
                 fontsize=9, loc="left")

    # Panel C: CA1 raster (unchanged)
    ax = axes[2]
    ax.scatter(t_ca1p, s_ca1p, s=1.2, color="slategray", rasterized=True)
    shade(ax)
    ax.set_ylabel("CA1 PYR ID", fontsize=9)
    ax.set_title("C  CA1 PYR raster", fontsize=9, loc="left")

    # Panel D: Population rates
    ax = axes[3]
    tc, rc_sup  = _binned_rate(t_sup,  len(net["CA3_SUP"]),  sim_ms, 10.0)
    tc, rc_deep = _binned_rate(t_deep, len(net["CA3_DEEP"]), sim_ms, 10.0)
    tc, rc1     = _binned_rate(t_ca1p, len(net["PYR"]),      sim_ms, 10.0)
    ax.plot(tc, rc_sup,  color="darkorange", lw=1.2, label="CA3 SUP")
    ax.plot(tc, rc_deep, color="royalblue",  lw=1.2, label="CA3 DEEP")  # UPDATE-7
    ax.plot(tc, rc1,     color="steelblue",  lw=1.2, alpha=0.7, label="CA1 PYR")
    shade(ax)
    ax.legend(fontsize=7, ncol=3)
    ax.set_ylabel("Rate (Hz)", fontsize=9)
    ax.set_title("D  Population rates  [10 ms bins]  |  SUP=orange, DEEP=blue", fontsize=9, loc="left")

    # Panel E: Inhibitory rates (UPDATE-7: INT_SUP and INT_DEEP separately)
    ax = axes[4]
    tf, ri_sup  = _binned_rate(t_ca3i_sup,  len(net["CA3_INT_SUP"]),  sim_ms, 2.0)
    tf, ri_deep = _binned_rate(t_ca3i_deep, len(net["CA3_INT_DEEP"]), sim_ms, 2.0)
    tf, rb      = _binned_rate(t_ca1b,      len(net["BASKET"]),       sim_ms, 2.0)
    ax.plot(tf, ri_sup,  color="firebrick",    lw=0.8, alpha=0.85, label="CA3 INT_SUP")
    ax.plot(tf, ri_deep, color="salmon",       lw=0.8, alpha=0.85, label="CA3 INT_DEEP")  # UPDATE-7
    ax.plot(tf, rb,      color="mediumorchid", lw=0.8, alpha=0.85, label="CA1 Basket")
    shade(ax)
    ax.legend(fontsize=7, ncol=3)
    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_ylabel("Rate (Hz)", fontsize=9)
    ax.set_title("E  Inhibitory rates  [2 ms bins]  |  UPDATE-5: sublayer-specific INT pools",
                 fontsize=9, loc="left")

    fig.tight_layout()
    p = f"{save_prefix}_fig1_overview.png"
    fig.savefig(p, dpi=150); plt.close(fig); out_paths.append(p)
    print(f"  saved {p}")

    # -- Fig 2: Sequence group heatmap (CA3_SUP only -- sequence lives there) --
    bin_ms = 5.0
    edges  = np.arange(0.0, sim_ms + bin_ms, bin_ms)
    n_bins = len(edges) - 1
    gs_per = len(seq_groups[0])
    heat   = np.zeros((n_groups, n_bins))
    for k, grp in enumerate(seq_groups):
        m = np.isin(s_sup, np.array(grp))
        counts, _ = np.histogram(t_sup[m], bins=edges)
        heat[k]   = counts / (bin_ms / 1e3) / max(gs_per, 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(heat, aspect="auto", origin="lower",
                   extent=[0, sim_ms, -0.5, n_groups - 0.5],
                   cmap="inferno", interpolation="nearest")
    fig.colorbar(im, ax=ax, pad=0.02).set_label("Rate (Hz)", fontsize=9)
    ax.axvspan(*swr_fwd, color="white", alpha=0.20, label="SWR-1 fwd")
    ax.axvspan(*swr_rev, color="cyan",  alpha=0.15, label="SWR-2 rev")
    ax.plot([swr_fwd[0], swr_fwd[0] + (swr_fwd[1]-swr_fwd[0]) * 0.75],
            [0, n_groups - 1], "--w", lw=1.8, alpha=0.9, label="Expected fwd slope")
    ax.plot([swr_rev[0], swr_rev[0] + (swr_rev[1]-swr_rev[0]) * 0.75],
            [n_groups - 1, 0], "--c", lw=1.8, alpha=0.9, label="Expected rev slope")
    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_ylabel("Sequence group #", fontsize=10)
    ax.set_title("CA3 SUPERFICIAL Sequence Group Heatmap [UPDATE-4: replay in SUP layer]", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    p = f"{save_prefix}_fig2_heatmap.png"
    fig.savefig(p, dpi=150); plt.close(fig); out_paths.append(p)
    print(f"  saved {p}")

    # -- Fig 3: Zoom forward vs reverse (CA3_SUP) ----------------------------
    pad = 30.0
    fig, (axf, axr) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Zoom: Forward vs Reverse Replay  (CA3 SUPERFICIAL, group-coloured)",
                 fontsize=12, fontweight="bold")
    for ax, (s, e), title in [
        (axf, swr_fwd, "SWR-1 -> Forward\n(seed: SUP group 0, sweep up)"),
        (axr, swr_rev, "SWR-2 -> Reverse\n(seed: SUP group N-1, sweep down)"),
    ]:
        for k, grp in enumerate(seq_groups):
            m = (t_sup >= s - pad) & (t_sup <= e + pad) & np.isin(s_sup, np.array(grp))
            ax.scatter(t_sup[m], s_sup[m], s=5, color=group_colors[k], rasterized=True)
        ax.axvspan(s, e, color="gold", alpha=0.22, label="SWR window")
        ax.set_xlim(s - pad, e + pad)
        ax.set_xlabel("Time (ms)", fontsize=9)
        ax.set_ylabel("Neuron ID (SUP)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        sm2 = ScalarMappable(cmap=cmap_seq, norm=Normalize(0, n_groups - 1))
        sm2.set_array([])
        fig.colorbar(sm2, ax=ax, pad=0.02).set_label("Group #", fontsize=8)
    fig.tight_layout()
    p = f"{save_prefix}_fig3_zoom.png"
    fig.savefig(p, dpi=150); plt.close(fig); out_paths.append(p)
    print(f"  saved {p}")

    # -- Fig 4: Per-group peak times -----------------------------------------
    def peak_times_per_group(t_spk, s_spk, groups, win_s, win_e):
        gidx, peaks = [], []
        for k, grp in enumerate(groups):
            m = ((t_spk >= win_s - 5) & (t_spk <= win_e + 20)
                 & np.isin(s_spk, np.array(grp)))
            t_g = t_spk[m]
            if len(t_g) >= 2:
                bins_f    = np.arange(win_s - 5, win_e + 25, 5.0)
                cnt, _    = np.histogram(t_g, bins=bins_f)
                gidx.append(k)
                peaks.append(float(bins_f[np.argmax(cnt)] + 2.5))
        return np.array(gidx), np.array(peaks)

    fig, (ax4f, ax4r) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Per-Group Peak Activation Time (slope sign = replay direction)",
                 fontsize=11, fontweight="bold")
    for ax, (s, e), title in [
        (ax4f, swr_fwd, "SWR-1: Forward  (positive slope = correct)"),
        (ax4r, swr_rev, "SWR-2: Reverse  (negative slope = correct)"),
    ]:
        gi, pt       = peak_times_per_group(t_sup, s_sup, seq_groups, s, e)
        colors_pt    = [group_colors[k] for k in gi]
        ax.scatter(pt, gi, c=colors_pt, s=60, zorder=3, edgecolors="k", linewidths=0.4)
        if len(gi) >= 2:
            pf = np.polyfit(pt, gi, 1)
            tl = np.linspace(s - 10, e + 25, 200)
            ax.plot(tl, np.polyval(pf, tl), "--k", lw=1.2, alpha=0.7,
                    label=f"slope = {pf[0]:.3f} gr/ms")
        ax.axvspan(s, e, color="gold", alpha=0.18)
        ax.set_xlabel("Peak time (ms)", fontsize=9)
        ax.set_ylabel("Sequence group #", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(s - 20, e + 30)
        ax.set_yticks(range(n_groups))
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = f"{save_prefix}_fig4_peak_times.png"
    fig.savefig(p, dpi=150); plt.close(fig); out_paths.append(p)
    print(f"  saved {p}")

    # -- Fig 5: Ripple-band rates (UPDATE-7: INT_SUP and INT_DEEP separately) -
    fig, axes5 = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Ripple-Band Inhibitory Rates  [2 ms bins]  |  UPDATE-5: sublayer INTs",
                 fontsize=11, fontweight="bold")
    for ax, (s, e), title in [
        (axes5[0], swr_fwd, "SWR-1 (Forward)"),
        (axes5[1], swr_rev, "SWR-2 (Reverse)"),
    ]:
        pad2 = 50.0
        tb, rb2       = _binned_rate(t_ca1b,      len(net["BASKET"]),       sim_ms, 2.0)
        ti_sup, ri_s2 = _binned_rate(t_ca3i_sup,  len(net["CA3_INT_SUP"]),  sim_ms, 2.0)
        ti_dep, ri_d2 = _binned_rate(t_ca3i_deep, len(net["CA3_INT_DEEP"]), sim_ms, 2.0)
        m = (tb >= s - pad2) & (tb <= e + pad2)
        ax.plot(tb[m],     rb2[m],   color="mediumorchid", lw=1.0, label="CA1 Basket")
        ax.plot(ti_sup[m], ri_s2[m], color="firebrick",    lw=1.0, alpha=0.8, label="CA3 INT_SUP")
        ax.plot(ti_dep[m], ri_d2[m], color="salmon",       lw=1.0, alpha=0.8, label="CA3 INT_DEEP")
        ax.axvspan(s, e, color="gold", alpha=0.25, label="SWR window")
        ax.set_xlabel("Time (ms)", fontsize=9)
        ax.set_ylabel("Rate (Hz)", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = f"{save_prefix}_fig5_ripple_rates.png"
    fig.savefig(p, dpi=150); plt.close(fig); out_paths.append(p)
    print(f"  saved {p}")

    # -- Fig 6: Vm traces (UPDATE-7: SUP and DEEP overlaid, different colours) -
    ev_vm = nest.GetStatus(net["vm"], "events")[0]
    vt    = np.array(ev_vm["times"])
    vs    = np.array(ev_vm["senders"])
    vv    = np.array(ev_vm["V_m"])

    sup_ids_set  = set(net["CA3_SUP"].tolist())
    deep_ids_set = set(net["CA3_DEEP"].tolist())

    fig, ax = plt.subplots(figsize=(14, 4))
    for gid in np.unique(vs):
        m   = vs == gid
        idx = np.argsort(vt[m])
        if gid in deep_ids_set:
            # UPDATE-7: DEEP cells in blue -- burst-firing visible
            ax.plot(vt[m][idx], vv[m][idx], lw=0.7, alpha=0.9, color="royalblue",
                    label="DEEP" if gid == list(deep_ids_set)[0] else "")
        else:
            ax.plot(vt[m][idx], vv[m][idx], lw=0.5, alpha=0.6, color="darkorange",
                    label="SUP" if gid == list(sup_ids_set)[0] else "")
    ax.axvspan(*swr_fwd, color="steelblue", alpha=0.12, label="SWR-1")
    ax.axvspan(*swr_rev, color="tomato",    alpha=0.12, label="SWR-2")
    ax.legend(fontsize=7)
    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_ylabel("V_m (mV)", fontsize=9)
    ax.set_title("Membrane Voltage Traces  |  UPDATE-1: orange=SUP(reg.spiking), blue=DEEP(bursting)",
                 fontsize=10)
    fig.tight_layout()
    p = f"{save_prefix}_fig6_vmtraces.png"
    fig.savefig(p, dpi=150); plt.close(fig); out_paths.append(p)
    print(f"  saved {p}")

    return out_paths


# ============================================================================
# Console report  (UPDATE-7: extended for two-layer CA3)
# ============================================================================

def print_report(net, sim_ms):
    print("\n" + "=" * 70)
    print("SIMULATION REPORT  (Watson et al. 2025 two-layer CA3)")
    print("=" * 70)
    for label, pop, spk in [
        ("CA1 PYR",        net["PYR"],          net["spk_pyr"]),
        ("CA1 BASKET",     net["BASKET"],        net["spk_ba"]),
        ("CA1 OLM",        net["OLM"],           net["spk_olm"]),
        # UPDATE-7: Report SUP and DEEP separately
        ("CA3 SUP [UPDATE-1]",  net["CA3_SUP"],  net["spk_ca3_sup"]),
        ("CA3 DEEP[UPDATE-1]",  net["CA3_DEEP"], net["spk_ca3_deep"]),
        ("CA3 INT_SUP[UPD-5]",  net["CA3_INT_SUP"],  net["spk_ca3_int_sup"]),
        ("CA3 INT_DEEP[UPD-5]", net["CA3_INT_DEEP"], net["spk_ca3_int_deep"]),
    ]:
        ev   = nest.GetStatus(spk, "events")[0]
        rate = mean_rate(pop, spk, sim_ms)
        print(f"  {label:26s}: {len(ev['times']):6d} spikes | {rate:6.2f} Hz")

    print("\n--- Replay quality (Spearman rho, CA3 SUP sequence) ---")
    t_sup, s_sup = _get_spikes(net["spk_ca3_sup"])
    for label, win, expected_sign in [
        ("SWR-1 forward", net["swr_fwd"], +1),
        ("SWR-2 reverse", net["swr_rev"], -1),
    ]:
        rho, pval = replay_score(t_sup, s_sup, net["ca3_seq_groups"],
                                 win[0] - 5, win[1] + 30)
        if rho is not None and not np.isnan(rho):
            ok      = (expected_sign > 0 and rho > 0.5) or (expected_sign < 0 and rho < -0.5)
            verdict = "PASS" if ok else "WEAK"
            print(f"  {label:20s}: rho = {rho:+.3f}  p = {pval:.3f}  [{verdict}]")
        else:
            print(f"  {label:20s}: insufficient spikes or scipy missing")

    print("\n--- Deep layer following (DEEP fires after SUP during replay) ---")
    t_deep, s_deep = _get_spikes(net["spk_ca3_deep"])
    for label, (ws, we) in [("SWR-1 fwd", net["swr_fwd"]), ("SWR-2 rev", net["swr_rev"])]:
        n_sup_win  = np.sum((t_sup  >= ws) & (t_sup  <= we))
        n_deep_win = np.sum((t_deep >= ws) & (t_deep <= we))
        print(f"  {label}: SUP spikes={n_sup_win}  DEEP spikes={n_deep_win}"
              f"  ratio={n_deep_win/max(n_sup_win,1):.2f}")
    print("=" * 70)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    SIM_MS = 1000.0

    print("\n>>> Building Watson et al. 2025 two-layer CA3 replay network...")
    net = build_replay_network(
        N_ca1_pyr=200, N_ca1_basket=15, N_ca1_olm=12,
        # UPDATE-1: split CA3 pool (240 SUP + 60 DEEP = 300 total, ~20% deep)
        N_ca3_sup=240, N_ca3_deep=60,
        # UPDATE-5: two INT pools
        N_ca3_int_sup=30, N_ca3_int_deep=10,
        n_seq_groups=10,
        p_seq_fwd=0.28,  w_seq_fwd=1.50,
        p_seq_bwd=0.06,  w_seq_bwd=0.30,
        # UPDATE-4: SUP->DEEP feedforward
        p_sup_to_deep=0.28, w_sup_to_deep=1.30,
        swr_fwd_start=300.0, swr_fwd_stop=420.0,
        swr_rev_start=600.0, swr_rev_stop=720.0,
        scaffold_on=True, scaffold_step_ms=8.0,
        # UPDATE-2: differentiated drive (SUP high, DEEP ~3-4x lower)
        rate_dg_ca3_sup=820.0,  rate_dg_ca3_deep=220.0,
        rate_ec_ca3_sup=530.0,  rate_ec_ca3_deep=150.0,
        rate_ca3_drive_sup=400.0, rate_ca3_drive_deep=120.0,
        rate_drive_ca1_basket=820.0,
        rate_drive_ca3_int_sup=820.0, rate_drive_ca3_int_deep=820.0,
        theta_on=True, theta_hz=8.0,
    )

    print(f"\n>>> Simulating {SIM_MS} ms...")
    nest.Simulate(SIM_MS)

    print_report(net, SIM_MS)

    print("\n>>> Generating figures...")
    out_dir = os.path.join(_script_dir, "replay_output_watson2025")
    os.makedirs(out_dir, exist_ok=True)
    prefix  = os.path.join(out_dir, "bidir_replay_watson2025")
    paths   = plot_bidirectional_replay(net, sim_ms=SIM_MS, save_prefix=prefix)

    print(f"\n>>> All done.  Figures in: {out_dir}/")
    for p in paths:
        print(f"    {os.path.basename(p)}")
