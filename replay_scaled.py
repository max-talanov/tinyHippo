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
# SCALE CONFIGURATIONS
# ============================================================================

SCALE_CONFIGS = {

    # ---- 1% (default) -------------------------------------------------------
    # All counts divisible by n_seq_groups=20
    "1pct": dict(
        label             = "1% scale — test/debug",
        N_ca3_sup         = 2_640,    # 264k x 0.01 ; /20 = 132/group
        N_ca3_deep        = 660,      #  66k x 0.01 ; /20 =  33/group
        N_ca3_int_sup     = 240,
        N_ca3_int_deep    = 80,
        N_ca1_pyr         = 4_600,
        N_ca1_basket      = 140,
        N_ca1_olm         = 90,
        n_seq_groups      = 20,
        n_threads_default = 8,
        slurm_nodes       = 1,
        slurm_ntasks      = 4,        # 4 ranks x 8 threads = 32 cores
    ),

    # ---- 12% ----------------------------------------------------------------
    # All counts divisible by n_seq_groups=50
    "12pct": dict(
        label             = "12% scale — HPC development",
        N_ca3_sup         = 31_500,   # /50 = 630/group
        N_ca3_deep        = 7_500,    # /50 = 150/group
        N_ca3_int_sup     = 3_000,
        N_ca3_int_deep    = 1_000,
        N_ca1_pyr         = 55_200,
        N_ca1_basket      = 1_680,
        N_ca1_olm         = 1_080,
        n_seq_groups      = 50,
        n_threads_default = 28,
        slurm_nodes       = 16,
        slurm_ntasks      = 64,       # 4 ranks/node x 16 nodes
    ),

    # ---- 100% ---------------------------------------------------------------
    # All counts divisible by n_seq_groups=100
    "100pct": dict(
        label             = "100% scale — full rat hippocampus",
        N_ca3_sup         = 264_000,  # /100 = 2640/group
        N_ca3_deep        = 66_000,   # /100 =  660/group
        N_ca3_int_sup     = 24_000,
        N_ca3_int_deep    = 8_000,
        N_ca1_pyr         = 460_000,
        N_ca1_basket      = 14_000,
        N_ca1_olm         = 9_000,
        n_seq_groups      = 100,
        n_threads_default = 28,
        slurm_nodes       = 256,
        slurm_ntasks      = 1_024,    # 4 ranks/node x 256 nodes
    ),
}


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


def fixed_connect(pre, post, indegree, weight, delay):
    """
    NEST native fixed_indegree — C++, fully MPI-parallel.
    Each post neuron receives exactly `indegree` inputs drawn from pre.
    """
    nest.Connect(
        _to_nc(pre), _to_nc(post),
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

def print_report(net, sim_ms, scale_label):
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
    print(f"{'='*72}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bidirectional hippocampal replay — Watson et al. 2025 (v2: native NEST rules)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scale summary (v2 with native NEST connection rules):
  1pct    ~7,710 neurons    MacBook M3: ~6-10 min   | MN5 100 CPUs: ~2-3 min
  12pct  ~93,460 neurons    MacBook M3: ~1-3 h      | MN5 100 CPUs: ~12-25 min
  100pct ~781,000 neurons   MacBook M3: not feasible | MN5 100 CPUs: ~3-5 h

MareNostrum5 submission examples:
  sbatch --nodes=1   --ntasks=4    --cpus-per-task=28 run_replay_mn5.sh   # 1pct
  sbatch --nodes=16  --ntasks=64   --cpus-per-task=28 run_replay_mn5.sh   # 12pct
  sbatch --nodes=256 --ntasks=1024 --cpus-per-task=28 run_replay_mn5.sh   # 100pct
        """,
    )
    parser.add_argument(
        "--scale", choices=["1pct", "12pct", "100pct"], default="1pct",
        help="Network scale (default: 1pct)")
    parser.add_argument(
        "--sim-ms", type=float, default=1000.0,
        help="Simulation duration in ms (default: 1000)")
    parser.add_argument(
        "--threads", type=int, default=None,
        help="OpenMP threads per MPI rank (overrides OMP_NUM_THREADS and scale default)")
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip figure generation — recommended for 12pct/100pct on HPC")
    args = parser.parse_args()

    cfg = SCALE_CONFIGS[args.scale]

    n_threads = (args.threads
                 if args.threads is not None
                 else int(os.environ.get("OMP_NUM_THREADS", cfg["n_threads_default"])))

    SIM_MS = args.sim_ms
    total_N = (cfg["N_ca3_sup"] + cfg["N_ca3_deep"]
               + cfg["N_ca3_int_sup"] + cfg["N_ca3_int_deep"]
               + cfg["N_ca1_pyr"] + cfg["N_ca1_basket"] + cfg["N_ca1_olm"])

    print(f"\n{'='*72}")
    print(f"  Watson et al. 2025 — Bidirectional Replay  [v2: native NEST rules]")
    print(f"  Scale    : {cfg['label']}")
    print(f"  CA3_SUP  : {cfg['N_ca3_sup']:>10,}  CA3_DEEP : {cfg['N_ca3_deep']:>8,}")
    print(f"  CA1_PYR  : {cfg['N_ca1_pyr']:>10,}  groups   : {cfg['n_seq_groups']:>8,}")
    print(f"  Total N  : {total_N:>10,}")
    print(f"  Threads  : {n_threads} per MPI rank  |  Sim: {SIM_MS} ms")
    print(f"  MN5 hint : {cfg['slurm_nodes']} nodes, {cfg['slurm_ntasks']} MPI ranks")
    print(f"  Connect  : fixed_indegree + pairwise_bernoulli (C++, MPI-parallel)")
    print(f"{'='*72}\n")

    t_wall = time.perf_counter()

    print(">>> Building network...")
    net = build_replay_network(
        N_ca3_sup     = cfg["N_ca3_sup"],
        N_ca3_deep    = cfg["N_ca3_deep"],
        N_ca3_int_sup = cfg["N_ca3_int_sup"],
        N_ca3_int_deep= cfg["N_ca3_int_deep"],
        N_ca1_pyr     = cfg["N_ca1_pyr"],
        N_ca1_basket  = cfg["N_ca1_basket"],
        N_ca1_olm     = cfg["N_ca1_olm"],
        n_seq_groups  = cfg["n_seq_groups"],
        n_threads     = n_threads,
    )

    print(f"\n>>> Simulating {SIM_MS} ms...")
    t_sim = time.perf_counter()
    nest.Simulate(SIM_MS)
    print(f"    Simulation done in {time.perf_counter()-t_sim:.1f}s")

    print_report(net, SIM_MS, cfg["label"])

    if not args.no_figures:
        print("\n>>> Generating figures...")
        out_dir = os.path.join(_script_dir, f"replay_output_{args.scale}")
        os.makedirs(out_dir, exist_ok=True)
        prefix  = os.path.join(out_dir, f"bidir_replay_{args.scale}")
        paths   = plot_bidirectional_replay(net, sim_ms=SIM_MS, save_prefix=prefix)
        print(f">>> Figures saved to: {out_dir}/")
        for pp in paths:
            print(f"    {os.path.basename(pp)}")
    else:
        print("\n>>> Figure generation skipped (--no-figures).")

    print(f"\n>>> Total wall time: {time.perf_counter()-t_wall:.1f}s")
    print(">>> Done.")
