#!/usr/bin/env python3
"""
bidirectional_replay.py
============================
Bidirectional sequence replay over SWR events, built on top of tiny.py.

New additions vs tiny.py
  1.  _to_nc()                     – coerce any neuron-ID format to NodeCollection
  2.  conn_stats() override         – NEST-3.x-safe version (accepts lists or NC)
  3.  bernoulli_connect() override  – NEST-3.x-safe version (wraps IDs in NC)
  4.  sequence_connect_ca3()        – asymmetric forward/backward CA3 E->E chain
  5.  make_replay_trigger()         – brief Poisson seed that kicks one end
  6.  make_staggered_replay_drive() – per-group temporal scaffold (safety net)
  7.  build_replay_network()        – full CA1+CA3 net with 2 SWR events:
                                       SWR-1 -> forward replay
                                       SWR-2 -> reverse replay
  8.  plot_bidirectional_replay()   – 6-figure visualisation suite
  9.  replay_score()                – Spearman rho quantitative quality check
  10. print_report()                – console summary

Usage:
    python bidirectional_replay.py

Requirements:
    NEST >= 3.x, numpy, matplotlib, scipy (optional for replay score)
    tiny.py must be in the same directory.
"""

import sys
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless; swap to "TkAgg" / "Qt5Agg" for interactive
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# -- locate tiny.py ----------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import nest
# Import only what tiny.py has that we do NOT override below
from tiny import (
    safe_set_seeds,
    maybe_make_theta_generators,
    make_swr_event_generators,
)


# ============================================================================
# NEST-3.x compatibility layer
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
        f"{label:18s}: {n_conn:7d} conns | density={density:.4f} | "
        f"out={n_conn/max(n_pre,1):.2f} | in={n_conn/max(n_post,1):.2f}"
    )


def bernoulli_connect(pre, post, p: float, weight: float, delay: float, rng):
    """
    NEST-3.x-safe Bernoulli connectivity.

    Converts pre/post to plain int lists BEFORE building NodeCollections.
    This avoids:
      - The numpy 2.0 DeprecationWarning (np.array on NodeCollection)
      - The NEST connect_arrays path that demands numpy weight arrays
    """
    # Materialise to plain Python int lists first
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
# 1.  Sequence connectivity
# ============================================================================

def sequence_connect_ca3(
    ca3_pyr,
    n_groups: int   = 10,
    p_fwd: float    = 0.28,
    p_bwd: float    = 0.06,
    p_local: float  = 0.12,
    w_fwd: float    = 1.50,
    w_bwd: float    = 0.30,
    w_local: float  = 0.90,
    delay: float    = 1.5,
    rng=None,
) -> list:
    """
    Wire CA3 pyramidal cells with ordered, asymmetric Hebbian weights.

    w_fwd >> w_bwd is the key mechanism:
      - Forward  replay travels the strong  k -> k+1 chain.
      - Reverse  replay travels the weak    k -> k-1 chain
        (non-zero so it can self-sustain after seeding from the top).
    If w_bwd == 0, reverse replay will fail to propagate.

    Returns list[list[int]] -- one sublist of plain neuron IDs per group.
    """
    if rng is None:
        rng = np.random.default_rng(99)

    # Use .tolist() to get plain ints -- never pass NodeCollection to np.array
    all_ids = list(ca3_pyr.tolist())
    n_total = len(all_ids)
    assert n_total % n_groups == 0, (
        f"N_ca3_pyr ({n_total}) must be divisible by n_groups ({n_groups})"
    )
    gs     = n_total // n_groups
    groups = [all_ids[k * gs : (k + 1) * gs] for k in range(n_groups)]

    for k, grp_k in enumerate(groups):
        bernoulli_connect(grp_k, grp_k, p_local, w_local, delay, rng)
        if k + 1 < n_groups:
            bernoulli_connect(grp_k, groups[k + 1], p_fwd, w_fwd, delay, rng)
        if k - 1 >= 0:
            bernoulli_connect(grp_k, groups[k - 1], p_bwd, w_bwd, delay, rng)

    return groups   # list[list[int]]


# ============================================================================
# 2.  Replay trigger
# ============================================================================

def make_replay_trigger(
    group_ids,
    trigger_start_ms: float,
    trigger_dur_ms: float   = 16.0,
    trigger_rate: float     = 2600.0,
    weight: float           = 0.95,
    delay: float            = 1.0,
):
    """
    Brief high-rate Poisson seed targeting one sequence group.
      Forward replay  -> point at group 0   (low index)
      Reverse replay  -> point at group N-1 (high index)
    """
    ids = [int(i) for i in group_ids]   # always plain ints
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
# 3.  Staggered scaffold  (safety net for small N)
# ============================================================================

def make_staggered_replay_drive(
    seq_groups: list,
    swr_start_ms: float,
    direction: str       = "forward",
    inter_step_ms: float = 8.0,
    drive_dur_ms: float  = 10.0,
    drive_rate: float    = 750.0,
    weight: float        = 0.55,
    delay: float         = 1.0,
):
    """
    Per-group, time-staggered Poisson bursts.

    Each group receives a brief burst offset by inter_step_ms from the
    previous group, scaffolding the temporal envelope so replay propagates
    even when recurrent weights alone are marginally strong.

    direction: "forward" -> group 0 first; "reverse" -> group N-1 first.
    """
    n     = len(seq_groups)
    order = list(range(n)) if direction == "forward" else list(range(n - 1, -1, -1))
    all_gens = []
    for step, k in enumerate(order):
        t_start = swr_start_ms + step * inter_step_ms
        ids     = [int(i) for i in seq_groups[k]]   # plain ints
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
# 4.  Network builder
# ============================================================================

def build_replay_network(
    # CA1 sizes
    N_ca1_pyr: int    = 200,
    N_ca1_basket: int = 15,
    N_ca1_olm: int    = 12,
    # CA3 sizes  (N_ca3_pyr must be divisible by n_seq_groups)
    N_ca3_pyr: int    = 300,
    N_ca3_int: int    = 40,
    # sequence structure
    n_seq_groups: int  = 10,
    p_seq_fwd: float   = 0.28,
    p_seq_bwd: float   = 0.06,
    w_seq_fwd: float   = 1.50,
    w_seq_bwd: float   = 0.30,
    # SWR event windows [ms]
    swr_fwd_start: float = 300.0,
    swr_fwd_stop:  float = 420.0,
    swr_rev_start: float = 600.0,
    swr_rev_stop:  float = 720.0,
    # SWR generator params
    swr_sharpwave_rate: float = 280.0,
    swr_ripple_hz:      float = 180.0,
    swr_ripple_mean:    float = 1100.0,
    swr_ripple_amp:     float = 850.0,
    # replay trigger params
    trigger_dur_ms: float = 16.0,
    trigger_rate:   float = 2600.0,
    trigger_weight: float = 0.95,
    # staggered scaffold
    scaffold_on: bool       = True,
    scaffold_step_ms: float = 8.0,
    scaffold_rate:    float = 720.0,
    scaffold_weight:  float = 0.55,
    # background drive rates [Hz]
    rate_ec_ca1_pyr:       float = 580.0,
    rate_dg_ca3_pyr:       float = 820.0,
    rate_ec_ca3_pyr:       float = 530.0,
    rate_ca3_drive_pyr:    float = 400.0,
    rate_drive_ca1_basket: float = 820.0,
    rate_drive_ca3_int:    float = 820.0,
    # theta drive
    theta_on: bool    = True,
    theta_hz: float   = 8.0,
    theta_mean: float = 1100.0,
    theta_amp:  float = 1000.0,
    # CA1 local connectivity
    p_ca1_EE: float = 0.02,
    p_ca1_EI: float = 0.10,
    p_ca1_IE: float = 0.15,
    p_ca1_OE: float = 0.10,
    # CA3 inter-population  (E->E handled by sequence_connect_ca3)
    p_ca3_EI: float = 0.12,
    p_ca3_IE: float = 0.20,
    p_ca3_II: float = 0.10,
    # Schaffer collaterals
    p_schaffer_pyr:    float = 0.10,
    p_schaffer_basket: float = 0.12,
    # misc
    seed_connect: int = 42,
    n_threads: int    = 4,
):
    """Build full CA1+CA3 network with bidirectional replay."""

    nest.ResetKernel()
    nest.SetKernelStatus({
        "resolution":        0.1,
        "local_num_threads": n_threads,
        "print_time":        True,
        "overwrite_files":   True,
    })
    safe_set_seeds()

    # NEST 3.x uses nest.node_models; graceful fallback for older builds
    try:
        available = list(nest.node_models)
    except AttributeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            available = list(nest.Models("nodes"))
    if "izhikevich" not in available:
        raise RuntimeError("NEST model 'izhikevich' not found in this build.")

    # -- Izhikevich parameter sets -------------------------------------------
    pyr_params    = dict(a=0.02, b=0.2,  c=-65.0, d=8.0, V_m=-65.0, U_m=-13.0,  I_e=0.0)
    basket_params = dict(a=0.10, b=0.2,  c=-65.0, d=2.0, V_m=-65.0, U_m=-13.0,  I_e=0.0)
    olm_params    = dict(a=0.02, b=0.25, c=-65.0, d=2.0, V_m=-65.0, U_m=-16.25, I_e=0.0)

    # -- Populations ---------------------------------------------------------
    CA1_PYR    = nest.Create("izhikevich", N_ca1_pyr,    params=pyr_params)
    CA1_BASKET = nest.Create("izhikevich", N_ca1_basket, params=basket_params)
    CA1_OLM    = nest.Create("izhikevich", N_ca1_olm,    params=olm_params)
    CA3_PYR    = nest.Create("izhikevich", N_ca3_pyr,    params=pyr_params)
    CA3_INT    = nest.Create("izhikevich", N_ca3_int,    params=basket_params)

    # -- Background / external inputs ----------------------------------------
    ECIII_CA1  = nest.Create("poisson_generator", N_ca1_pyr,    params={"rate": rate_ec_ca1_pyr})
    DRIVE_CA1B = nest.Create("poisson_generator", N_ca1_basket, params={"rate": rate_drive_ca1_basket})
    EC_CA3     = nest.Create("poisson_generator", N_ca3_pyr,    params={"rate": rate_ec_ca3_pyr})
    DG_CA3     = nest.Create("poisson_generator", N_ca3_pyr,    params={"rate": rate_dg_ca3_pyr})
    DRIV_CA3   = nest.Create("poisson_generator", N_ca3_pyr,    params={"rate": rate_ca3_drive_pyr})
    DRIV_CA3I  = nest.Create("poisson_generator", N_ca3_int,    params={"rate": rate_drive_ca3_int})

    d_fast = 1.5;  d_slow = 3.0

    nest.Connect(ECIII_CA1,  CA1_PYR,    conn_spec="one_to_one", syn_spec={"weight": 2.0, "delay": d_slow})
    nest.Connect(DRIVE_CA1B, CA1_BASKET, conn_spec="one_to_one", syn_spec={"weight": 2.0, "delay": d_fast})
    nest.Connect(EC_CA3,     CA3_PYR,    conn_spec="one_to_one", syn_spec={"weight": 2.0, "delay": d_slow})
    nest.Connect(DG_CA3,     CA3_PYR,    conn_spec="one_to_one", syn_spec={"weight": 3.0, "delay": d_fast})
    nest.Connect(DRIV_CA3,   CA3_PYR,    conn_spec="one_to_one", syn_spec={"weight": 2.0, "delay": d_fast})
    nest.Connect(DRIV_CA3I,  CA3_INT,    conn_spec="one_to_one", syn_spec={"weight": 2.0, "delay": d_fast})

    # -- Theta drive (optional) ----------------------------------------------
    if theta_on:
        for pop, w_th in [
            (CA3_PYR, 0.80), (CA3_INT, 1.80),
            (CA1_PYR, 1.00), (CA1_BASKET, 2.00), (CA1_OLM, 2.00),
        ]:
            th = maybe_make_theta_generators(len(pop), theta_mean, theta_amp, theta_hz)
            if th is not None:
                nest.Connect(th, pop, conn_spec="one_to_one",
                             syn_spec={"weight": float(w_th), "delay": 1.0})

    # -- SWR event generators: two events (fwd + rev) ------------------------
    for swr_s, swr_e in [(swr_fwd_start, swr_fwd_stop), (swr_rev_start, swr_rev_stop)]:
        sw_sh_c3, sw_rip_c3 = make_swr_event_generators(
            n=N_ca3_pyr, start_ms=swr_s, stop_ms=swr_e,
            sharpwave_rate=swr_sharpwave_rate,
            ripple_rate_mean=swr_ripple_mean, ripple_rate_amp=swr_ripple_amp,
            ripple_hz=swr_ripple_hz)
        nest.Connect(sw_sh_c3,  CA3_PYR, conn_spec="one_to_one", syn_spec={"weight": 0.35, "delay": 1.0})
        nest.Connect(sw_rip_c3, CA3_PYR, conn_spec="one_to_one", syn_spec={"weight": 0.15, "delay": 1.0})

        _, sw_rip_c3i = make_swr_event_generators(
            n=N_ca3_int, start_ms=swr_s, stop_ms=swr_e,
            sharpwave_rate=0.0,
            ripple_rate_mean=swr_ripple_mean, ripple_rate_amp=swr_ripple_amp,
            ripple_hz=swr_ripple_hz)
        nest.Connect(sw_rip_c3i, CA3_INT, conn_spec="one_to_one", syn_spec={"weight": 0.60, "delay": 1.0})

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
        nest.Connect(sw_rip_c1b, CA1_BASKET, conn_spec="one_to_one", syn_spec={"weight": 0.55, "delay": 1.0})

    # -- Recurrent connectivity ----------------------------------------------
    rng = np.random.default_rng(seed_connect)

    # CA3: asymmetric sequence chain (replaces uniform E->E)
    ca3_seq_groups = sequence_connect_ca3(
        CA3_PYR, n_groups=n_seq_groups,
        p_fwd=p_seq_fwd, p_bwd=p_seq_bwd,
        w_fwd=w_seq_fwd, w_bwd=w_seq_bwd,
        p_local=0.12, w_local=0.9,
        delay=d_fast, rng=rng,
    )

    # CA3 E->I, I->E, I->I
    bernoulli_connect(CA3_PYR, CA3_INT, p_ca3_EI,  1.6,  d_fast, rng)
    bernoulli_connect(CA3_INT, CA3_PYR, p_ca3_IE, -5.5,  d_fast, rng)
    bernoulli_connect(CA3_INT, CA3_INT, p_ca3_II, -4.5,  d_fast, rng)

    # CA1 local
    bernoulli_connect(CA1_PYR,    CA1_PYR,    p_ca1_EE,  0.8,  d_fast, rng)
    bernoulli_connect(CA1_PYR,    CA1_BASKET, p_ca1_EI,  1.5,  d_fast, rng)
    bernoulli_connect(CA1_PYR,    CA1_OLM,    0.08,      1.2,  d_slow, rng)
    bernoulli_connect(CA1_BASKET, CA1_PYR,    p_ca1_IE, -5.0,  d_fast, rng)
    bernoulli_connect(CA1_BASKET, CA1_BASKET, 0.10,     -4.0,  d_fast, rng)
    bernoulli_connect(CA1_OLM,    CA1_PYR,    p_ca1_OE, -3.0,  d_slow, rng)

    # Schaffer collaterals
    bernoulli_connect(CA3_PYR, CA1_PYR,    p_schaffer_pyr,    2.4, d_slow, rng)
    bernoulli_connect(CA3_PYR, CA1_BASKET, p_schaffer_basket, 2.8, d_fast, rng)

    # -- Replay triggers -----------------------------------------------------
    make_replay_trigger(ca3_seq_groups[0],  swr_fwd_start,
                        trigger_dur_ms=trigger_dur_ms,
                        trigger_rate=trigger_rate, weight=trigger_weight)
    make_replay_trigger(ca3_seq_groups[-1], swr_rev_start,
                        trigger_dur_ms=trigger_dur_ms,
                        trigger_rate=trigger_rate, weight=trigger_weight)

    # -- Staggered scaffold (optional) ---------------------------------------
    if scaffold_on:
        make_staggered_replay_drive(
            ca3_seq_groups, swr_fwd_start, direction="forward",
            inter_step_ms=scaffold_step_ms, drive_rate=scaffold_rate, weight=scaffold_weight)
        make_staggered_replay_drive(
            ca3_seq_groups, swr_rev_start, direction="reverse",
            inter_step_ms=scaffold_step_ms, drive_rate=scaffold_rate, weight=scaffold_weight)

    # -- Recorders -----------------------------------------------------------
    spk_ca1_pyr = nest.Create("spike_recorder")
    spk_ca1_ba  = nest.Create("spike_recorder")
    spk_ca1_olm = nest.Create("spike_recorder")
    nest.Connect(CA1_PYR,    spk_ca1_pyr)
    nest.Connect(CA1_BASKET, spk_ca1_ba)
    nest.Connect(CA1_OLM,    spk_ca1_olm)

    spk_ca3_pyr = nest.Create("spike_recorder")
    spk_ca3_int = nest.Create("spike_recorder")
    nest.Connect(CA3_PYR, spk_ca3_pyr)
    nest.Connect(CA3_INT, spk_ca3_int)

    try:
        vm = nest.Create("multimeter", params={"record_from": ["V_m", "U_m"], "interval": 0.2})
    except Exception:
        vm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 0.2})
    nest.Connect(vm, CA1_PYR[:5])
    nest.Connect(vm, CA3_PYR[:5])

    # -- Connectivity stats --------------------------------------------------
    print("\n=== Connectivity stats ===")
    conn_stats("CA3E->CA3I",  CA3_PYR, CA3_INT)
    conn_stats("CA3I->CA3E",  CA3_INT, CA3_PYR)
    conn_stats("Sch->CA1E",   CA3_PYR, CA1_PYR)
    conn_stats("CA1E->CA1E",  CA1_PYR, CA1_PYR)
    conn_stats("CA1I->CA1E",  CA1_BASKET, CA1_PYR)
    # Sample forward link to confirm sequence wiring
    conn_stats("Seq g0->g1",
               nest.NodeCollection(ca3_seq_groups[0]),
               nest.NodeCollection(ca3_seq_groups[1]))

    return dict(
        PYR=CA1_PYR, BASKET=CA1_BASKET, OLM=CA1_OLM,
        CA3_PYR=CA3_PYR, CA3_INT=CA3_INT,
        spk_pyr=spk_ca1_pyr, spk_ba=spk_ca1_ba, spk_olm=spk_ca1_olm,
        spk_ca3_pyr=spk_ca3_pyr, spk_ca3_int=spk_ca3_int,
        vm=vm,
        ca3_seq_groups=ca3_seq_groups,
        n_seq_groups=n_seq_groups,
        swr_on=True,
        swr_fwd=(swr_fwd_start, swr_fwd_stop),
        swr_rev=(swr_rev_start, swr_rev_stop),
        swr_events=[(swr_fwd_start, swr_fwd_stop), (swr_rev_start, swr_rev_stop)],
        swr_ripple_hz=swr_ripple_hz,
        theta_on=theta_on, theta_hz=theta_hz,
    )


# ============================================================================
# 5.  Replay quality metric
# ============================================================================

def replay_score(spk_times, spk_senders, seq_groups, window_start, window_stop):
    """
    Spearman rho between sequence group index and mean spike time within window.
    rho ~ +1.0  -> clean forward replay
    rho ~ -1.0  -> clean reverse replay
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
# 6.  Visualisation
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

    Fig 1 -- Full overview raster + rate traces
    Fig 2 -- Sequence group heatmap (groups x time)
    Fig 3 -- Side-by-side zoom: forward vs reverse window
    Fig 4 -- Per-group peak activation time scatter (slope = direction)
    Fig 5 -- Ripple-band inhibitory rates around each SWR
    Fig 6 -- Per-neuron membrane voltage traces
    """
    t_ca3p, s_ca3p = _get_spikes(net["spk_ca3_pyr"])
    t_ca3i, s_ca3i = _get_spikes(net["spk_ca3_int"])
    t_ca1p, s_ca1p = _get_spikes(net["spk_pyr"])
    t_ca1b, s_ca1b = _get_spikes(net["spk_ba"])

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

    # -- Fig 1: Overview -----------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Bidirectional Replay -- Simulation Overview",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    for k, grp in enumerate(seq_groups):
        m = np.isin(s_ca3p, np.array(grp))
        ax.scatter(t_ca3p[m], s_ca3p[m], s=1.5, color=group_colors[k], rasterized=True)
    shade(ax)
    ax.set_ylabel("CA3 PYR ID", fontsize=9)
    ax.set_title("A  CA3 PYR raster  [colour = sequence group]", fontsize=9, loc="left")
    sm = ScalarMappable(cmap=cmap_seq, norm=Normalize(0, n_groups - 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.01).set_label("Group #", fontsize=8)

    ax = axes[1]
    ax.scatter(t_ca1p, s_ca1p, s=1.2, color="slategray", rasterized=True)
    shade(ax)
    ax.set_ylabel("CA1 PYR ID", fontsize=9)
    ax.set_title("B  CA1 PYR raster", fontsize=9, loc="left")

    ax = axes[2]
    tc, rc3 = _binned_rate(t_ca3p, len(net["CA3_PYR"]), sim_ms, 10.0)
    tc, rc1 = _binned_rate(t_ca1p, len(net["PYR"]),     sim_ms, 10.0)
    ax.plot(tc, rc3, color="darkorange", lw=1.2, label="CA3 PYR")
    ax.plot(tc, rc1, color="steelblue",  lw=1.2, label="CA1 PYR")
    shade(ax)
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylabel("Rate (Hz)", fontsize=9)
    ax.set_title("C  Population rates  [10 ms bins]", fontsize=9, loc="left")

    ax = axes[3]
    tf, ri = _binned_rate(t_ca3i, len(net["CA3_INT"]), sim_ms, 2.0)
    tf, rb = _binned_rate(t_ca1b, len(net["BASKET"]),  sim_ms, 2.0)
    ax.plot(tf, ri, color="firebrick",    lw=0.8, alpha=0.85, label="CA3 INT")
    ax.plot(tf, rb, color="mediumorchid", lw=0.8, alpha=0.85, label="CA1 Basket")
    shade(ax)
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_ylabel("Rate (Hz)", fontsize=9)
    ax.set_title("D  Inhibitory rates  [2 ms ripple-scale bins]", fontsize=9, loc="left")

    fig.tight_layout()
    p = f"{save_prefix}_fig1_overview.png"
    fig.savefig(p, dpi=150); plt.close(fig); out_paths.append(p)
    print(f"  saved {p}")

    # -- Fig 2: Heatmap ------------------------------------------------------
    bin_ms = 5.0
    edges  = np.arange(0.0, sim_ms + bin_ms, bin_ms)
    n_bins = len(edges) - 1
    gs_per = len(seq_groups[0])
    heat   = np.zeros((n_groups, n_bins))
    for k, grp in enumerate(seq_groups):
        m = np.isin(s_ca3p, np.array(grp))
        counts, _ = np.histogram(t_ca3p[m], bins=edges)
        heat[k]   = counts / (bin_ms / 1e3) / max(gs_per, 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(heat, aspect="auto", origin="lower",
                   extent=[0, sim_ms, -0.5, n_groups - 0.5],
                   cmap="inferno", interpolation="nearest")
    fig.colorbar(im, ax=ax, pad=0.02).set_label("Rate (Hz)", fontsize=9)
    ax.axvspan(*swr_fwd, color="white", alpha=0.20, label="SWR-1 fwd")
    ax.axvspan(*swr_rev, color="cyan",  alpha=0.15, label="SWR-2 rev")
    # Expected diagonal slopes
    ax.plot([swr_fwd[0], swr_fwd[0] + (swr_fwd[1]-swr_fwd[0]) * 0.75],
            [0, n_groups - 1], "--w", lw=1.8, alpha=0.9, label="Expected fwd slope")
    ax.plot([swr_rev[0], swr_rev[0] + (swr_rev[1]-swr_rev[0]) * 0.75],
            [n_groups - 1, 0], "--c", lw=1.8, alpha=0.9, label="Expected rev slope")
    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_ylabel("Sequence group #", fontsize=10)
    ax.set_title("Sequence Group Heatmap -- CA3 PYR firing rate [Hz]", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    p = f"{save_prefix}_fig2_heatmap.png"
    fig.savefig(p, dpi=150); plt.close(fig); out_paths.append(p)
    print(f"  saved {p}")

    # -- Fig 3: Zoom forward vs reverse --------------------------------------
    pad = 30.0
    fig, (axf, axr) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Zoom: Forward  vs  Reverse Replay  (CA3 PYR, group-coloured)",
                 fontsize=12, fontweight="bold")

    for ax, (s, e), title in [
        (axf, swr_fwd, "SWR-1 -> Forward\n(seed: group 0, sweep up)"),
        (axr, swr_rev, "SWR-2 -> Reverse\n(seed: group N-1, sweep down)"),
    ]:
        for k, grp in enumerate(seq_groups):
            m = (t_ca3p >= s - pad) & (t_ca3p <= e + pad) & np.isin(s_ca3p, np.array(grp))
            ax.scatter(t_ca3p[m], s_ca3p[m], s=5, color=group_colors[k], rasterized=True)
        ax.axvspan(s, e, color="gold", alpha=0.22, label="SWR window")
        ax.set_xlim(s - pad, e + pad)
        ax.set_xlabel("Time (ms)", fontsize=9)
        ax.set_ylabel("Neuron ID", fontsize=9)
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
    fig.suptitle("Per-Group Peak Activation Time  (slope sign = replay direction)",
                 fontsize=11, fontweight="bold")
    for ax, (s, e), title in [
        (ax4f, swr_fwd, "SWR-1: Forward  (positive slope = correct)"),
        (ax4r, swr_rev, "SWR-2: Reverse  (negative slope = correct)"),
    ]:
        gi, pt = peak_times_per_group(t_ca3p, s_ca3p, seq_groups, s, e)
        colors_pt = [group_colors[k] for k in gi]
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

    # -- Fig 5: Ripple-band rates --------------------------------------------
    fig, axes5 = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Ripple-Band Inhibitory Rates  [2 ms bins]",
                 fontsize=11, fontweight="bold")
    for ax, (s, e), title in [
        (axes5[0], swr_fwd, "SWR-1 (Forward)"),
        (axes5[1], swr_rev, "SWR-2 (Reverse)"),
    ]:
        pad2    = 50.0
        tb, rb2 = _binned_rate(t_ca1b, len(net["BASKET"]),  sim_ms, 2.0)
        ti, ri2 = _binned_rate(t_ca3i, len(net["CA3_INT"]), sim_ms, 2.0)
        m = (tb >= s - pad2) & (tb <= e + pad2)
        ax.plot(tb[m], rb2[m], color="mediumorchid", lw=1.0, label="CA1 Basket")
        ax.plot(ti[m], ri2[m], color="firebrick",    lw=1.0, alpha=0.8, label="CA3 INT")
        ax.axvspan(s, e, color="gold", alpha=0.25, label="SWR window")
        ax.set_xlabel("Time (ms)", fontsize=9)
        ax.set_ylabel("Rate (Hz)", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)

    fig.tight_layout()
    p = f"{save_prefix}_fig5_ripple_rates.png"
    fig.savefig(p, dpi=150); plt.close(fig); out_paths.append(p)
    print(f"  saved {p}")

    # -- Fig 6: Vm traces ----------------------------------------------------
    ev_vm = nest.GetStatus(net["vm"], "events")[0]
    vt    = np.array(ev_vm["times"])
    vs    = np.array(ev_vm["senders"])
    vv    = np.array(ev_vm["V_m"])

    fig, ax = plt.subplots(figsize=(14, 4))
    for gid in np.unique(vs):
        m   = vs == gid
        idx = np.argsort(vt[m])
        ax.plot(vt[m][idx], vv[m][idx], lw=0.5, alpha=0.8)
    ax.axvspan(*swr_fwd, color="steelblue", alpha=0.12, label="SWR-1")
    ax.axvspan(*swr_rev, color="tomato",    alpha=0.12, label="SWR-2")
    ax.legend(fontsize=7)
    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_ylabel("V_m (mV)", fontsize=9)
    ax.set_title("Membrane Voltage Traces  (per neuron, no diagonal artefact)", fontsize=10)
    fig.tight_layout()
    p = f"{save_prefix}_fig6_vmtraces.png"
    fig.savefig(p, dpi=150); plt.close(fig); out_paths.append(p)
    print(f"  saved {p}")

    return out_paths


# ============================================================================
# 7.  Console report
# ============================================================================

def print_report(net, sim_ms):
    print("\n" + "=" * 62)
    print("SIMULATION REPORT")
    print("=" * 62)
    for label, pop, spk in [
        ("CA1 PYR",    net["PYR"],     net["spk_pyr"]),
        ("CA1 BASKET", net["BASKET"],  net["spk_ba"]),
        ("CA1 OLM",    net["OLM"],     net["spk_olm"]),
        ("CA3 PYR",    net["CA3_PYR"], net["spk_ca3_pyr"]),
        ("CA3 INT",    net["CA3_INT"], net["spk_ca3_int"]),
    ]:
        ev   = nest.GetStatus(spk, "events")[0]
        rate = mean_rate(pop, spk, sim_ms)
        print(f"  {label:12s}: {len(ev['times']):6d} spikes | {rate:6.2f} Hz")

    print("\n--- Replay quality (Spearman rho) ---")
    t_ca3p, s_ca3p = _get_spikes(net["spk_ca3_pyr"])
    for label, win, expected_sign in [
        ("SWR-1 forward", net["swr_fwd"], +1),
        ("SWR-2 reverse", net["swr_rev"], -1),
    ]:
        rho, pval = replay_score(t_ca3p, s_ca3p, net["ca3_seq_groups"],
                                 win[0] - 5, win[1] + 30)
        if rho is not None and not np.isnan(rho):
            ok      = (expected_sign > 0 and rho > 0.5) or (expected_sign < 0 and rho < -0.5)
            verdict = "PASS" if ok else "WEAK"
            print(f"  {label:18s}: rho = {rho:+.3f}  p = {pval:.3f}  [{verdict}]")
        else:
            print(f"  {label:18s}: insufficient spikes or scipy missing")
    print("=" * 62)


# ============================================================================
# 8.  Main
# ============================================================================

if __name__ == "__main__":
    SIM_MS = 1000.0

    print("\n>>> Building replay network...")
    net = build_replay_network(
        N_ca1_pyr=200,  N_ca1_basket=15, N_ca1_olm=12,
        N_ca3_pyr=300,  N_ca3_int=40,
        n_seq_groups=10,
        p_seq_fwd=0.28,  w_seq_fwd=1.50,
        p_seq_bwd=0.06,  w_seq_bwd=0.30,
        swr_fwd_start=300.0, swr_fwd_stop=420.0,
        swr_rev_start=600.0, swr_rev_stop=720.0,
        scaffold_on=True,
        scaffold_step_ms=8.0,
        rate_ec_ca1_pyr=580.0,
        rate_dg_ca3_pyr=820.0,
        rate_ec_ca3_pyr=530.0,
        rate_ca3_drive_pyr=400.0,
        rate_drive_ca1_basket=820.0,
        rate_drive_ca3_int=820.0,
        theta_on=True, theta_hz=8.0,
    )

    print(f"\n>>> Simulating {SIM_MS} ms...")
    nest.Simulate(SIM_MS)

    print_report(net, SIM_MS)

    print("\n>>> Generating figures...")
    out_dir = os.path.join(_script_dir, "replay_output")
    os.makedirs(out_dir, exist_ok=True)
    prefix  = os.path.join(out_dir, "bidir_replay")
    paths   = plot_bidirectional_replay(net, sim_ms=SIM_MS, save_prefix=prefix)

    print(f"\n>>> All done.  Figures in: {out_dir}/")
    for p in paths:
        print(f"    {os.path.basename(p)}")
