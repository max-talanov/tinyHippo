#!/usr/bin/env python3
"""
CA1-like microcircuit in NEST using *only* Izhikevich neurons + spike generators as inputs.

Key points (per NEST izhikevich model):
- Incoming spikes change V_m directly by the synaptic weight (no PSC kernel inside this neuron model).
- So weights are in mV “jumps” and can be much smaller than iaf_psc_exp currents.

Populations:
  PYR    : Regular Spiking (RS) Izhikevich params
  BASKET : Fast Spiking (FS) params
  OLM    : Low-Threshold Spiking-ish (LTS) params (proxy for O-LM style inhibition)

Inputs (generators only):
  CA3-like independent Poisson -> PYR and BASKET
  ECIII-like independent Poisson -> PYR
"""

import nest
import numpy as np
from collections import defaultdict


# -------------------------
# Helpers
# -------------------------

def safe_set_seeds(master_seed: int = 20260111):
    n_threads = nest.GetKernelStatus().get("local_num_threads", 1)
    try:
        nest.SetKernelStatus(
            {
                "grng_seed": master_seed,
                "rng_seeds": list(range(master_seed + 1, master_seed + 1 + n_threads)),
            }
        )
    except Exception:
        # older fallback
        try:
            nest.SetKernelStatus({"rng_seed": master_seed})
        except Exception:
            pass


def bernoulli_connect(pre, post, p, weight, delay, rng):
    """Explicit Bernoulli connectivity; easy to hack for motif studies."""
    pre_ids = np.array(pre, dtype=int)
    post_ids = np.array(post, dtype=int)
    for src in pre_ids:
        mask = rng.random(post_ids.size) < p
        targets = post_ids[mask].tolist()
        if targets:
            nest.Connect([int(src)], targets, syn_spec={"weight": float(weight), "delay": float(delay)})


def conn_stats(label: str, pre, post):
    conns = nest.GetConnections(pre, post)
    n_pre, n_post = len(pre), len(post)
    try:
        n_conn = conns.get("source").size
    except Exception:
        n_conn = len(conns)
    density = n_conn / (n_pre * n_post) if n_pre * n_post > 0 else 0.0
    print(
        f"{label:12s}: {n_conn:7d} connections | density={density:.4f} | "
        f"avg outdegree={n_conn/n_pre:.2f} | avg indegree={n_conn/n_post:.2f}"
    )


def mean_rate(pop, spk, sim_ms: float) -> float:
    ev = nest.GetStatus(spk, "events")[0]
    return len(ev["senders"]) / (len(pop) * (sim_ms / 1000.0))



def _events_to_spike_dict(*spike_recorders):
    """
    Build {gid: np.ndarray(times_ms)} from one or more NEST spike_recorders.
    Works for neuron populations and also poisson_generators (inputs).
    """
    spikes = defaultdict(list)
    for rec in spike_recorders:
        ev = nest.GetStatus(rec, "events")[0]
        senders = ev.get("senders", [])
        times = ev.get("times", [])
        for s, t in zip(senders, times):
            spikes[int(s)].append(float(t))
    # sort + convert to arrays for fast indexing
    for gid in list(spikes.keys()):
        arr = np.asarray(spikes[gid], dtype=float)
        arr.sort()
        spikes[gid] = arr
    return spikes


def _get_conn_arrays(target_gids):
    # NEST requires NodeCollection for GetConnections(target=...)
    if target_gids is None:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    # Accept either NodeCollection or a Python iterable of gids
    if not isinstance(target_gids, nest.NodeCollection):
        target_gids = nest.NodeCollection(list(target_gids))

    if len(target_gids) == 0:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    conns = nest.GetConnections(source=None, target=target_gids)

    # GetStatus returns a list aligned with conns
    sources = np.array(nest.GetStatus(conns, "source"), dtype=int)
    weights = np.array(nest.GetStatus(conns, "weight"), dtype=float)
    delays  = np.array(nest.GetStatus(conns, "delay"),  dtype=float)
    return sources, weights, delays


def compute_lfp_proxy(
    spikes_by_gid: dict,
    target_gids,
    sim_ms: float,
    dt: float = 0.1,
    tau_ms: float = 5.0,
    normalize_by_targets: bool = True,
):
    """
    LFP proxy for a population based on *summed synaptic events* into that population.

    Why a proxy?
      NEST's 'izhikevich' neuron updates V_m directly on spike arrival and does not expose
      synaptic currents (I_syn). So we approximate the population LFP as the weighted sum of
      incoming spike events (weight in mV jump), optionally convolved with an exponential
      kernel to mimic postsynaptic filtering.

    Returns:
      t (ms), lfp (arbitrary units ~ mV jump / neuron after normalization)
    """
    target_gids = list(target_gids)
    n_targets = len(target_gids)
    nbins = int(np.floor(sim_ms / dt)) + 1
    lfp = np.zeros(nbins, dtype=float)

    sources, weights, delays = _get_conn_arrays(target_gids)

    # accumulate weighted impulses at (t_spike + delay)
    for src, w, d in zip(sources, weights, delays):
        ts = spikes_by_gid.get(int(src))
        if ts is None or ts.size == 0:
            continue
        idx = np.floor((ts + d) / dt).astype(int)
        idx = idx[(idx >= 0) & (idx < nbins)]
        if idx.size:
            np.add.at(lfp, idx, w)

    # simple exponential "synaptic" smoothing
    if tau_ms and tau_ms > 0:
        k_len = max(1, int(np.ceil(10.0 * tau_ms / dt)))
        tk = np.arange(k_len, dtype=float) * dt
        kernel = np.exp(-tk / tau_ms)
        lfp = np.convolve(lfp, kernel, mode="full")[:nbins]

    if normalize_by_targets and n_targets > 0:
        lfp = lfp / float(n_targets)

    t = np.arange(nbins, dtype=float) * dt
    return t, lfp

# -------------------------
# Build CA1 microcircuit
# -------------------------

def build_ca1_izh(
    N_pyr=200,
    N_basket=40,
    N_olm=30,
    p_EE=0.02,
    p_EI=0.10,
    p_IE=0.15,
    p_OE=0.10,
    rate_ca3_pyr=1200.0,   # Hz per generator (independent per neuron)
    rate_ec_pyr=900.0,
    rate_ca3_ba=1200.0,
):
    nest.ResetKernel()
    nest.SetKernelStatus(
        {
            "resolution": 0.1,        # ms
            "local_num_threads": 4,   # tune for your CPU
            "print_time": True,
            "overwrite_files": True,
        }
    )
    safe_set_seeds()

    if "izhikevich" not in nest.Models("nodes"):
        raise RuntimeError("NEST model 'izhikevich' not found in this installation.")

    ''' --- Izhikevich parameter sets (canonical-ish)'''
    # RS (regular spiking)
    pyr_params = dict(a=0.02, b=0.2, c=-65.0, d=8.0, V_m=-65.0, U_m=-13.0, I_e=0.0)
    # FS (fast spiking interneuron proxy)
    basket_params = dict(a=0.1, b=0.2, c=-65.0, d=2.0, V_m=-65.0, U_m=-13.0, I_e=0.0)
    # LTS-ish (slow / adapting interneuron proxy; used here as OLM-like)
    olm_params = dict(a=0.02, b=0.25, c=-65.0, d=2.0, V_m=-65.0, U_m=-16.25, I_e=0.0)

    PYR = nest.Create("izhikevich", N_pyr, params=pyr_params)
    BASKET = nest.Create("izhikevich", N_basket, params=basket_params)
    OLM = nest.Create("izhikevich", N_olm, params=olm_params)

    ''' --- Independent external inputs (generators only) '''
    CA3_to_PYR = nest.Create("poisson_generator", N_pyr, params={"rate": float(rate_ca3_pyr)})
    ECIII_to_PYR = nest.Create("poisson_generator", N_pyr, params={"rate": float(rate_ec_pyr)})
    CA3_to_BA = nest.Create("poisson_generator", N_basket, params={"rate": float(rate_ca3_ba)})

    # --- Synapse weights (IMPORTANT: for izhikevich, weight directly jumps V_m in mV)
    w_ca3_pyr = 3.0
    w_ec_pyr = 2.0
    w_ca3_basket = 3.0

    w_pyr_pyr = 0.8
    w_pyr_basket = 1.5
    w_pyr_olm = 1.2

    w_basket_pyr = -5.0
    w_basket_basket = -4.0
    w_olm_pyr = -3.0

    d_fast = 1.5
    d_slow = 3.0

    # --- Connect inputs one-to-one
    nest.Connect(CA3_to_PYR, PYR, conn_spec="one_to_one", syn_spec={"weight": w_ca3_pyr, "delay": d_fast})
    nest.Connect(ECIII_to_PYR, PYR, conn_spec="one_to_one", syn_spec={"weight": w_ec_pyr, "delay": d_slow})
    nest.Connect(CA3_to_BA, BASKET, conn_spec="one_to_one", syn_spec={"weight": w_ca3_basket, "delay": d_fast})

    # --- Recurrent connectivity
    rng = np.random.default_rng(42)

    bernoulli_connect(PYR, PYR, p_EE, w_pyr_pyr, d_fast, rng)           # PYR->PYR
    bernoulli_connect(PYR, BASKET, p_EI, w_pyr_basket, d_fast, rng)     # PYR->Basket
    bernoulli_connect(PYR, OLM, 0.08, w_pyr_olm, d_slow, rng)           # PYR->OLM

    bernoulli_connect(BASKET, PYR, p_IE, w_basket_pyr, d_fast, rng)     # Basket->PYR
    bernoulli_connect(BASKET, BASKET, 0.10, w_basket_basket, d_fast, rng)

    bernoulli_connect(OLM, PYR, p_OE, w_olm_pyr, d_slow, rng)           # OLM->PYR

    # --- Recorders
    spk_pyr = nest.Create("spike_recorder")
    spk_ba = nest.Create("spike_recorder")
    spk_olm = nest.Create("spike_recorder")
    nest.Connect(PYR, spk_pyr)
    nest.Connect(BASKET, spk_ba)
    nest.Connect(OLM, spk_olm)

    # Input spike recorders (so LFP proxy can include external drive)
    spk_in_ca3_pyr = nest.Create("spike_recorder")
    spk_in_ec_pyr = nest.Create("spike_recorder")
    spk_in_ca3_ba = nest.Create("spike_recorder")
    nest.Connect(CA3_to_PYR, spk_in_ca3_pyr)
    nest.Connect(ECIII_to_PYR, spk_in_ec_pyr)
    nest.Connect(CA3_to_BA, spk_in_ca3_ba)

    # Vm traces (plot per neuron to avoid “diagonal ramps” artifact)
    try:
        vm = nest.Create("multimeter", params={"record_from": ["V_m", "U_m"], "interval": 0.2})
    except Exception:
        vm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 0.2})
    pyr_probe = PYR[:5]
    nest.Connect(vm, pyr_probe)

    # Connectivity stats
    print("\nConnectivity stats:")
    conn_stats("PYR->PYR", PYR, PYR)
    conn_stats("PYR->BA", PYR, BASKET)
    conn_stats("PYR->OLM", PYR, OLM)
    conn_stats("BA->PYR", BASKET, PYR)
    conn_stats("OLM->PYR", OLM, PYR)

    return dict(
        PYR=PYR,
        BASKET=BASKET,
        OLM=OLM,
        spk_pyr=spk_pyr,
        spk_ba=spk_ba,
        spk_olm=spk_olm,
        spk_in_ca3_pyr=spk_in_ca3_pyr,
        spk_in_ec_pyr=spk_in_ec_pyr,
        spk_in_ca3_ba=spk_in_ca3_ba,
        vm=vm,
        pyr_probe=pyr_probe,
    )


def run_report_plot(net, sim_ms=1000.0):
    nest.Simulate(float(sim_ms))

    # Spike counts + mean rates
    ev_p = nest.GetStatus(net["spk_pyr"], "events")[0]
    ev_b = nest.GetStatus(net["spk_ba"], "events")[0]
    ev_o = nest.GetStatus(net["spk_olm"], "events")[0]

    print("\nSpike counts:")
    print("  PYR   :", len(ev_p["times"]))
    print("  BASKET:", len(ev_b["times"]))
    print("  OLM   :", len(ev_o["times"]))

    print("\nMean firing rates (Hz):")
    print(f"  PYR   : {mean_rate(net['PYR'], net['spk_pyr'], sim_ms):.2f}")
    print(f"  BASKET: {mean_rate(net['BASKET'], net['spk_ba'], sim_ms):.2f}")
    print(f"  OLM   : {mean_rate(net['OLM'], net['spk_olm'], sim_ms):.2f}")


    # -------------------------
    # LFP proxy (population-level)
    # -------------------------
    spikes_by_gid = _events_to_spike_dict(
        net["spk_pyr"], net["spk_ba"], net["spk_olm"],
        net["spk_in_ca3_pyr"], net["spk_in_ec_pyr"], net["spk_in_ca3_ba"],
    )

    # PYR LFP proxy: synaptic events into PYR
    t_lfp, lfp_pyr = compute_lfp_proxy(
        spikes_by_gid,
        target_gids=net["PYR"],
        sim_ms=sim_ms,
        dt=0.1,
        tau_ms=5.0,
        normalize_by_targets=True,
    )

    # INH LFP proxy: synaptic events into inhibitory interneurons (Basket + OLM)
    inh_targets = list(net["BASKET"]) + list(net["OLM"])
    _, lfp_inh = compute_lfp_proxy(
        spikes_by_gid,
        target_gids=inh_targets,
        sim_ms=sim_ms,
        dt=0.1,
        tau_ms=5.0,
        normalize_by_targets=True,
    )

    out_path = "ca1_lfp_proxy.npz"
    np.savez(
        out_path,
        t_ms=t_lfp,
        lfp_pyr=lfp_pyr,
        lfp_inh=lfp_inh,
        dt_ms=0.1,
        tau_ms=5.0,
        sim_ms=float(sim_ms),
    )
    print(f"\nSaved LFP proxy traces to: {out_path}")

    # Plots
    import matplotlib.pyplot as plt

    def raster(spk, title):
        ev = nest.GetStatus(spk, "events")[0]
        plt.figure()
        plt.plot(ev["times"], ev["senders"], ".")
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron ID")
        plt.title(title)

    raster(net["spk_pyr"], "CA1 PYR spikes (izhikevich)")
    raster(net["spk_ba"], "CA1 Basket spikes (izhikevich)")
    raster(net["spk_olm"], "CA1 OLM spikes (izhikevich)")


    # LFP proxy plots
    plt.figure()
    plt.plot(t_lfp, lfp_pyr)
    plt.xlabel("Time (ms)")
    plt.ylabel("LFP proxy (a.u.)")
    plt.title("Population LFP proxy: PYR (incoming weighted events)")

    plt.figure()
    plt.plot(t_lfp, lfp_inh)
    plt.xlabel("Time (ms)")
    plt.ylabel("LFP proxy (a.u.)")
    plt.title("Population LFP proxy: INH (Basket+OLM incoming weighted events)")

    # Vm trace per neuron (critical to avoid fake diagonals)
    ev = nest.GetStatus(net["vm"], "events")[0]
    times = np.array(ev["times"])
    senders = np.array(ev["senders"])
    V = np.array(ev["V_m"])

    plt.figure()
    for gid in np.unique(senders):
        m = (senders == gid)
        idx = np.argsort(times[m])
        plt.plot(times[m][idx], V[m][idx])
    plt.xlabel("Time (ms)")
    plt.ylabel("V_m (mV)")
    plt.title("PYR membrane traces (per neuron)")
    plt.show()


if __name__ == "__main__":
    net = build_ca1_izh(
        N_pyr=200,
        N_basket=40,
        N_olm=30,
        p_EE=0.02,
        p_EI=0.10,
        p_IE=0.15,
        p_OE=0.10,
        rate_ca3_pyr=2000.0,
        rate_ec_pyr=900.0,
        rate_ca3_ba=1200.0,
    )
    run_report_plot(net, sim_ms=1000.0)

"""
Fast tuning cheats (because biology is rude):
- Too quiet (no spikes): increase rate_ca3_pyr and/or w_ca3_pyr (e.g., 3.0 -> 4.0).
- PYR too hot: make w_basket_pyr more negative (e.g., -5.0 -> -7.0) or increase p_IE.
- Basket silent but PYR active: increase rate_ca3_ba or w_ca3_basket.
"""