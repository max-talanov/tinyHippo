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