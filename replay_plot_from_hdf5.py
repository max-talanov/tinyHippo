#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replay_plot_from_hdf5.py
========================
Read the HDF5 produced by replay_scaled.py (--out-hdf5 flag) and regenerate
all figures *locally*, without needing NEST installed.

Examples
--------
  # Basic: show plots interactively
  python replay_plot_from_hdf5.py --in replay_output_1pct/replay_1pct.h5

  # Save PNGs silently (typical post-MN5 workflow)
  python replay_plot_from_hdf5.py --in replay_output_1pct/replay_1pct.h5 \\
         --save-prefix figures/run1

  # Limit raster to first 1000 ms and use finer rate bins
  python replay_plot_from_hdf5.py --in replay_1pct.h5 \\
         --save-prefix figs/run1 --rate-bin-ms 5
"""

import argparse
import sys
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binned_rate(spk_times, n_cells, t_stop, bin_ms):
    edges     = np.arange(0.0, t_stop + bin_ms, bin_ms)
    counts, _ = np.histogram(spk_times, bins=edges)
    centres   = edges[:-1] + bin_ms / 2.0
    rate      = counts / (bin_ms / 1e3) / max(int(n_cells), 1)
    return centres, rate


def _shade(ax, swr_fwd, swr_rev, alpha=0.18):
    ax.axvspan(*swr_fwd, color="steelblue", alpha=alpha, label="SWR-1 fwd")
    ax.axvspan(*swr_rev, color="tomato",    alpha=alpha, label="SWR-2 rev")


def _maybe_save(fig, path):
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Plot bidirectional replay results from an HDF5 file "
                    "(produced by replay_scaled.py --out-hdf5).")
    ap.add_argument("--in",          dest="inp",        required=True,
                    help="Input HDF5 file from HPC")
    ap.add_argument("--save-prefix", dest="save_prefix", default="",
                    help="Save PNGs as <prefix>_fig1_overview.png etc. "
                         "Parent directories are created automatically.")
    ap.add_argument("--show",        action="store_true",
                    help="Show plots interactively (default: off when saving)")
    ap.add_argument("--rate-bin-ms", type=float, default=10.0,
                    help="Bin size for population-rate panels (ms, default 10)")
    ap.add_argument("--inh-bin-ms",  type=float, default=2.0,
                    help="Bin size for inhibitory-rate panel (ms, default 2)")
    ap.add_argument("--raster-max-spikes", type=int, default=300_000,
                    help="Subsample rasters to this many spikes for speed "
                         "(default 300000; 0 = no limit)")
    args = ap.parse_args()

    # Switch to non-GUI backend only when saving silently
    if args.save_prefix and not args.show:
        matplotlib.use("Agg")

    if args.save_prefix:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.save_prefix)) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Load HDF5
    # ------------------------------------------------------------------
    with h5py.File(args.inp, "r") as h5:
        sim_ms    = float(h5.attrs["sim_ms"])
        n_groups  = int(h5.attrs["n_groups"])
        swr_fwd   = (float(h5.attrs["swr_fwd_start"]), float(h5.attrs["swr_fwd_stop"]))
        swr_rev   = (float(h5.attrs["swr_rev_start"]), float(h5.attrs["swr_rev_stop"]))
        scale     = str(h5.attrs.get("scale", ""))

        print("=== Run metadata ===")
        for k in ["created_utc", "nest_version", "sim_ms", "dt_ms", "scale", "n_groups"]:
            if k in h5.attrs:
                print(f"  {k}: {h5.attrs[k]}")
        if "stats" in h5:
            s = h5["stats"].attrs
            print("--- replay quality ---")
            for lbl, key_r, key_p in [("SWR-1 fwd", "rho_fwd", "pval_fwd"),
                                       ("SWR-2 rev", "rho_rev", "pval_rev")]:
                rho  = s.get(key_r, float("nan"))
                pval = s.get(key_p, float("nan"))
                verdict = ""
                if not np.isnan(rho):
                    verdict = "PASS" if abs(rho) > 0.5 else "WEAK"
                print(f"  {lbl}: rho={rho:+.3f}  p={pval:.3f}  [{verdict}]")
            print("--- mean firing rates ---")
            for pop in ["ca3_sup", "ca3_deep", "ca3_int_sup", "ca3_int_deep",
                        "ca1_pyr", "ca1_basket", "ca1_olm"]:
                key = f"mean_rate_{pop}"
                if key in s:
                    print(f"  {pop}: {s[key]:.2f} Hz")
        print("====================")

        # Spike arrays
        def _spk(grp_name):
            g = h5[grp_name]
            return (np.array(g["spk_times"],   dtype=np.float64),
                    np.array(g["spk_senders"], dtype=np.int64),
                    int(g.attrs["n_cells"]))

        t_sup,  s_sup,  n_sup  = _spk("ca3_sup")
        t_deep, s_deep, n_deep = _spk("ca3_deep")
        t_ints, s_ints, n_ints = _spk("ca3_int_sup")
        t_intd, s_intd, n_intd = _spk("ca3_int_deep")
        t_pyr,  s_pyr,  n_pyr  = _spk("ca1_pyr")
        t_bsk,  s_bsk,  n_bsk  = _spk("ca1_basket")
        _,      _,      n_olm  = _spk("ca1_olm")

        # EC LII/III — optional (present only when --ec-lii was used)
        ec_present = "ec_lii" in h5
        if ec_present:
            t_ec, s_ec, n_ec = _spk("ec_lii")
            ec_attrs = dict(h5["ec_lii"].attrs)
            print(f"  EC LII/III: N={n_ec:,}  K_ca1={ec_attrs.get('K_ca1_ec','?')}  "
                  f"w_init={ec_attrs.get('w_init','?')}")
            if "w_ca1_ec_note" in ec_attrs:
                print(f"  EC weights: {ec_attrs['w_ca1_ec_note']}")
        else:
            t_ec = s_ec = None
            n_ec = 0
            print("  EC LII/III: not present in this HDF5")

        # Group membership arrays  [n_groups × group_size]
        group_ids_sup  = np.array(h5["ca3_sup"]["group_ids"],  dtype=np.int64)
        group_ids_deep = np.array(h5["ca3_deep"]["group_ids"], dtype=np.int64)

        # Pre-computed heatmaps  [n_groups × n_bins]
        heatmap_sup  = np.array(h5["ca3_sup"]["heatmap"],  dtype=np.float32)
        heatmap_deep = np.array(h5["ca3_deep"]["heatmap"], dtype=np.float32)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    bin_ms  = args.rate_bin_ms
    ibin_ms = args.inh_bin_ms
    max_spk = args.raster_max_spikes

    cmap_seq     = plt.cm.viridis
    group_colors = [cmap_seq(k / max(n_groups - 1, 1)) for k in range(n_groups)]

    def _subsample(t, s, max_n):
        """Thin arrays for raster speed without bias."""
        if max_n > 0 and len(t) > max_n:
            idx = np.random.choice(len(t), size=max_n, replace=False)
            idx.sort()
            return t[idx], s[idx]
        return t, s

    t_sup_r,  s_sup_r  = _subsample(t_sup,  s_sup,  max_spk)
    t_deep_r, s_deep_r = _subsample(t_deep, s_deep, max_spk)
    t_pyr_r,  s_pyr_r  = _subsample(t_pyr,  s_pyr,  max_spk)

    # ------------------------------------------------------------------
    # Fig 1 — Overview  (5 or 6 panels depending on EC presence)
    # ------------------------------------------------------------------
    n_panels = 6 if ec_present else 5
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 14 + 2*(n_panels-5)), sharex=True)
    fig.suptitle(f"Bidirectional Replay — Watson et al. 2025  [{scale}]",
                 fontsize=13, fontweight="bold")

    # Panel A: CA3 SUP raster coloured by group
    ax = axes[0]
    for k in range(n_groups):
        grp = group_ids_sup[k]
        m   = np.isin(s_sup_r, grp)
        if m.any():
            ax.scatter(t_sup_r[m], s_sup_r[m], s=1.0,
                       color=group_colors[k], rasterized=True)
    _shade(ax, swr_fwd, swr_rev)
    ax.set_ylabel("CA3 SUP neuron ID", fontsize=9)
    ax.set_title("A  CA3 SUPERFICIAL raster  [colour = seq group]", fontsize=9, loc="left")
    sm = ScalarMappable(cmap=cmap_seq, norm=Normalize(0, n_groups - 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.01).set_label("Group #", fontsize=8)

    # Panel B: CA3 DEEP raster
    ax = axes[1]
    for k in range(n_groups):
        grp = group_ids_deep[k]
        m   = np.isin(s_deep_r, grp)
        if m.any():
            ax.scatter(t_deep_r[m], s_deep_r[m], s=1.5,
                       color=group_colors[k], marker="^", alpha=0.7, rasterized=True)
    _shade(ax, swr_fwd, swr_rev)
    ax.set_ylabel("CA3 DEEP neuron ID", fontsize=9)
    ax.set_title("B  CA3 DEEP raster  [burst-firing, tetrasynaptic output]",
                 fontsize=9, loc="left")

    # Panel C: CA1 PYR raster
    ax = axes[2]
    ax.scatter(t_pyr_r, s_pyr_r, s=0.8, color="slategray", rasterized=True)
    _shade(ax, swr_fwd, swr_rev)
    ax.set_ylabel("CA1 PYR neuron ID", fontsize=9)
    ax.set_title("C  CA1 PYR raster", fontsize=9, loc="left")

    # Panel D: Population rates
    ax = axes[3]
    tc,  rc_sup  = _binned_rate(t_sup,  n_sup,  sim_ms, bin_ms)
    _,   rc_deep = _binned_rate(t_deep, n_deep, sim_ms, bin_ms)
    _,   rc_pyr  = _binned_rate(t_pyr,  n_pyr,  sim_ms, bin_ms)
    ax.plot(tc, rc_sup,  color="darkorange", lw=1.2, label="CA3 SUP")
    ax.plot(tc, rc_deep, color="royalblue",  lw=1.2, label="CA3 DEEP")
    ax.plot(tc, rc_pyr,  color="steelblue",  lw=1.2, alpha=0.7, label="CA1 PYR")
    _shade(ax, swr_fwd, swr_rev)
    ax.legend(fontsize=7, ncol=3)
    ax.set_ylabel("Rate (Hz)", fontsize=9)
    ax.set_title(f"D  Population rates  [{bin_ms:.0f} ms bins]", fontsize=9, loc="left")

    # Panel E: Inhibitory rates
    ax = axes[4]
    tf,  ri_sup  = _binned_rate(t_ints, n_ints, sim_ms, ibin_ms)
    _,   ri_deep = _binned_rate(t_intd, n_intd, sim_ms, ibin_ms)
    _,   rb      = _binned_rate(t_bsk,  n_bsk,  sim_ms, ibin_ms)
    ax.plot(tf, ri_sup,  color="firebrick",    lw=0.8, alpha=0.85, label="CA3 INT_SUP")
    ax.plot(tf, ri_deep, color="salmon",       lw=0.8, alpha=0.85, label="CA3 INT_DEEP")
    ax.plot(tf, rb,      color="mediumorchid", lw=0.8, alpha=0.85, label="CA1 Basket")
    _shade(ax, swr_fwd, swr_rev)
    ax.legend(fontsize=7, ncol=3)
    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_ylabel("Rate (Hz)", fontsize=9)
    ax.set_title(f"E  Inhibitory rates  [{ibin_ms:.0f} ms bins]", fontsize=9, loc="left")

    # Panel F: EC LII/III raster + rate (only when present)
    if ec_present:
        ax = axes[5]
        t_ec_r, s_ec_r = _subsample(t_ec, s_ec, max_spk)
        ax.scatter(t_ec_r, s_ec_r, s=0.6, color="coral", rasterized=True, alpha=0.6)
        _shade(ax, swr_fwd, swr_rev)
        ax.set_ylabel("EC LII/III neuron ID", fontsize=9)
        ax.set_title("F  EC LII/III raster  [cortical consolidation target]",
                     fontsize=9, loc="left")
        # Overlay EC rate as twin axis
        ax2 = ax.twinx()
        _, r_ec = _binned_rate(t_ec, n_ec, sim_ms, bin_ms)
        tc_ec   = np.arange(0.0, sim_ms, bin_ms) + bin_ms / 2.0
        ax2.plot(tc_ec[:len(r_ec)], r_ec, color="darkred", lw=1.0, alpha=0.7,
                 label=f"EC rate ({n_ec:,} neurons)")
        ax2.set_ylabel("Rate (Hz)", fontsize=8, color="darkred")
        ax2.tick_params(axis="y", labelcolor="darkred", labelsize=7)
        ax2.legend(fontsize=7, loc="upper right")

    ax.set_xlabel("Time (ms)", fontsize=9)

    fig.tight_layout()
    _maybe_save(fig, f"{args.save_prefix}_fig1_overview.png" if args.save_prefix else None)

    # ------------------------------------------------------------------
    # Fig 2 — CA3 SUP sequence heatmap
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    im = ax2.imshow(
        heatmap_sup, aspect="auto", origin="lower",
        extent=[0, sim_ms, -0.5, n_groups - 0.5],
        cmap="inferno", interpolation="nearest",
    )
    fig2.colorbar(im, ax=ax2, pad=0.02).set_label("Rate (Hz)", fontsize=9)
    ax2.axvspan(*swr_fwd, color="white", alpha=0.20, label="SWR-1 fwd")
    ax2.axvspan(*swr_rev, color="cyan",  alpha=0.15, label="SWR-2 rev")
    fwd_dur = swr_fwd[1] - swr_fwd[0]
    rev_dur = swr_rev[1] - swr_rev[0]
    ax2.plot([swr_fwd[0], swr_fwd[0] + fwd_dur * 0.75], [0, n_groups - 1],
             "--w", lw=1.8, alpha=0.9, label="Fwd slope")
    ax2.plot([swr_rev[0], swr_rev[0] + rev_dur * 0.75], [n_groups - 1, 0],
             "--c", lw=1.8, alpha=0.9, label="Rev slope")
    ax2.set_xlabel("Time (ms)", fontsize=10)
    ax2.set_ylabel("Sequence group #", fontsize=10)
    ax2.set_title(f"CA3 SUP Sequence Group Heatmap  [{scale}]", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    fig2.tight_layout()
    _maybe_save(fig2, f"{args.save_prefix}_fig2_heatmap.png" if args.save_prefix else None)

    # ------------------------------------------------------------------
    # Fig 3 — CA3 DEEP sequence heatmap
    # ------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    im3 = ax3.imshow(
        heatmap_deep, aspect="auto", origin="lower",
        extent=[0, sim_ms, -0.5, n_groups - 0.5],
        cmap="inferno", interpolation="nearest",
    )
    fig3.colorbar(im3, ax=ax3, pad=0.02).set_label("Rate (Hz)", fontsize=9)
    ax3.axvspan(*swr_fwd, color="white", alpha=0.20, label="SWR-1 fwd")
    ax3.axvspan(*swr_rev, color="cyan",  alpha=0.15, label="SWR-2 rev")
    ax3.set_xlabel("Time (ms)", fontsize=10)
    ax3.set_ylabel("Sequence group #", fontsize=10)
    ax3.set_title(f"CA3 DEEP Sequence Group Heatmap  [{scale}]", fontsize=11)
    ax3.legend(fontsize=8, loc="upper right")
    fig3.tight_layout()
    _maybe_save(fig3, f"{args.save_prefix}_fig3_deep_heatmap.png" if args.save_prefix else None)

    # ------------------------------------------------------------------
    # Fig 4 — Per-population firing-rate summary  (bar chart)
    # ------------------------------------------------------------------
    pop_labels  = ["CA3 SUP", "CA3 DEEP", "CA3 INT_SUP", "CA3 INT_DEEP",
                   "CA1 PYR", "CA1 Basket"]
    pop_spk_t   = [t_sup, t_deep, t_ints, t_intd, t_pyr, t_bsk]
    pop_n_cells = [n_sup, n_deep, n_ints, n_intd, n_pyr, n_bsk]
    bar_colors  = ["darkorange", "royalblue", "firebrick", "salmon",
                   "steelblue", "mediumorchid"]
    if ec_present:
        pop_labels.append("EC LII/III")
        pop_spk_t.append(t_ec)
        pop_n_cells.append(n_ec)
        bar_colors.append("coral")
    mean_rates  = [len(t) / (n * sim_ms / 1e3) if n > 0 else 0
                   for t, n in zip(pop_spk_t, pop_n_cells)]

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    bars = ax4.bar(pop_labels, mean_rates, color=bar_colors)
    ax4.bar_label(bars, fmt="%.1f Hz", padding=3, fontsize=8)
    ax4.set_ylabel("Mean firing rate (Hz)", fontsize=10)
    ax4.set_title(f"Population mean firing rates  [{scale}]", fontsize=11)
    ax4.set_ylim(0, max(mean_rates) * 1.25)
    fig4.tight_layout()
    _maybe_save(fig4, f"{args.save_prefix}_fig4_rates_bar.png" if args.save_prefix else None)

    # ------------------------------------------------------------------
    # Fig 5 — EC LII/III analysis (only when EC present)
    # ------------------------------------------------------------------
    if ec_present:
        pad = 50.0
        fig5, axes5 = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
        fig5.suptitle(f"EC LII/III — Cortical Consolidation Target  [{scale}]",
                      fontsize=12, fontweight="bold")

        # Panel A: full EC raster
        ax = axes5[0]
        t_ec_r2, s_ec_r2 = _subsample(t_ec, s_ec, max_spk)
        ax.scatter(t_ec_r2, s_ec_r2, s=0.5, color="coral", rasterized=True, alpha=0.5)
        _shade(ax, swr_fwd, swr_rev)
        ax.set_xlim(0, sim_ms)
        ax.set_ylabel("EC LII/III neuron ID", fontsize=9)
        ax.set_title("A  Full EC LII/III raster", fontsize=9, loc="left")

        # Panel B: EC population rate with SWR windows marked
        ax = axes5[1]
        tc_ec, rc_ec = _binned_rate(t_ec, n_ec, sim_ms, bin_ms)
        ax.plot(tc_ec, rc_ec, color="coral", lw=1.2, label="EC LII/III")
        _, rc_ca1 = _binned_rate(t_pyr, n_pyr, sim_ms, bin_ms)
        ax.plot(tc_ec[:len(rc_ca1)], rc_ca1, color="steelblue", lw=0.8,
                alpha=0.6, label="CA1 PYR")
        _shade(ax, swr_fwd, swr_rev)
        ax.legend(fontsize=8)
        ax.set_xlim(0, sim_ms)
        ax.set_ylabel("Rate (Hz)", fontsize=9)
        ax.set_title(f"B  EC vs CA1 population rates  [{bin_ms:.0f} ms bins]",
                     fontsize=9, loc="left")

        # Panel C: zoom on SWR-1 fwd window — EC vs CA1 overlap
        ax = axes5[2]
        t0z, t1z = swr_fwd[0] - pad, swr_fwd[1] + pad
        m_ec  = (t_ec  >= t0z) & (t_ec  <= t1z)
        m_ca1 = (t_pyr >= t0z) & (t_pyr <= t1z)
        ax.scatter(t_ec[m_ec],   s_ec[m_ec],   s=2.0, color="coral",
                   alpha=0.5, label=f"EC ({m_ec.sum():,} spikes)", rasterized=True)
        ax.scatter(t_pyr[m_ca1], s_pyr[m_ca1], s=0.8, color="steelblue",
                   alpha=0.3, label=f"CA1 PYR ({m_ca1.sum():,} spikes)", rasterized=True)
        ax.axvspan(*swr_fwd, color="steelblue", alpha=0.15, label="SWR-1 window")
        ax.set_xlim(t0z, t1z)
        ax.legend(fontsize=7, ncol=3)
        ax.set_xlabel("Time (ms)", fontsize=9)
        ax.set_ylabel("Neuron ID", fontsize=9)
        ax.set_title("C  Zoom: SWR-1 forward window — EC response vs CA1 drive",
                     fontsize=9, loc="left")

        fig5.tight_layout()
        _maybe_save(fig5, f"{args.save_prefix}_fig5_ec_analysis.png" if args.save_prefix else None)

    # ------------------------------------------------------------------
    # Show / done
    # ------------------------------------------------------------------
    if args.show or not args.save_prefix:
        plt.show()
    else:
        plt.close("all")

    print(">>> Done.")


if __name__ == "__main__":
    main()
