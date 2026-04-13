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
    # Fig 2 — CA3 SUP sequence heatmap — zoomed SWR windows
    # Layout:
    #   Row 0: full 7000 ms overview  [5 ms bins]  — context
    #   Row 1: epoch 1 SWR-1 forward  [2 ms bins]  — zoom + ρ scatter
    #   Row 2: epoch 1 SWR-2 reverse  [2 ms bins]  — zoom + ρ scatter
    #
    # The 5 ms bins in the overview are narrow enough to show individual
    # SWR bursts as vertical columns; the 2 ms zoom panels resolve the
    # inter-group temporal offset (~3-8 ms per step) that constitutes replay.
    # ------------------------------------------------------------------

    cmap_seq2    = plt.cm.viridis
    group_colors2 = [cmap_seq2(k / max(n_groups - 1, 1)) for k in range(n_groups)]

    def _zoom_heatmap(ax_heat, ax_scatter, t_spk, s_spk, grp_ids,
                      t0, t1, pad_ms=20.0, bin_ms=2.0,
                      title_heat="", title_scat="", expected_sign=+1,
                      baseline_rate_per_group=None):
        """
        Left panel : baseline-subtracted 2ms-bin heatmap inside [t0-pad, t1+pad].
          Showing *excess* firing rate (rate − baseline) with a diverging colormap
          makes the sequential diagonal visible even against background theta activity.
        Right panel: per-group mean spike time scatter with Spearman ρ annotation.
        Returns (im, rho, pval).
        """
        tw0, tw1 = t0 - pad_ms, t1 + pad_ms
        edges_z  = np.arange(tw0, tw1 + bin_ms, bin_ms)
        n_bins_z = len(edges_z) - 1
        ng       = len(grp_ids)

        mask  = (t_spk >= tw0) & (t_spk <= tw1)
        t_w   = t_spk[mask];  s_w = s_spk[mask]

        heat_z  = np.zeros((ng, n_bins_z), dtype=np.float32)
        gidx_sc, gmean_sc = [], []
        gs_n = grp_ids.shape[1]

        for k in range(ng):
            m   = np.isin(s_w, grp_ids[k])
            t_g = t_w[m]
            cnt, _ = np.histogram(t_g, bins=edges_z)
            heat_z[k] = cnt / (bin_ms / 1e3) / max(gs_n, 1)
            # Only count spikes within the actual SWR window for mean time
            m_win = m & (t_w >= t0) & (t_w <= t1)
            t_g_win = t_w[m_win]
            if len(t_g_win) >= 3:
                gidx_sc.append(k)
                gmean_sc.append(float(t_g_win.mean()))

        # Subtract per-group baseline (background theta rate) → excess activity
        if baseline_rate_per_group is not None:
            heat_excess = heat_z - baseline_rate_per_group[:, None]
        else:
            heat_excess = heat_z - heat_z[:, :int(pad_ms/bin_ms)].mean(axis=1, keepdims=True)

        vmax = max(abs(heat_excess).max(), 1.0)

        # ---- heatmap: diverging colormap, excess firing ----
        im = ax_heat.imshow(
            heat_excess, aspect="auto", origin="lower",
            extent=[edges_z[0], edges_z[-1], -0.5, ng - 0.5],
            cmap="RdBu_r", interpolation="nearest",
            vmin=-vmax * 0.5, vmax=vmax)
        ax_heat.axvline(t0, color="gold", lw=1.5, ls="--", alpha=0.9, label="SWR start")
        ax_heat.axvline(t1, color="gold", lw=1.5, ls="--", alpha=0.9, label="SWR end")
        ax_heat.set_xlim(edges_z[0], edges_z[-1])
        ax_heat.set_xlabel("Time (ms)", fontsize=8)
        ax_heat.set_ylabel("Seq group #", fontsize=8)
        ax_heat.set_title(title_heat, fontsize=9, loc="left")
        ax_heat.tick_params(labelsize=7)

        # Scaffold expected timing lines (group k starts at t0 + k × step_ms)
        n_steps = ng
        win_dur  = t1 - t0
        step_est = win_dur * 0.85 / n_steps  # matches auto-compute in simulation
        fwd_times = [t0 + k * step_est for k in range(ng)]
        rev_times = [t0 + (ng - 1 - k) * step_est for k in range(ng)]
        exp_times = fwd_times if expected_sign > 0 else rev_times
        ax_heat.plot(exp_times, range(ng), ":", color="yellow", lw=1.2,
                     alpha=0.7, label="Expected scaffold")
        ax_heat.legend(fontsize=6, loc="upper right")

        rho, pval = np.nan, np.nan

        # ---- scatter + Spearman ----
        if len(gidx_sc) >= 5:
            gi  = np.array(gidx_sc);  gm = np.array(gmean_sc)

            # Compute Spearman safely — handle scipy version differences
            try:
                from scipy.stats import spearmanr
                result = spearmanr(gi, gm)
                # scipy >= 1.9 returns SpearmanrResult; earlier returns tuple
                rho  = float(getattr(result, "statistic",
                             getattr(result, "correlation", result[0])))
                pval = float(getattr(result, "pvalue", result[1]))
            except Exception as exc:
                print(f"    [warn] spearmanr failed: {exc}")

            # Linear fit for slope line
            try:
                from scipy.stats import linregress
                slope, intercept, _, _, _ = linregress(gm, gi)
                tl = np.linspace(gm.min() - 2, gm.max() + 2, 200)
                color_fit = "tomato" if expected_sign < 0 else "royalblue"
                ax_scatter.plot(tl, slope * tl + intercept, "--",
                                color=color_fit, lw=1.8, alpha=0.85)
            except Exception:
                pass

            colors_sc = [group_colors2[k] for k in gi]
            ax_scatter.scatter(gm, gi, c=colors_sc, s=55, edgecolors="k",
                               linewidths=0.4, zorder=4)
            # Overlay mean times on heatmap as small white crosses
            ax_heat.scatter(gm, gi, c="white", s=20, marker="+",
                            linewidths=0.8, zorder=5)

            if not np.isnan(rho):
                sign_ok = (expected_sign > 0 and rho > 0.5) or                           (expected_sign < 0 and rho < -0.5)
                verdict = "✓ PASS" if sign_ok else                           ("WEAK" if abs(rho) > 0.2 else "✗ FAIL")
            else:
                verdict = "? NaN"
            ax_scatter.set_title(
                f"{title_scat}\nρ = {rho:+.3f}  p = {pval:.3f}  [{verdict}]"
                if not np.isnan(rho) else f"{title_scat}\nρ = nan  [? NaN — check scipy]",
                fontsize=9, loc="left")
        else:
            ax_scatter.set_title(
                f"{title_scat}\n({len(gidx_sc)} groups with spikes — need ≥5)",
                fontsize=9, loc="left")

        ax_scatter.axvspan(t0, t1, color="gold", alpha=0.18, label="SWR window")
        ax_scatter.set_xlim(edges_z[0], edges_z[-1])
        ax_scatter.set_ylim(-1, ng)
        ax_scatter.set_yticks(range(0, ng, max(1, ng // 8)))
        ax_scatter.set_xlabel("Mean spike time (ms)", fontsize=8)
        ax_scatter.set_ylabel("Seq group #", fontsize=8)
        ax_scatter.tick_params(labelsize=7)
        ax_scatter.legend(fontsize=7, loc="upper right")
        return im, rho, pval


    # Compute per-group baseline firing rate from non-SWR periods
    # (used for background subtraction in the diverging heatmap)
    def _compute_baseline(t_spk, s_spk, grp_ids, sim_ms,
                         swr_fwd, swr_rev, n_epochs=7):
        """Mean firing rate per group during non-SWR periods."""
        ng   = len(grp_ids)
        gs_n = grp_ids.shape[1]
        # Build boolean mask of non-SWR time
        non_swr = np.ones(len(t_spk), dtype=bool)
        for ep in range(n_epochs):
            for ws, we in [(swr_fwd[0]+ep*1000, swr_fwd[1]+ep*1000),
                           (swr_rev[0]+ep*1000, swr_rev[1]+ep*1000)]:
                non_swr &= ~((t_spk >= ws - 20) & (t_spk <= we + 20))
        t_non = t_spk[non_swr];  s_non = s_spk[non_swr]
        # Total non-SWR duration (approximate)
        swr_ms = n_epochs * 2 * (swr_fwd[1]-swr_fwd[0] + 40)
        baseline_s = max((sim_ms - swr_ms) / 1000.0, 1.0)
        baseline = np.zeros(ng, dtype=np.float32)
        for k in range(ng):
            m = np.isin(s_non, grp_ids[k])
            baseline[k] = t_non[m].size / (baseline_s * max(gs_n, 1))
        return baseline

    # Re-load raw spikes from HDF5 for the 2 ms recomputation
    with h5py.File(args.inp, "r") as h5:
        t_sup2   = np.array(h5["ca3_sup"]["spk_times"],   dtype=np.float64)
        s_sup2   = np.array(h5["ca3_sup"]["spk_senders"], dtype=np.int64)
        grp_sup2 = np.array(h5["ca3_sup"]["group_ids"],   dtype=np.int64)  # [n_groups, gs]

    # Build 5ms overview heatmap from raw spikes
    ov_bin  = 5.0
    ov_edge = np.arange(0.0, sim_ms + ov_bin, ov_bin)
    ov_nbin = len(ov_edge) - 1
    gs_sup2   = grp_sup2.shape[1]
    baseline2 = _compute_baseline(t_sup2, s_sup2, grp_sup2, sim_ms,
                                  swr_fwd, swr_rev, n_epochs=7)
    heat_ov = np.zeros((n_groups, ov_nbin), dtype=np.float32)
    for k in range(n_groups):
        m = np.isin(s_sup2, grp_sup2[k])
        cnt, _ = np.histogram(t_sup2[m], bins=ov_edge)
        heat_ov[k] = cnt / (ov_bin / 1e3) / max(gs_sup2, 1)

    # Build figure with gridspec
    # Row heights: overview 1, fwd zoom pair 2, rev zoom pair 2
    fig2 = plt.figure(figsize=(18, 14))
    fig2.suptitle(
        f"CA3 SUP Sequence Replay — Zoomed SWR Windows  [{scale}]",
        fontsize=13, fontweight="bold")
    gs2 = fig2.add_gridspec(3, 4, height_ratios=[1, 2, 2],
                             hspace=0.42, wspace=0.32)

    # ---- Row 0: full overview (span all 4 cols) ----
    ax_ov2 = fig2.add_subplot(gs2[0, :])
    im_ov  = ax_ov2.imshow(
        heat_ov, aspect="auto", origin="lower",
        extent=[0, sim_ms, -0.5, n_groups - 0.5],
        cmap="inferno", interpolation="nearest")
    fig2.colorbar(im_ov, ax=ax_ov2, fraction=0.015).set_label("Rate (Hz)", fontsize=8)
    # Mark every SWR window across all epochs
    for ep in range(7):
        ax_ov2.axvspan(swr_fwd[0] + ep * 1000, swr_fwd[1] + ep * 1000,
                       color="white", alpha=0.22)
        ax_ov2.axvspan(swr_rev[0] + ep * 1000, swr_rev[1] + ep * 1000,
                       color="cyan",  alpha=0.15)
    ax_ov2.set_xlabel("Time (ms)", fontsize=9)
    ax_ov2.set_ylabel("Seq group #", fontsize=9)
    ax_ov2.set_title(
        "A  Full simulation — CA3 SUP  [5 ms bins]  "
        "│  white = SWR-1 fwd windows   cyan = SWR-2 rev windows",
        fontsize=9, loc="left")
    sm2 = ScalarMappable(cmap=cmap_seq2, norm=Normalize(0, n_groups - 1))
    sm2.set_array([])

    # ---- Rows 1-2: zoom panels — heatmap (left pair) + scatter (right pair) ----
    # Epoch 1 forward
    ax_h1f = fig2.add_subplot(gs2[1, 0:2])
    ax_s1f = fig2.add_subplot(gs2[1, 2:4])
    im1f, rho1f, pval1f = _zoom_heatmap(
        ax_h1f, ax_s1f,
        t_sup2, s_sup2, grp_sup2,
        swr_fwd[0], swr_fwd[1], pad_ms=15.0, bin_ms=2.0,
        title_heat="B  SWR-1 forward  [2 ms bins, epoch 1]  ←  group 0 seed",
        title_scat="C  Per-group mean time — SWR-1 fwd\n   (positive slope = ✓ forward order)",
        expected_sign=+1, baseline_rate_per_group=baseline2)
    fig2.colorbar(im1f, ax=ax_h1f, fraction=0.02).set_label("Hz", fontsize=7)

    # Epoch 1 reverse
    ax_h1r = fig2.add_subplot(gs2[2, 0:2])
    ax_s1r = fig2.add_subplot(gs2[2, 2:4])
    im1r, rho1r, pval1r = _zoom_heatmap(
        ax_h1r, ax_s1r,
        t_sup2, s_sup2, grp_sup2,
        swr_rev[0], swr_rev[1], pad_ms=15.0, bin_ms=2.0,
        title_heat="D  SWR-2 reverse  [2 ms bins, epoch 1]  ←  group N-1 seed",
        title_scat="E  Per-group mean time — SWR-2 rev\n   (negative slope = ✓ reverse order)",
        expected_sign=-1, baseline_rate_per_group=baseline2)
    fig2.colorbar(im1r, ax=ax_h1r, fraction=0.02).set_label("Hz", fontsize=7)

    # Add colorbar legend for group colours on the scatter panels
    for ax_s in (ax_s1f, ax_s1r):
        cbar_s = fig2.colorbar(
            ScalarMappable(cmap=cmap_seq2, norm=Normalize(0, n_groups - 1)),
            ax=ax_s, fraction=0.03, pad=0.02)
        cbar_s.set_label("Seq group #", fontsize=7)
        cbar_s.ax.tick_params(labelsize=6)

    fig2.subplots_adjust(top=0.93)
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
    # Figs 6-8 — STC consolidation figures (only when /stc group present)
    # ------------------------------------------------------------------
    with h5py.File(args.inp, "r") as h5:
        stc_present = "stc" in h5
        if stc_present:
            s = h5["stc"]
            stc_event      = np.array(s["event"],        dtype=np.int32)
            stc_t_start    = np.array(s["t_swr_start"],  dtype=np.float32)
            stc_n_tagged   = np.array(s["n_tagged_syn"], dtype=np.int32)   if "n_tagged_syn" in s else np.array(s["n_active_syn"], dtype=np.int32)
            stc_n_ec_fired = np.array(s["n_ec_fired"],   dtype=np.int32)   if "n_ec_fired"   in s else np.zeros_like(stc_event)
            stc_prp_mean   = np.array(s["prp_mean"],     dtype=np.float32) if "prp_mean"     in s else np.zeros_like(stc_event, dtype=np.float32)
            stc_prp_max    = np.array(s["prp_max"],      dtype=np.float32) if "prp_max"      in s else np.zeros_like(stc_event, dtype=np.float32)
            stc_n_ltp_new  = np.array(s["n_ltp_new"],   dtype=np.int32)
            stc_n_ltp_tot  = np.array(s["n_ltp_total"], dtype=np.int32)
            stc_w_mean     = np.array(s["w_mean"],       dtype=np.float32)
            stc_w_ltp_mean = np.array(s["w_ltp_mean"],  dtype=np.float32)
            stc_w_final    = np.array(s["w_final"],      dtype=np.float32)
            stc_ltp_mask   = np.array(s["ltp_mask"],     dtype=bool)
            stc_w_init     = float(s.attrs.get("w_init", 1.0))
            stc_n_syn      = int(s.attrs.get("n_synapses", len(stc_w_final)))
            stc_n_ec       = int(s.attrs.get("n_ec_neurons", 1))
            # Optional per-synapse arrays
            stc_post_idx   = np.array(s["post_idx"],       dtype=np.int32)  if "post_idx"       in s else None
            stc_prp_pool   = np.array(s["prp_pool_final"], dtype=np.float32) if "prp_pool_final" in s else None
            stc_tag_final  = np.array(s["tag_final"],      dtype=np.float32) if "tag_final"      in s else None

    if stc_present:
        # ---------------------------------------------------------------
        # Fig 6 — Consolidation curve
        # Mean CA1→EC weight + L-LTP fraction vs SWR event number
        # ---------------------------------------------------------------
        fig6, (ax6a, ax6b, ax6c, ax6d) = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
        fig6.suptitle(f"STC Consolidation Curve  [{scale}]",
                      fontsize=12, fontweight="bold")

        ev = stc_event

        ax6a.plot(ev, stc_w_mean, "o-", color="steelblue",  lw=1.5, ms=5, label="All synapses")
        ax6a.plot(ev, np.where(np.isfinite(stc_w_ltp_mean), stc_w_ltp_mean, np.nan),
                  "s--", color="darkorange", lw=1.5, ms=5, label="L-LTP synapses only")
        ax6a.axhline(stc_w_init, color="gray", lw=1, ls=":", label=f"w_init = {stc_w_init:.2f}")
        ax6a.set_ylabel("Mean CA1→EC weight", fontsize=9)
        ax6a.set_title("A  Consolidation curve — mean synaptic weight", fontsize=9, loc="left")
        ax6a.legend(fontsize=7)
        ax6a.grid(alpha=0.3)

        ltp_frac = stc_n_ltp_tot / max(stc_n_syn, 1) * 100.0
        ax6b.bar(ev, ltp_frac, color="darkorange", alpha=0.75, label="L-LTP fraction")
        ax6b.plot(ev, stc_n_ltp_new / max(stc_n_syn, 1) * 100, "^-",
                  color="firebrick", ms=5, lw=1, label="New captures this event")
        ax6b.set_ylabel("% of CA1→EC synapses", fontsize=9)
        ax6b.set_title("B  Cumulative L-LTP fraction per SWR event", fontsize=9, loc="left")
        ax6b.legend(fontsize=7)
        ax6b.grid(alpha=0.3)

        ax6c.plot(ev, stc_prp_mean, "o-", color="mediumseagreen", lw=1.5, ms=5,
                  label="Mean PRP pool (SWR events)")
        ax6c.plot(ev, stc_prp_max,  "v--", color="olive", lw=1, ms=4, alpha=0.7,
                  label="Max PRP pool")
        ax6c.plot(ev, stc_n_tagged / max(stc_n_syn, 1) * 100, "s-",
                  color="slateblue", lw=1, ms=4, alpha=0.8,
                  label="Tagged syn %")
        ax6c.set_xlabel("SWR event #", fontsize=9)
        ax6c.set_ylabel("PRP pool / Tag occupancy", fontsize=9)
        ax6c.set_title("C  PRP pool accumulation + tag occupancy per event", fontsize=9, loc="left")
        ax6c.legend(fontsize=7)
        ax6c.grid(alpha=0.3)

        # Panel D: Structural plasticity (3rd timescale)
        ax6d.plot(ev, stc_n_struct_tot / max(stc_n_syn, 1) * 100,
                  "D-", color="darkgreen", lw=1.5, ms=5,
                  label="Structural (spine-enlarged) %")
        ax6d.plot(ev, stc_n_struct_new / max(stc_n_syn, 1) * 100,
                  "^-", color="limegreen", lw=1, ms=4, alpha=0.8,
                  label="New structural captures this event")
        ax6d.set_xlabel("SWR event #", fontsize=9)
        ax6d.set_ylabel("% of CA1→EC synapses", fontsize=9)
        ax6d.set_title(
            "D  Structural plasticity (3rd timescale: spine enlargement)\n"
            "   Analogy: AMPA-receptor clustering + spine volume increase\n"
            "   (biology: days–weeks; requires repeated L-LTP induction)",
            fontsize=9, loc="left")
        ax6d.legend(fontsize=7)
        ax6d.grid(alpha=0.3)
        ax6d.set_ylim(-2, 105)

        fig6.tight_layout()
        _maybe_save(fig6, f"{args.save_prefix}_fig6_stc_consolidation.png" if args.save_prefix else None)

        # ---------------------------------------------------------------
        # Fig 7 — Tag occupancy map
        # % tagged synapses per EC neuron (final state) + PRP heatmap
        # ---------------------------------------------------------------
        fig7, axes7 = plt.subplots(1, 3, figsize=(16, 5))
        fig7.suptitle(f"Tag Occupancy Map — final state  [{scale}]",
                      fontsize=12, fontweight="bold")

        # Panel A: final weight distribution
        ax = axes7[0]
        ax.hist(stc_w_final[~stc_ltp_mask], bins=50, color="steelblue",
                alpha=0.75, label="E-LTP / untagged", density=True)
        ax.hist(stc_w_final[stc_ltp_mask],  bins=50, color="darkorange",
                alpha=0.75, label="L-LTP captured", density=True)
        ax.axvline(stc_w_init, color="gray", ls=":", lw=1.5, label=f"w_init")
        ax.set_xlabel("Synaptic weight", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title("A  Final weight distribution", fontsize=9, loc="left")
        ax.legend(fontsize=7)

        # Panel B: per-EC-neuron tag occupancy (if post_idx available)
        ax = axes7[1]
        if stc_post_idx is not None and stc_tag_final is not None:
            tag_occ = np.zeros(stc_n_ec, dtype=np.float32)
            np.add.at(tag_occ, stc_post_idx, (stc_tag_final > 1e-4).astype(np.float32))
            # Normalise: how many synapses each EC neuron receives
            syn_per_ec = np.bincount(stc_post_idx, minlength=stc_n_ec).astype(np.float32)
            tag_occ_pct = np.where(syn_per_ec > 0, tag_occ / syn_per_ec * 100, 0)
            ax.hist(tag_occ_pct, bins=50, color="slateblue", alpha=0.8, density=False)
            ax.set_xlabel("% tagged input synapses", fontsize=9)
            ax.set_ylabel("# EC neurons", fontsize=9)
            ax.set_title("B  Tag occupancy per EC neuron (final)", fontsize=9, loc="left")
        else:
            ax.text(0.5, 0.5, "post_idx not\nin this HDF5\n(re-run with v6+)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)

        # Panel C: per-EC-neuron PRP pool (if available)
        ax = axes7[2]
        if stc_prp_pool is not None:
            ax.hist(stc_prp_pool, bins=40, color="mediumseagreen", alpha=0.8, density=False)
            ax.set_xlabel("PRP pool (SWR events fired)", fontsize=9)
            ax.set_ylabel("# EC neurons", fontsize=9)
            ax.set_title("C  PRP pool distribution (final)", fontsize=9, loc="left")
        else:
            ax.text(0.5, 0.5, "prp_pool not\nin this HDF5\n(re-run with v6+)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)

        fig7.tight_layout()
        _maybe_save(fig7, f"{args.save_prefix}_fig7_tag_occupancy.png" if args.save_prefix else None)

        # ---------------------------------------------------------------
        # Fig 8 — Consolidation index vs replay quality
        # Fraction L-LTP synapses per event vs Spearman ρ (from /stats)
        # ---------------------------------------------------------------
        with h5py.File(args.inp, "r") as h5:
            rho_fwd  = float(h5["stats"].attrs.get("rho_fwd",  float("nan")))
            rho_rev  = float(h5["stats"].attrs.get("rho_rev",  float("nan")))

        fig8, axes8 = plt.subplots(1, 2, figsize=(12, 4))
        fig8.suptitle(f"Consolidation Index  [{scale}]", fontsize=12, fontweight="bold")

        ax = axes8[0]
        ltp_frac_final = stc_n_ltp_tot / max(stc_n_syn, 1) * 100
        ax.plot(ev, ltp_frac_final, "o-", color="darkorange", lw=1.5, ms=5)
        # Mark where L-LTP first appears
        first_ltp = np.argmax(stc_n_ltp_tot > 0) if stc_n_ltp_tot.max() > 0 else None
        if first_ltp is not None:
            ax.axvline(ev[first_ltp], color="firebrick", ls="--", lw=1.2, alpha=0.8,
                       label=f"First L-LTP @ event {ev[first_ltp]}")
            ax.legend(fontsize=7)
        ax.set_xlabel("SWR event #", fontsize=9)
        ax.set_ylabel("% L-LTP synapses", fontsize=9)
        ax.set_title("A  Consolidation build-up over SWR events", fontsize=9, loc="left")
        ax.grid(alpha=0.3)

        ax = axes8[1]
        # Weight distribution at end: L-LTP vs E-LTP mean
        means  = [stc_w_final[~stc_ltp_mask].mean() if (~stc_ltp_mask).any() else float("nan"),
                  stc_w_final[stc_ltp_mask].mean()  if stc_ltp_mask.any()    else float("nan")]
        labels = ["E-LTP\n(untagged/\ndecaying)", "L-LTP\n(consolidated)"]
        colors = ["steelblue", "darkorange"]
        bars   = ax.bar(labels, means, color=colors, alpha=0.8, width=0.5)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_ylabel("Mean CA1→EC weight", fontsize=9)
        ax.set_title(
            f"B  E-LTP vs L-LTP weight means\n"
            f"Replay quality: ρ_fwd={rho_fwd:+.3f}  ρ_rev={rho_rev:+.3f}",
            fontsize=9, loc="left")
        ax.set_ylim(0, max(filter(np.isfinite, means + [stc_w_init * 2])) * 1.3 if means else 2)
        ax.axhline(stc_w_init, color="gray", ls=":", lw=1.2, label=f"w_init")
        ax.legend(fontsize=7)

        fig8.tight_layout()
        _maybe_save(fig8, f"{args.save_prefix}_fig8_consolidation_index.png" if args.save_prefix else None)

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
