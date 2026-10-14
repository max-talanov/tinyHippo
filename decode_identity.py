#!/usr/bin/env python3
"""Pattern-identity readouts for replay_scaled.py HDF5 output (RESULTS.md SS23-27).

Two tests per run, both on SWR-window spikes (300-420 and 600-720 ms of each
1000 ms block), with significance from shuffling which pattern each block had:

1. Home-group z (populations with home groups: CA3 SUP sequence groups, and cell
   i -> group i % n_groups for CA1 PYR and EC LII). Each block's group-mean
   spike counts are z-scored across groups and averaged over the groups that
   belong to that block's pattern (group g is in pattern g % n_patterns).
2. G1 decoder (the branch-plan gate): leave-one-block-out nearest-centroid
   classification of the block's pattern from per-cell spike counts in 10 ms
   bins inside the SWR windows (rate + timing), and from per-cell counts only.

Cell positions come from each population's `gid_first` attribute. Files written
before that attribute existed need --gid-first POP=GID: the lowest spiking GID is
NOT reliable (an edge cell that never fires shifts every home-group label, see
RESULTS.md SS27).

    python decode_identity.py run1.h5 run2.h5 --gid-first ec_lii=1097544
"""
import argparse

import h5py
import numpy as np

WINS = [(300.0, 420.0), (600.0, 720.0)]
BIN = 10.0
NBINS = int(sum((b - a) / BIN for a, b in WINS))
HOME_POPS = ["ca1_pyr", "ec_lii"]
DECODE_POPS = ["ca1_pyr", "ec_lii", "ec_lv", "mpfc"]


def win_bin(r):
    out = np.full(len(r), -1)
    off = 0
    for a, b in WINS:
        m = (r >= a) & (r < b)
        out[m] = off + ((r[m] - a) // BIN).astype(int)
        off += int((b - a) / BIN)
    return out


def binned(t, s, gid_first, n, n_blocks, epoch_ms=1000.0):
    """(n_blocks, n, NBINS) spike counts; spikes outside [gid_first, +n) dropped."""
    ep = (t // epoch_ms).astype(int)
    wb = win_bin(t - ep * epoch_ms)
    idx = (s - gid_first).astype(int)
    m = (wb >= 0) & (ep < n_blocks) & (idx >= 0) & (idx < n)
    X = np.zeros((n_blocks, n, NBINS), np.float32)
    np.add.at(X, (ep[m], idx[m], wb[m]), 1)
    return X, int((~((idx >= 0) & (idx < n))).sum())


def group_z(M, sched, n_pat):
    z = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-12)
    pog = np.arange(M.shape[1]) % n_pat
    return np.array([z[e, pog == sched[e]].mean() for e in range(len(sched))])


def loo_acc(F, y, n_pat):
    F = F - F.mean(1, keepdims=True)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    hits = 0
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        best, pred = -np.inf, None
        for c in range(n_pat):
            mc = tr & (y == c)
            if not mc.any():
                continue
            cen = F[mc].mean(0)
            cen = cen - cen.mean()
            r = F[i] @ cen / (np.linalg.norm(cen) + 1e-12)
            if r > best:
                best, pred = r, c
        hits += pred == y[i]
    return hits / len(y)


def perm_p(stat_fn, sched, rng, n_perm):
    obs = stat_fn(sched)
    null = np.array([stat_fn(rng.permutation(sched)) for _ in range(n_perm)])
    return obs, (np.sum(null >= obs) + 1) / (n_perm + 1)


def schedule(f, n_blocks, n_pat):
    """Pattern per block. Uses the file's record when present, else cyclic."""
    if "epoch_pattern" in f.attrs:
        return np.asarray(f.attrs["epoch_pattern"])[:n_blocks]
    return np.array([e % n_pat for e in range(n_blocks)])


def analyse(path, gid_override, n_blocks, n_pat, n_perm, seed):
    rng = np.random.default_rng(seed)
    f = h5py.File(path, "r")
    sched = schedule(f, n_blocks, n_pat)
    print(f"== {path}")
    pops = {}
    for key in sorted(set(HOME_POPS + DECODE_POPS + ["ca3_sup"])):
        if key not in f:
            continue
        s = f[f"{key}/spk_senders"][:]
        n = int(f[key].attrs["n_cells"])
        if key in gid_override:
            g0, src = gid_override[key], "--gid-first"
        elif "gid_first" in f[key].attrs:
            g0, src = int(f[key].attrs["gid_first"]), "file"
        elif key == "ca3_sup":
            g0, src = int(f["ca3_sup/group_ids"][:].min()), "group_ids"
        else:
            g0, src = int(s.min()), "lowest spiking GID (UNRELIABLE)"
        X, n_out = binned(f[f"{key}/spk_times"][:], s, g0, n, n_blocks)
        pops[key] = X
        warn = f"  WARNING {n_out} spikes outside range" if n_out else ""
        print(f"   {key:8s} n={n:6d} gid_first={g0} [{src}]{warn}")

    gids = f["ca3_sup/group_ids"][:]
    G = gids.shape[0]
    groups = {"ca3_sup": [gids[g] - gids.min() for g in range(G)]}
    for key in HOME_POPS:
        if key in pops:
            n = pops[key].shape[1]
            groups[key] = [np.flatnonzero(np.arange(n) % G == g) for g in range(G)]
    pog = np.arange(G) % n_pat
    for key, idx in groups.items():
        c = pops[key].sum(2)
        M = np.stack([c[:, ix].mean(1) for ix in idx], 1)
        z = group_z(M, sched, n_pat)
        _, p = perm_p(lambda sc: group_z(M, sc, n_pat).mean(), sched, rng, n_perm)
        on = np.array([M[e, pog == sched[e]].mean() for e in range(n_blocks)])
        off = np.array([M[e, pog != sched[e]].mean() for e in range(n_blocks)])
        print(f"   home-group {key:8s} z={z.mean():+.3f}  positive {int((z > 0).sum())}/{n_blocks}"
              f"  p={p:.4f}  modulation {100 * np.mean(on / off - 1):+.1f}%")
    for key in DECODE_POPS:
        if key not in pops:
            continue
        X = pops[key]
        a1, p1 = perm_p(lambda sc: loo_acc(X.reshape(n_blocks, -1), sc, n_pat), sched, rng, n_perm)
        a2, p2 = perm_p(lambda sc: loo_acc(X.sum(2), sc, n_pat), sched, rng, n_perm)
        print(f"   G1 decode  {key:8s} rate+timing acc {a1:.2f} p={p1:.4f}   counts acc {a2:.2f} p={p2:.4f}"
              f"   (chance {1 / n_pat:.2f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5", nargs="+")
    ap.add_argument("--gid-first", action="append", default=[], metavar="POP=GID",
                    help="override a population's first GID, e.g. ec_lii=1097544")
    ap.add_argument("--n-blocks", type=int, default=14)
    ap.add_argument("--n-patterns", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    override = {k: int(v) for k, v in (x.split("=") for x in a.gid_first)}
    for path in a.h5:
        analyse(path, override, a.n_blocks, a.n_patterns, a.n_perm, a.seed)


if __name__ == "__main__":
    main()
