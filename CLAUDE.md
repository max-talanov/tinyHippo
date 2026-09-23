# tinyHippo

## Project goal

Build a **bio-plausible spiking model of the rat hippocampus** (NEST, Izhikevich
neurons) that shows **memory consolidation through bidirectional replay and
sharp-wave ripples (SWRs)** across the full entorhinal–hippocampal loop:

```
EC LII → DG → CA3 → CA1 → EC LII/LV → mPFC
   ↑                              │
   └──────────────────────────────┘
```

The same code path scales from a 1% test network (~8k neurons, laptop) to the
full rat hippocampus (~780k neurons, MareNostrum 5).

The core scientific claims the model has to support:

1. **Replay and consolidation are separable mechanisms.** Forward and reverse
   replay in CA3 during SWRs drive synaptic tagging and capture (STC) on
   CA1→EC. Blocking late-LTP capture (`--prp-threshold 999`) must leave replay
   quality unchanged while cortical consolidation goes to zero (the Phase 5
   falsification).
2. **Each stage does its circuit job.** DG does pattern separation (target
   2–4% of granule cells active per pattern), and CA3 does pattern completion
   through its recurrent collaterals (Watson et al. 2025 wiring).
3. **Consolidated cortical traces are real engrams** by the Josselyn & Tonegawa
   (2020) criteria: sparse, persistent, **specific**, sufficient and necessary.

A companion goal is hardware. The model's validated timescales are turned into a
memristive neuromorphic circuit spec in
[hippocampal_timescales_as_circuit_spec.md](hippocampal_timescales_as_circuit_spec.md).

### Where things stand

Claims 1 and 2 are validated at 12% scale. The open problem is **engram
specificity** (claim 3). Cortical traces are sparse and persistent, but pattern
identity does not yet carry from the hippocampus to cortex. Current work fixes
the convergent, uniformly random projections one hop at a time with
topographic or group clustering: EC LII→DG (`--dg-ec-cluster-sigma`), then
CA3→CA1 Schaffer (`--schaffer-group-frac`). The next candidate is CA1→EC LII.
[RESULTS.md](RESULTS.md) has the up-to-date record; its "Open items" section is
the current to-do list.

## Key files

- [replay_scaled.py](replay_scaled.py): the main simulation. Holds every
  population and capability, plus the HDF5 export.
- [tiny.py](tiny.py): shared helpers for seeding and for the theta and SWR
  generators.
- [run.sh](run.sh): SLURM launcher for MN5. Named jobs (JOB A…, H1…H11) are
  documented in its header comments.
- [RESULTS.md](RESULTS.md): the numbered experiment log (§1…). Add a new
  section for each result; don't rewrite older ones.
- [mem_cons_plan.md](mem_cons_plan.md): the phase plan.
  [watson2025_update_report.md](watson2025_update_report.md) covers the CA3
  wiring history.

Capabilities are opt-in flags (`--dg`, `--ec-lii`, `--stc`, `--ec-lv`, `--mpfc`,
`--homeostasis`, `--pattern-completion`, `--n-patterns`). Each flag gates a
self-contained module with a dataclass and a `build_*` function. Follow that
pattern for new modules.

## Running

```bash
python -c "import nest; print(nest.__version__)"   # verify NEST ≥ 3.9 in .venv first
python replay_scaled.py --scale 1 --dg --dg-scale 2 --no-figures
```

- 1% runs are local and take ~10–20 min with the full stack. Anything above
  ~1% runs on MN5 through `sbatch --export=ALL,SCALE=12,... run.sh`.
- Local ad hoc sweeps go to `out/`. MN5 results the user uploads land in
  `res/<date>/`. Neither is committed.

## Methodological rules (learned the hard way)

- **Score replay on the SWR window itself**, `(win[0], win[1])`. Never pad past
  the window. A +30 ms tail picks up a forward rebound burst that cancels
  reverse-replay ρ.
- **Measure rheobase with injected DC current, not Poisson drive rate.** For
  Izhikevich neurons, `I_rheo = (5-b)²/0.16 - 140 - I_e`.
- **Izhikevich `b` is a bifurcation parameter.** Keep heterogeneity on `b`
  small and below each population's `b_crit`.
- **Testing within-pattern selectivity needs at least 2 epochs of the same
  pattern**, e.g. `--n-patterns 2 --n-swr >= 4`.
- **CA3 SUP and CA1 PYR run near saturation** (90–99% active), which makes the
  Jaccard identity metric useless there. Use rate-based readouts instead
  (per-group z-scores, rate-vector correlation, permutation tests).
- **DG activity is chaotic across time blocks.** Report block-by-block activity,
  not only whole-run means. Don't expect exact reproducibility even at a fixed
  seed.
- Treat single-seed or 1%-only results as pilots until they replicate at 12%
  and across seeds.
