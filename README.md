# tinyHippo

A bio-plausible spiking model of the rat hippocampus in [NEST](https://nest-simulator.readthedocs.io)
(Izhikevich neurons), built to study **memory consolidation through bidirectional
replay and sharp-wave ripples** across the full entorhinal–hippocampal loop:

```
EC LII → DG → CA3 → CA1 → EC LII/LV → mPFC
   ↑                              │
   └──────────────────────────────┘
```

The model scales from a 1% test network (~8k neurons, runs on a laptop) to the
full rat hippocampus (~780k neurons, MareNostrum 5), with the same code path.

A companion document,
[`hippocampal_timescales_as_circuit_spec.md`](hippocampal_timescales_as_circuit_spec.md),
takes this model's validated timescales and mechanisms (synaptic tagging and
capture, replay-driven consolidation) and specifies them as a memristor-based
neuromorphic circuit — organic volatile devices for the hippocampal tag,
inorganic non-volatile crossbars for the cortical store, with replay as the
handoff between them.

## What the model does

| Capability | Flag | Status |
|---|---|---|
| **Bidirectional replay** — forward & reverse sequence replay during SWRs | *(default)* | validated at 12% (ρ_fwd +0.70, ρ_rev −0.54) |
| **CA3 recurrent feedback loops** — Watson 2025 four-way asymmetric wiring + E↔I | *(always on)* | core |
| **DG pattern separation** — granule sparse coding via basket feedback | `--dg` | validated at 12% (2.2% active/pattern) |
| **CA3 pattern completion** — auto-association, partial cue → full pattern | `--pattern-completion` | validated at 12% (sharp ~30% threshold, ablation-controlled) |
| **Synaptic tagging & capture** — early-LTP → tag → PRP → late-LTP on CA1→EC | `--stc` | implemented |
| **Memory consolidation** — hippocampo-cortical loop (EC LII, EC LV, mPFC) | `--ec-lii --ec-lv --mpfc` | implemented |
| **Synaptic homeostasis** — sleep downscaling, cortical L-LTP exempt | `--homeostasis` | implemented |
| **Topographic DG wiring** — clustered EC LII→GC fan-in, restores pattern identity washed out by uniform random sampling | `--dg-ec-cluster-sigma` | validated at 1% and 12% (small, real effect) |
| **Pattern discrimination probes** — per-population Jaccard identity/timing separation across interleaved patterns | `--n-patterns` | implemented |

An interactive map of all capabilities, with the functions implementing each,
is in [`capability_map.html`](capability_map.html) (open it in a browser).

## Quick start

Install NEST ≥ 3.9 (see [INSTALL.md](INSTALL.md)), then:

```bash
# 1% test network: replay + dentate gyrus, ~1 min on a laptop
python replay_scaled.py --scale 1 --dg --dg-scale 2 --no-figures

# CA3 pattern completion probe (intact vs ablated recurrence)
python replay_scaled.py --scale 1 --pattern-completion

# full consolidation stack
python replay_scaled.py --scale 1 --dg --ec-lii --stc --n-swr 3
```

Results are written to a self-describing HDF5 file (all populations, spike
times, rates, and per-capability metrics), so analysis and plotting run
anywhere without NEST.

## Running on HPC

[`run.sh`](run.sh) is the SLURM launcher (MareNostrum 5, MPI + OpenMP):

```bash
# replay + DG at 12%
sbatch --export=ALL,SCALE=12,DG=1,NO_STC=1,EC_LII=0,EC_LV=0,MPFC=0 run.sh

# pattern completion at 12%
sbatch --export=ALL,SCALE=12,PATTERN_COMPLETION=1 run.sh

# full stack with DG + consolidation
sbatch --export=ALL,SCALE=12,DG=1,N_SWR=14 run.sh
```

Single-neuron f-I calibration has its own job:
`sbatch --export=ALL,PROBE=dc run_calibrate.sh` → [`run_calibrate.sh`](run_calibrate.sh).

## Layout

| File | Purpose |
|---|---|
| [`replay_scaled.py`](replay_scaled.py) | Main simulation — all populations, capabilities, and HDF5 export |
| [`tiny.py`](tiny.py) | Shared helpers (seeding, theta and SWR generators) |
| [`nest_dg_ca3_fi_calibration.py`](nest_dg_ca3_fi_calibration.py) | DG/CA3 single-neuron f-I + DC-rheobase calibration |
| [`run.sh`](run.sh) / [`run_calibrate.sh`](run_calibrate.sh) | SLURM launchers |
| [`plot_pattern_completion.py`](plot_pattern_completion.py), [`replay_plot.py`](replay_plot.py), [`make_paper_figure.py`](make_paper_figure.py) | Offline plotting from HDF5 |
| [`reconstruct_connectivity.py`](reconstruct_connectivity.py) / [`run_reconstruct.sh`](run_reconstruct.sh) | Extract a projection's actual wired connectivity without a full simulation |
| [`capability_map.html`](capability_map.html) | Visual capability reference |
| [`hippocampal_timescales_as_circuit_spec.md`](hippocampal_timescales_as_circuit_spec.md) | Companion hardware doc — memristive circuit spec derived from this model's timescales |

## Results

**Memory consolidation is dissociable from replay.** Blocking late-LTP capture
(`--prp-threshold 999`) leaves replay quality identical (Δρ_fwd = 0.000) while
cortical consolidation goes to zero — separating the replay mechanism from the
consolidation mechanism in the same model. Full circuit, 12% scale:

![consolidation vs replay](/figures/final_comparison_fig9.png)

**CA3 performs pattern completion.** A partial cue of a stored assembly is
restored to the full pattern by the recurrent collaterals, with a sharp
attractor threshold near 30% cue. Ablating the within-group recurrence
(`sup_local = 0`) abolishes it, confirming the collaterals — not the cue — do
the work:

![pattern completion](/figures/pattern_completion.png)

**Sparsifying the cortex makes consolidation selective — but not yet
specific.** A cortical sparsity retune (§11 in [RESULTS.md](RESULTS.md)) turns
saturated, all-cell L-LTP into a differentiated trace (7.2% of EC cells
consolidate, weight CV 0.17). Scored against standard engram criteria
(Josselyn & Tonegawa 2020), that trace is sparse and persistent but not yet
**specific**: no cortical population discriminates pattern A from pattern B
(§13), and hippocampal lesion does not impair cortical recall at the 12%
scale tested (§12, Test 3 negative). Two convergent, non-topographic
projections are implicated — DG's perforant path and, more severely, CA3→CA1
Schaffer collaterals at 100% density — and a topographic fix
(`--dg-ec-cluster-sigma`) moves DG's identity signal off zero for the first
time, though the effect is still small (§17–18). Full scorecard and
in-progress work: [RESULTS.md](RESULTS.md).

## Key references

- Watson et al. (2025) *Cell Reports* 44:116080 — cell-specific CA3 wiring
  (superficial/deep split)
- Marr (1971); Nakazawa et al. (2002) — CA3 auto-association / pattern completion
- Frey & Morris (1997) — synaptic tagging and capture
- Kassab & Alexandre (2018) — DG mossy-cell threshold classes
- Andersen et al. (2007) *The Hippocampus Book* — reference neuron counts
- Josselyn & Tonegawa (2020) *Science* 367:eaaw4325 — engram criteria (sparse,
  persistent, specific, sufficient, necessary)
- Dolorfo & Amaral (1998) — entorhinal-dentate medial-lateral topography,
  motivating the clustered perforant-path fan-in
- Izhikevich (2006) — polychronization / delay-based temporal coding
- Caus, Sławek, Mazur, Zawal, Baś, Szaciłowski, Talanov & Abdi (2026) — memristive
  hippocampus hypothesis; candidate tag-element devices for
  [`hippocampal_timescales_as_circuit_spec.md`](hippocampal_timescales_as_circuit_spec.md)
