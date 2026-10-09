# Branch plan: bio-plausible plasticity

Implement the plasticity specified in
[hippocampal_timescales_as_circuit_spec.md](hippocampal_timescales_as_circuit_spec.md)
inside the tinyHippo model, so that every mechanism in the spec has a simulated
counterpart, the spec's falsification tests run in the model, and the model can act
as a hardware-aware testbed for the proposed tag/capture devices.

- **Branch:** `bio-plasticity`, created from `main` at the commit where the gate
  below passes.
- **Sibling branch:** `llm-integration` ([plan_branch_llm_integration.md](plan_branch_llm_integration.md)),
  created from the same gate commit.

## 0. Entry condition: pattern identity survives consolidation

Neither branch starts until the current architecture passes this gate. "Identify
the IDs of inbound information after consolidation" is turned into a pass/fail test:

| # | Test | Pass criterion |
|---|---|---|
| G1 | **Decodable after consolidation.** A leave-one-block-out nearest-centroid classifier reads the pattern ID from SWR-window per-cell spike counts in mPFC (primary) and EC LV. | Accuracy above chance (1/3), permutation p < 0.01, in each of ≥2 seeds at 12% |
| G2 | **Survives the lesion.** With `--cortical-recall`, after the hippocampal lesion, a partial cue of pattern k reactivates k's mPFC assembly more than the other patterns' assemblies. | Specificity > 0, permutation p < 0.01, ≥2 seeds |
| G3 | **Claims 1–2 still hold.** Replay ρ stays in the JOB F band, PRP-block falsification stays flat, and DG stays 2–4% active. | No regression against the RESULTS.md reference runs |

Notes:
- Use rate-based readouts (CLAUDE.md: Jaccard saturates in CA3 and CA1).
- Run the gate with `--n-swr 28` or more. With 14 blocks and 3 patterns, each
  class has only 4–5 blocks, which is too few for a classifier. That is about 11h
  per 12% run, within the 20h limit.
- The gate can first be met with `--pattern-source ca3`. The full gate uses
  `--pattern-source ec-lii`, the model's intended input path.

**Status (RESULTS.md §26):**

| Hop | Status |
|---|---|
| CA3 → CA1 | Passes: group code z = +0.79 at 12%, 1 seed |
| CA1 → EC LII | Open (JOB H12) |
| EC → mPFC | Not started |
| EC LII → CA3 groups (§21) | Open |

The shared scorer `decode_identity.py` (G1/G2) is built while working toward the
gate, and both branches use it as their main metric.

## 1. What exists today

| Mechanism | Where | Current rule | Spec element | Gap |
|---|---|---|---|---|
| STC on CA1→EC LII | `STCHook` (Python, runs between blocks) | Pair STDP in the SWR window (A₊ = 0.01). The tag is \|Δw\| and decays with τ_tag = 2000 ms of sim time. Each EC cell has a PRP pool that counts the SWR events it fired in. At PRP ≥ threshold, the synapse is captured with a ×1.3 weight gain. After 5 L-LTP events, a structural boost is applied. | §2, §4.4 | The tag amplitude is set by STDP alone (no `V_seg`). Only the ideal exponential tag is implemented, with no device model. The time compression is not documented anywhere: 2 s of sim time stands in for the spec's 1–4 h, a factor of about 1,800–7,200. |
| EC LV→mPFC association | `MPFCAssocHook` | Continuous Hebbian potentiation, with depression when post fires without pre | §4.5 slow tier | The spec's slow tier accumulates SET pulses until it latches non-volatile. The model's rule is continuous instead. |
| Homeostasis | `--homeostasis` | Global multiplicative downscaling by α | §3.2 synaptic homeostasis | Global only. The spec places it per synaptic area and makes it activity-dependent. |
| Input clustering | `--dg-ec-cluster-sigma`, `--schaffer-group-frac`, `--ca1-ec-group-frac` | Wired by hand at build time | §3.3 clustering, §4.2 cooperativity | No mechanism produces the clustering. It is the least bio-plausible part of the current fix. |
| Short-term plasticity | none | none | §1, row 2 | Missing |
| Dendritic thr → junction → soma pipeline | none (Izhikevich point neurons) | none | §3, §4.1 | Out of scope here, see §5 |

**Engineering constraints carried over from `main`:**
- On MN5, `stdp_synapse` connects run on a serial path (~300 syn/s), so new plasticity
  stays in Python hooks unless a NESTML model is shown to connect fast.
- Hooks fetch their synapses source-side with `get_conns_between` (RESULTS §24).
- Every new mechanism is an opt-in flag with a dataclass and a `build_*` function
  (CLAUDE.md).
- With all new flags off, a build must be bit-identical to `main`.

## 2. Phases

Every phase runs a 1% pilot first and is then confirmed at 12% on ≥2 seeds (CLAUDE.md).
Each phase ends with a falsification test taken from the spec.

### P0 — Branch setup, regression harness, and a timescale map (no behavior change)

1. Create the branch: `git switch -c bio-plasticity <gate-commit>`.
2. Add `tests/regression_build.py`, built from the checks that already worked on
   `main` (§24–25). It hashes the Schaffer, CA1→EC and LV→mPFC synapse sets at 1%,
   and the STC weight vector after N events. Every later phase must keep these
   hashes identical when its flag is off.
3. Add a documented sim-time ↔ biological-time table, `TIMESCALES` in a small
   module, with a copy in RESULTS.md:
   - one 1000 ms block ↔ one SWR-bearing sleep episode;
   - τ_tag of 2000 ms ↔ the spec's 1–4 h;
   - the PRP threshold in SWR events ↔ the ~30–60 min onset of late-LTP.
   Choose the compression so the *ratios* that drive capture match the spec,
   τ_tag : inter-SWR interval in particular (4 : 1 today). Record the choice.

**Acceptance:** all hashes match `main`, and the table is reviewed.

### P1 — A pluggable tag element (spec §2, §4.4)

1. Move the tag state out of `STCHook` into a `TagElement` interface with the
   operations `set(amplitude)`, `decay(dt)`, `try_capture(prp_ok)` and `value`.
2. Add `--tag-backend` with three backends:
   - `ideal`: today's exponential tag. It must reproduce `main` exactly (same hash).
   - `digital` (§4.4.1): an n-bit register with a slow down-counter and a zero
     comparator. Capture copies the register into a non-decrementing capture
     register. Parameters: bits and decrement period.
   - `memristor` (§4.4.2): a volatile relaxation fitted to the copper-aspirinate
     parent retention curve (LRS→HRS in about 50 min, scaled through the P0 table).
     Capture switches the site to non-volatile (the benzimidazole derivative).
     Device-to-device variability σ_τ and σ_G is a parameter.
3. **Falsification test:** with capture blocked (`--prp-threshold 999`), the weight
   distribution must stay flat and replay ρ unchanged under every backend (spec Fig. 2).
4. **Main deliverable:** a hardware-tolerance curve, G1 decodability against σ_τ
   and against register bit depth. This answers how much device variability
   consolidation can tolerate.

### P2 — Graded tag amplitude, `V_seg` (spec §4.2)

1. Izhikevich cells have no dendrites, so define a *segment* as a group of synapses
   on one post cell. The natural choice is the inputs from the same home group,
   since those clusters already carry the identity code.
2. In the hook, integrate each segment's presynaptic spikes during the SWR window:
   τ_v · dV_seg/dt = −V_seg + Σ wᵢ sᵢ(t), with τ_v = 10–30 ms. Then set
   `tag_amplitude = g_Ca(V_seg) · [coincidence]`, with
   `g_Ca(V) = 1 / (1 + K·exp(−λV))`.
3. The flag is `--vseg`, with parameters τ_v, K and λ.
4. **Falsification test (the spec's prediction):** a dose-dependent partial NMDA
   block, modelled as scaling down g_Ca, must shrink the captured step size
   *continuously*, not switch captured synapses off all at once.
5. **Hypothesis to test, not assume:** cooperativity gives co-active clustered inputs
   larger tags, so pattern-consistent synapses consolidate selectively and G1
   accuracy in EC and mPFC rises relative to P1's `ideal` backend.

### P3 — Two-tier consolidation (spec §4.5)

1. Add `--cortical-tier latch`. For EC LV→mPFC, each replay pass with pre/post
   co-activity adds one SET increment to a per-synapse accumulator. Once the
   accumulator crosses N_set, the synapse latches non-volatile. Unlatched
   accumulators decay slowly.
2. **Tests:**
   - Plot the systems-consolidation curve, the latched fraction against replay passes.
   - Run the before/after lesion test from spec §4.5. Recall after a lesion must
     fail with too few passes and succeed with enough. This is also the engram
     "sufficient and necessary" test for claim 3.
3. Hardware knob: the SET-accumulation rate N_set. Sweep it against G2.

### P4 — Clustering from plasticity instead of wiring (spec §3.3 + §4.2)

This is the bio-plausible replacement for the hand-set `--*-group-frac` flags, and
the most important scientific result the branch can produce.

1. Add `--struct-rewire`. Between blocks, prune the weakest few percent of each post
   cell's inputs and regrow the same number. Regrowth favors presynaptic cells whose
   SWR-window activity correlates with the segment's `V_seg` (Hebbian rewiring).
   In-degree stays fixed.
2. Start from uniform wiring (`group_frac = 0`) and track home-group purity over
   blocks (the §25 metric; chance is 1/n_groups).
3. **Success criterion:** purity rises well above chance without being wired in,
   CA1 group z reaches the §26 level (≈ +0.8), and G1 passes.
4. **Risks:**
   - NEST `Disconnect`/`Connect` cost at 12%. Measure it at 1% first, and cap
     rewiring at a few % per block.
   - Every rewire invalidates the cached SynapseCollections, so hooks must re-fetch
     through `get_conns_between` afterwards.

### P5 — Short-term plasticity (spec §1, optional)

1. Try `tsodyks2_synapse` on the mossy fibres (facilitation is their hallmark)
   and on Schaffer.
2. First measure its `Connect` speed on MN5. It may hit the same serial path as
   `stdp_synapse`. Adopt it only if it is fast there.

## 3. Metrics carried by every phase

| Metric | Source | Must not regress |
|---|---|---|
| G1 decodability, G2 lesion specificity | `decode_identity.py` | Yes, it is the branch's purpose |
| Replay ρ forward/reverse | `replay_score` | Yes (claim 1) |
| PRP-block falsification | `--prp-threshold 999` | Yes (claim 1) |
| DG active 2–4%, block-by-block | h5 | Yes (claim 2) |
| Wall time at 12% | slurm log | Keep under the 8h job limit where possible |

## 4. Risks

| Risk | Mitigation |
|---|---|
| The time compression distorts capture dynamics | Match ratios, not absolute times (P0), and document every constant |
| Python hooks become slow as rules get richer | Vectorize (the STC loop is still per-synapse Python), and profile at 12% in P1 |
| Rewiring cost in NEST (P4) | Start with a few % per block, and measure at 1% before 12% |
| Device models are over-fitted to one paper's retention data | Treat σ as a swept parameter rather than a fitted constant, and report tolerance curves |
| Changes to `main` (gate fixes) conflict with the branch | Merge `main` into the branch after every gate-relevant commit, and keep hooks behind flags |

## 5. Out of scope for this branch

- **NEST GPU.** This branch stays on CPU NEST by decision ([nest_gpu_migration.md](nest_gpu_migration.md) §0): tag elements, `V_seg` and NESTML models need per-synapse and per-neuron flexibility NEST GPU lacks.
- The full dendritic thr → junction → soma pipeline (spec §3.1, §4.1) with refractory
  junctions. That needs multi-compartment NESTML neurons, which would be a separate
  effort after P4.
- Neuromodulator-gated tagging, the second synaptic-area input in spec §3.2. It is
  noted as a follow-up and is also the natural link to the LLM branch's "importance"
  signal ([plan_branch_llm_integration.md](plan_branch_llm_integration.md), L6).

## 6. First steps once the gate passes

- [ ] `git switch -c bio-plasticity <gate-commit>` and push
- [ ] `tests/regression_build.py` (P0.2), green on the branch
- [ ] `TIMESCALES` table (P0.3), plus a RESULTS.md section
- [ ] `TagElement` + `ideal` backend, hash-identical to `main` (P1)
- [ ] `digital` backend, then the PRP-block falsification test at 1%
