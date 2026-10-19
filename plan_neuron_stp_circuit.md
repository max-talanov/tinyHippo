# Plan: minimal NEURON circuit for short-term plasticity and tag/capture dynamics

**Branch:** `claude/neuron-stp-circuit` (from `main` @ 750080d).
**Spec:** [hippocampal_timescales_as_circuit_spec.md](hippocampal_timescales_as_circuit_spec.md) §1, §2, §4.1, §4.2.
**Sibling plan:** [plan_branch_bio_plasticity.md](plan_branch_bio_plasticity.md), whose §1 lists
short-term plasticity (STP) as "Missing" in the NEST model.

## 0. Purpose and scope

The NEST model uses Izhikevich point neurons. They have no dendrites, no NMDA
Mg²⁺ block and no STP, so the spec's §4.2 mechanism (graded `V_seg` → Ca²⁺-like
gain → tag amplitude) cannot be tested there. NEURON can host it. This branch builds
a **small, biophysical testbed** that:

1. Measures the *weight dynamics* of identified synapse classes on a CA1 pyramidal
   cell: STP on 10 ms–1 s, then tag/capture on minutes–hours.
2. Reproduces the spec's Fig. 2 and its two falsification tests in a single cell.
3. Exports calibrated parameters back to NEST (`tsodyks2_synapse` already exists in
   NEST; `STCHook` takes τ_tag, A₊, PRP threshold).

"STP" here means **short-term plasticity** (spec §1, row 2). The branch also covers
the slower tag/capture axis (§2), because that is the weight dynamic the project
needs from the biophysics. If you meant only STP, stages S0–S3 below are the whole deliverable.

Out of scope: emergent SWR generation, replay ordering as a network property, DG/CA3
pattern separation/completion. These need hundreds of cells and stay in NEST.

## 1. How small can the circuit be?

STP and synaptic tagging are **synapse-local** (the PRP pool is cell-wide, but per
cell). So the minimum is set by what you want to *read out*, not by what you want to model:

| Tier | Cells | Enough for | Not enough for |
|---|---|---|---|
| **A** | 1 pyramidal (PYR), 0 interneurons; presynaptic inputs are spike generators (`VecStim`) | STP of each synapse class (paired-pulse ratio, train depression/facilitation); tag/capture; Frey–Morris weak/strong design | Any circuit-level effect of STP: E/I balance, dendritic gating |
| **B (recommended)** | **1 PYR + 3 interneurons** (PV basket, bistratified, O-LM) | Everything in A, plus how STP at each pathway shapes PYR output and the E/I ratio during a volley | Population effects: ensemble competition, ripple emergence |
| **C (optional)** | 5–10 PYR + same 3 interneurons | Reciprocal PYR→INT→PYR loops with more than one source; weak recurrent drive | Still not a ripple generator |

**Yes, 1 PYR + 3 interneurons is enough for the weight-dynamics goal.** The three
interneurons are chosen to cover the three canonical CA1 inhibitory motifs, not
by count:

| Interneuron | Motif | Target on PYR | Input | Why it is needed |
|---|---|---|---|---|
| **PV basket** | feedforward, perisomatic | soma | Schaffer collateral (SC) + PYR | The spec's `Cluster3` bypass (§4.1): inhibition that skips the threshold/junction pipeline |
| **Bistratified** | feedforward, dendritic | proximal apical (SR) + basal (SO) | SC + PYR | Dendritic inhibition acting directly on `V_seg` of the segment that carries SC synapses |
| **O-LM** | feedback, dendritic | distal apical (SLM) | PYR only | Facilitating feedback onto a segment separate from SC. A pure STP readout (facilitating PYR→INT) |

Caveat: with one PYR, the PYR→INT drive is one cell standing in for an ensemble. Scale
it with the contact count and report that factor. Do not read the E/I numbers from Tier B as
population values. Precedent for this size: Cutsuridis, Cobb & Graham (2010, *J Comput
Neurosci*) used a CA1 microcircuit with one cell of each interneuron class (plus a
pyramidal population); Bezaire et al. (2016, *eLife*) published the full CA1 NEURON model
(ModelDB 187604) from which channel and interneuron parameters can be taken.

## 2. Circuit specification (Tier B)

```
CA3 (SC) stand-in ──┬─► PYR.apical_prox (SR)   AMPA+NMDA, TM-STP        ← strong pathway S
 VecStim trains     ├─► PYR.basal (SO)
                    ├─► PV.soma                AMPA, TM-STP (depressing)
                    └─► BiS.soma               AMPA, TM-STP
T pathway (weak) ───────► PYR.apical_prox (SR)   AMPA+NMDA, TM-STP + tag/capture   ← tagged pathway T
PYR.soma ──► PV, BiS, OLM  (AMPA, TM-STP; PYR→OLM facilitating)
PV ──► PYR.soma (GABA_A)    BiS ──► PYR.apical_prox + basal (GABA_A)    OLM ──► PYR.apical_dist (GABA_A)
```

- **PYR:** soma + basal + apical_prox + apical_dist (4 sections). Na, Kdr, K_A and
  h conductances in the soma; Na/K in the dendrites only if the dendritic-spike
  `thr` of §3.1 is to be tested (stage S5). Start passive dendrites.
- **Interneurons:** one compartment each, Na/Kdr, with class-specific rheobase and
  membrane τ taken from Bezaire et al.
- **Synapses:** AMPA (Exp2Syn kinetics) and NMDA with the Jahr–Stevens Mg²⁺ block
  (needed for §4.2); GABA_A.
- **Cross-check cell:** an Izhikevich point-neuron variant of PYR (ModelDB
  NMODL), so the NEURON and NEST models can be compared on the same inputs (S6).

### Mechanisms to write (NMODL, all `NET_RECEIVE`-based, CVODE-safe via `state_discontinuity`)

1. **`TMSyn.mod`**: Tsodyks–Markram synapse. State `u` (utilisation), `x` (available
   resources). On spike n: `u⁺ = u⁻ + U(1−u⁻)`, `Δg = w·u⁺·x⁻`, `x⁺ = x⁻ − u⁺·x⁻`.
   Between spikes: `du/dt = −u/τ_fac`, `dx/dt = (1−x)/τ_rec`. Parameters `U, τ_rec, τ_fac`
   per pathway. Optional stochastic release: N sites, Bernoulli with `p = u⁺·x⁻`
   (Random123, seed from the project seed rule).
2. **`SegTag.mod`** (spec §2 + §4.2), one instance per tagged synapse group:
   - `τ_v · dV_seg/dt = −V_seg + Σ wᵢ·sᵢ` (τ_v 10–30 ms)
   - `g_Ca(V_seg) = 1/(1 + K·exp(−λ·V_seg))`
   - tag set by coincident pre/post event: `T ← T + g_Ca(V_seg)·[dSpike]`; `dT/dt = −T/τ_tag`
   - PRP pool `P`: incremented by somatic burst events, decays with `τ_P`
   - capture: if `P ≥ θ_PRP` while `T > 0`: `w_cap += κ·T`, decay of `T` stopped for that synapse
   - `θ_PRP = 999` disables capture (the Phase 5 falsification arm).
3. **STDP in the tag step:** ±20 ms window (A₊, A₋), matching `STCHook`, so the
   tag amplitude can be compared with the NEST value.

Time scales: simulate the slow module in **real units** (τ_tag 1–4 h). With
CVODE and no events between SWRs the ODEs are linear decays, so hours of sim time
should be cheap; this is a risk to check in S4 (fallback: advance the slow state
analytically between events). The NEST model compresses 2 s ≙ 1–4 h (factor
≈1.8×10³–7.2×10³, plan_branch_bio_plasticity.md §1); the NEURON branch should
report the compression factor it implies, not assume one.

## 3. Stages and acceptance criteria

| Stage | Experiment | Pass criterion |
|---|---|---|
| **S0** | Environment: `pip install neuron`, `nrnivmodl`, run a toy `Exp2Syn` | Imports and compiles in `.venv`; version recorded in `INSTALL.md` |
| **S1** | **Verification of `TMSyn`** against the closed-form TM solution implemented in numpy: 2-pulse and 20-pulse trains at 5, 20, 50 Hz | Relative error of every Δg < 1%; CVODE vs fixed dt = 25 µs differ < 2% |
| **S2** | **STP signatures per pathway** (voltage clamp at −70 mV, PYR and each INT): paired-pulse ratio at ISI 20–500 ms; steady-state amplitude vs frequency | Sign of PPR−1 matches the expected class: SC→PYR mild, SC→PV depressing, PYR→OLM facilitating. Target numbers are taken from the literature in S2's first step (Markram et al. 1998; Losonczy et al. 2002 for PYR→OLM), **before** fitting. `U, τ_rec, τ_fac` fit by least squares with CIs |
| **S3** | **STP × inhibition on PYR output.** Imposed 100–300 ms SC volley (theta-nested and ripple-like at ~150–200 Hz) through the full Tier-B circuit; vary each pathway's STP on/off (6 conditions) | Reports PYR spike count/latency, E/I conductance ratio over the volley, and Δ(effective weight `w·u·x`) trace per pathway; one pathway ablation changes the output in the predicted direction |
| **S4** | **Tag/capture, Fig. 2 reproduction.** Pathway T (subthreshold alone) receives K = 1…12 SWR-like events at fixed spacing, with PYR driven by pathway S (Frey–Morris weak/strong design). Arms: normal capture vs `θ_PRP = 999` | (a) Tag amplitude per event and decay τ_tag recovered from the trace within 10%. (b) Captured arm shows staircase after PRP crossing; blocked arm stays at baseline. (c) **Falsification:** PYR spike times during pathway-S volleys identical (to the sim dt) in both arms, since S is not plastic; only T's weight differs |
| **S5** | **Graded `V_seg`**: scale NMDAR conductance 0–100%; optionally enable dendritic Na to test the `thr` dSpike | Captured-weight step shrinks continuously with NMDAR dose (spec §4.2 prediction); not an all-or-nothing switch. Report the dose–response with CI over seeds |
| **S6** | **Forward vs reverse order.** Three SC afferent groups A, B, C onto the PYR; fire A→B→C (forward) and C→B→A (reverse) with 3–5 ms spacing (SWR-compressed) | Reports tag-amplitude pattern per group under STDP + STP for each order. Pilot only: with 1 PYR this tests synapse-level order sensitivity, not replay |
| **S7** | **Export to NEST.** Fit `U, τ_rec, τ_fac` per pathway, τ_tag, A±, PRP kinetics. Run the same S4 protocol in a 2-neuron NEST network (`tsodyks2_synapse` + `STCHook`) | NEST tag/capture trace matches NEURON within the S4 tolerance; the parameter table goes into RESULTS.md as a new section |

**Statistics (CLAUDE.md rules):** stochastic-release runs use ≥5 seeds, reported as
mean ± SD; runs are bit-identical at fixed seed/config/thread count, so a rerun of an
identical config adds nothing. Anything seen only at one seed is a pilot.

## 4. Proposed layout

```
neuron_stp/
  mod/            TMSyn.mod  SegTag.mod  NMDA_Mg.mod  (+ interneuron channels from Bezaire et al.)
  cells.py        build_pyr(), build_pv(), build_bis(), build_olm()   (dataclass + build_* pattern)
  circuit.py      build_tier_b(params) -> handles, connections, recorders
  protocols.py    paired_pulse(), train(), swr_volley(), frey_morris(), order_test()
  analysis.py     TM closed form, PPR fit, tag-trace fit, dose-response
  params.py       per-pathway U, tau_rec, tau_fac, tau_tag, ... with source citation per value
  tests/          test_tmsyn_vs_closed_form.py  test_segtag_block.py
  run_s1.py … run_s7.py
```

Each result becomes a new numbered section in [RESULTS.md](RESULTS.md); older sections stay untouched.

## 5. Risks

| Risk | Mitigation |
|---|---|
| Literature STP parameters vary with species, age, [Ca²⁺]ₒ and temperature | Record source and conditions for each value in `params.py`; fit to the S2 targets, not to remembered numbers |
| One PYR stands for an ensemble, so PYR→INT drive is under- or over-scaled | Report the contact-count scaling factor; Tier C as a check on S3 |
| Hours of sim time slow the S4 runs | Linear decays between events; analytic fallback; report wall-clock |
| NMDA/Mg²⁺ and `V_seg` parameters (K, λ, τ_v) are not constrained by the spec | Sweep them in S5 and report sensitivity, not a single value |
| Spec §4.4.2 device data does not show graded SET | S5 predicts the graded response from the biology; it does not validate the device |
| NEURON model and NEST Izhikevich model disagree on firing for the same input | S6/S7 cross-check cell; calibrate PYR rheobase by DC injection (CLAUDE.md rule), not by drive rate |

## 6. Decisions needed

1. Confirm the reading of "STP" (short-term plasticity + the tag/capture extension) vs short-term plasticity only.
2. Tier B (1 PYR + 3 INT) as the target, with Tier A as the first runnable milestone?
3. Slow module in real time (recommended) or compressed to match NEST?
