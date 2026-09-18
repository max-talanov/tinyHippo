# tinyHippo — results summary

A bio-plausible spiking model of the rat hippocampus (NEST 3.9, Izhikevich
neurons) demonstrating **memory consolidation through bidirectional replay and
sharp-wave ripples** across the full entorhinal–hippocampal loop.

```
EC LII → DG → CA3 → CA1 → EC LII/LV → mPFC
   ↑                            │
   └────────────────────────────┘
```

Scales 1 % → 100 % of the rat hippocampus from one code path
([`replay_scaled.py`](replay_scaled.py)); 12 % ≈ 101 k neurons on MareNostrum 5.

![overview](figures/consolidation_overview.png)

**Bottom line so far.** Replay, pattern separation (sparseness only), pattern
completion, tagging-and-capture, and *selective* consolidation (§11) all work
at 12 % scale. Pattern **specificity** does not yet survive to cortex — §13
shows no population discriminates pattern A from pattern B, and §17–18 trace
part of the cause to convergent, non-topographic wiring (DG's perforant path,
and worse, CA3→CA1 Schaffer collaterals at 100 % density) that averages
identity away by construction. A clustered, topographic fix moves DG's
identity signal off zero for the first time (§17–18) but the effect is still
small and the cortical read is confounded. Test 3 (hippocampal lesion →
cortical recall) is negative at 12 % (§12): §13 says that is expected, since
there is no pattern-specific trace yet to recall. See §13 for the full
engram scorecard and §18/Open items for what is still running.

The tag-decay/capture dynamic validated in §2, and the replay ⊥ consolidation
dissociation in §3, are also the two findings the companion hardware document,
[`hippocampal_timescales_as_circuit_spec.md`](hippocampal_timescales_as_circuit_spec.md),
takes as its circuit specification: STC as a volatile-memristor tag element,
and SWR replay as the volatile-organic → non-volatile-inorganic handoff that
drives systems consolidation. That document also cites Caus, Sławek, Mazur,
Zawal, Baś, Szaciłowski, Talanov & Abdi (2026) as a first fabricated-and-tested
candidate device for the tag element described here.

---

## 1. Bidirectional replay

Forward and reverse sequence replay during SWRs, scored as Spearman ρ between
CA3 sequence-group index and mean spike time.

| scale | ρ forward | ρ reverse |
|---|---|---|
| 12 % (+ real DG, consolidating) | **+0.613** (p<0.001) | **−0.789** (p<0.001) |
| 1 % full stack | +0.782 (p=0.008) | −0.794 (p=0.006) |

Both directions pass the ±0.5 criterion **while consolidation is running** —
the two are not in tension.

> **Correction to an earlier conclusion.** Reverse replay previously appeared
> incompatible with consolidation (ρ_rev ≈ −0.03). That was a scoring-window
> artifact: `replay_score()` was called with `(swr_start−5, swr_stop+30)`, and
> the +30 ms tail reaches into the post-SWR rebound where the strong forward
> chain re-ignites a *forward*-propagating burst. That rebound shares the
> forward ordering (so forward was unaffected) but cancels the reverse ordering.
> Re-scoring the archived runs on the SWR window itself recovers reverse replay
> everywhere, including runs from months earlier:
>
> | run | ρ_rev padded | ρ_rev SWR-window |
> |---|---|---|
> | 12 % STC (May) | −0.035 | −0.437 |
> | 25 % STC+LV+mPFC (May) | −0.033 | −0.512 |
> | 12 % + DG (Aug) | −0.094 | **−0.789** |
>
> ρ_fwd is essentially unchanged (+0.622 → +0.613), confirming a one-sided
> distortion. Fixed at all three call sites.

## 2. Synaptic tagging and capture

CA1→EC LII synapses carry STDP-derived tags; a per-EC-neuron PRP pool
accumulates one unit per SWR activation; L-LTP is captured where PRP crosses
threshold **and** a tag is still alive (Frey & Morris 1997).

At 12 %, 28 SWR events: tagging is immediate, the PRP pool builds linearly, and
L-LTP stays at zero until **event 15**, then rises sharply to **98 %** of the
612 k CA1→EC synapses. Mean weight 1.01 → 1.50. The threshold-then-jump shape
is the tag-and-capture signature, not gradual drift.

## 3. Memory consolidation

Consolidation is driven by replay: EC LII fires during SWRs, the PRP pool
builds, and tagged synapses are captured into L-LTP. With the real DG in the
loop the full cortical stack is active (EC LII 6.1 Hz, EC LV 13.0 Hz, mPFC
5.7 Hz) and the consolidation curve matches the pre-DG reference (600 250 vs
600 247 L-LTP synapses).

**Replay ⊥ consolidation.** Blocking L-LTP capture (`--prp-threshold 999`)
leaves replay quality identical (Δρ_fwd = 0.000) while cortical consolidation
goes to zero — the two mechanisms are dissociable in the same model
([`figures/final_comparison_fig9.png`](figures/final_comparison_fig9.png)).

## 4. Dentate gyrus — pattern separation

Real granule cells, two mossy-cell classes and basket feedback replace the
former Poisson proxy; CA3 is driven through mossy-fibre detonator synapses
(low in-degree, high weight).

- Granule active fraction **2.1 % per SWR window** at 12 % (target 2–4 %).
- MC_HIGH / MC_LOW DC-rheobase ratio **4.97×** vs the 5.0× Kassab & Alexandre
  target, confirmed on MN5 (`I_e = −15.1`; the original −9.0 gave only 3.25×).

Rheobase must be measured by **DC current injection**, not Poisson drive: at
weight 20.0 a single EPSP equals the granule cell's entire 20 mV
rest→threshold gap, so the cell relays rather than integrates.

## 5. CA3 — pattern completion

A partial cue of a stored assembly is restored by the recurrent collaterals.

![pattern completion](replay_output_1pct/pattern_completion.png)

| cue | intact | ablated (`sup_local=0`) |
|---|---|---|
| 10 % | 0.01 | 0.00 |
| 30 % | 0.50 | 0.00 |
| 50 % | 0.92 | 0.00 |
| 70 % | 1.00 | 0.00 |

Sharp attractor threshold ≈ 30 % cue; the ablated control is flat at zero with
cue recall still 1.0, proving the collaterals — not the cue — do the work
(Marr 1971; Nakazawa et al. 2002). The probe needs CA3 primed to a
sharp-wave-like state and a local E/I rebalance; the replay-tuned CA3 is
inhibition-dominated and is not an autonomous attractor without it.

---

## 6. Cortical association build-up (EC LV → mPFC)

A replay-gated Hebbian hook potentiates EC LV→mPFC synapses when an SWR
co-activates both ends, with weak heterosynaptic depression when the target
fires without the source. mPFC also has a k-winners-take-all interneuron loop
(pyramidal → FS → pyramidal), mirroring the DG.

**The association builds, but it is not yet an engram.** Weights grow steadily
(1.00 → 1.10 over 12 replay events at 1 %), but every synapse moves together:
the final weight distribution has a **single unique value**, CV = 0.0000.

The cause is upstream of mPFC, so lateral inhibition cannot fix it. EC LV fires
**all-or-nothing** per SWR window — 600/600 cells or 0/600 — so every mPFC cell
receives identical drive and the winners-take-all loop has no differences to
amplify. mPFC correspondingly fires 120/120 or 0/120. A genuine engram needs
pattern-specific activity to survive the CA1→EC→mPFC path so that different
replayed sequences recruit different cortical subsets.

> An earlier version of this section reported "37.5 % of synapses associated"
> as evidence of a selective engram. That was wrong: `frac_associated` compares
> uniform weights against a threshold that moves with event count, so it can
> report an apparent fraction while every synapse is identical. Weight CV is
> the honest test and is now what the report and HDF5 record.

## 7. Pattern discrimination — why the engram needs a temporal code

An engram is only meaningful with more than one thing to remember: "selective"
means selective for A rather than B. `--n-patterns P` splits the CA3 sequence
groups into P interleaved assemblies, each replayed in its own epochs, making
downstream selectivity measurable for the first time.

**Cortex does not discriminate the patterns.** Jaccard overlap of the active-cell
set, within-pattern (same pattern, different epochs) vs between-pattern:

| population | active | within | between | separation |
|---|---|---|---|---|
| CA3 SUP | 98.3 % | 0.968 | 0.967 | 0.001 |
| CA1 PYR | 49.3 % | 0.290 | 0.288 | 0.002 |
| EC LII | 46.9 % | 0.145 | 0.202 | −0.057 |
| EC LV | 71.0 % | 0.507 | 0.478 | 0.028 |
| mPFC | 86.1 % | 0.117 | 0.226 | −0.110 |

**But the patterns are strongly encoded in CA3 — in spike timing.** Correlating
the per-group *activation-time* profile across epochs, versus the per-group
*active-cell-count* profile:

| code | within | between | separation |
|---|---|---|---|
| **timing** (when each group fires) | **+0.954** | **−0.114** | **+1.069** |
| identity (how many cells fire) | +0.745 | +0.242 | +0.503 |

Timing discriminates the two patterns almost perfectly and ~2× better than
identity. The same conclusion follows from the replay score itself, which is a
timing measure (ρ = +0.72 / −0.55) and works fine.

So the information is present and temporal; what loses it is the readout. Every
projection in the model uses a **single scalar delay** (`d_fast=1.5`,
`d_slow=3.0`, `delay_ca1_ec=3.0`, `delay_lv_mpfc=8.0`), so all spikes arrive
simultaneously and timing carries nothing downstream — and the association hook
is a window-coincidence rule, which is blind to timing by construction.

This is the empirical case for a **polychronization**-style temporal code
(Izhikevich 2006): per-synapse delay heterogeneity plus STDP, so that a
polychronous group — a directed graph of (neuron, delay) edges — becomes the
carrier of pattern identity. It also motivates the cheaper complement: sparser,
more topographic Schaffer connectivity so identity survives CA3→CA1.

## 8. Phase C step 1 — delay heterogeneity alone is not sufficient

`--delay-jitter MS` gives each synapse on the feedforward readout projections
(Schaffer, CA1→EC LII/LV, EC LII→LV, EC LV→mPFC) its own delay, drawn uniformly
around the projection's base value. The CA3 sequence chain and EC LV→CA3
feedback stay scalar: those delays are load-bearing for replay *generation*, so
jittering them would perturb the thing being measured.

Controlled test at 1 %, 2 patterns × 4 replays (12 within-pattern pairs).
`--delay-jitter-wcomp` scales the jittered weights, because spreading arrival
times reduces coincident summation and lowers downstream rates — a confound
that has to be controlled rather than ignored.

| condition | CA3 | CA1 | EC LII | EC LV | mPFC |
|---|---|---|---|---|---|
| base, jitter 0 | **+0.200** | −0.057 | −0.033 | +0.052 | −0.037 |
| jitter 4 ms, w×1.0 | **+0.210** | +0.032 | −0.024 | −0.060 | — |
| jitter 4 ms, w×1.5 | **+0.206** | +0.001 | +0.025 | −0.014 | +0.032 |

(timing separation = within-pattern minus between-pattern correlation)

CA3 is unchanged across all three (+0.200 / +0.210 / +0.206), confirming the
manipulation is correctly targeted — replay generation is untouched. But no
condition produces downstream discrimination: every cortical value sits in
±0.06, i.e. noise.

The rate confound is **bracketed** rather than exactly matched: w×1.0 leaves
cortex at 87–89 % of baseline and w×1.5 overshoots to 128–163 %. Since the two
straddle 100 % and neither shows the effect, the negative result holds across a
±60 % range of cortical firing rate and is not a rate artifact.

This is the expected outcome from polychronization theory: in Izhikevich (2006)
it is **STDP that selects delay-matched paths**. Heterogeneous delays are the
substrate; without a mechanism that potentiates the delay-matched combinations,
jitter only smears arrival times. Delays are necessary but not sufficient —
so the informative next experiment is delays **plus** STDP, not delays alone.

## 9. Phase C step 2 — delay-aware STDP: **does not replicate**

Pair-based STDP on the Schaffer collateral using each synapse's **own** delay,
`dt = (t_post − t_pre) − delay_ij` — the polychronous selection rule, applied at
CA3→CA1 because that is where the signal still exists.

The **mechanism** demonstrably operates: Schaffer weight **CV rises
0.0008 → 0.0388** over 16 replay events, so synapses differentiate rather than
drift together.

A first run showed EC LV timing separation **+0.120**, surviving two static
controls that bracketed (and exceeded) its firing rates. That looked like a
controlled positive. **It does not survive replication.**

Three independent seeds (`--seed` sets both the NEST kernel RNG and the numpy
RNG; varying only one leaves runs partially identical):

| run | CA3 | CA1 | EC LII | EC LV | mPFC |
|---|---|---|---|---|---|
| original | 0.143 | 0.025 | 0.067 | **+0.120** | 0.048 |
| seed 101 | 0.159 | 0.021 | 0.012 | −0.037 | −0.035 |
| seed 202 | 0.171 | 0.018 | **0.556** | −0.029 | −0.002 |
| seed 303 | 0.195 | 0.046 | **0.135** | −0.006 | 0.042 |

EC LV across the three new seeds: mean **−0.024**, sd 0.016. The original
+0.120 sits ~9 sd above that — an outlier, not an effect.

Two further signs it is noise rather than signal:

- The population that "discriminates" **flips between runs**: CA3+EC LV, then
  CA3 only, then CA3+EC LII twice. A real effect lands on the same population
  each time.
- EC LII swings from 0.012 to 0.556 across seeds — a range no mechanism explains.

What *is* robust is the source: **CA3 timing separation 0.167 ± 0.022** across
all four runs. The encoding is real and reproducible; the downstream
transmission is not.

**Conclusion.** Delay heterogeneity alone does not carry the timing code to
cortex (§8), and delay-aware Schaffer STDP does not either. Polychronization
remains a plausible route — the substrate and the selection mechanism are now
both implemented and the mechanism verifiably runs — but at this scale, epoch
count and connectivity it produces no reproducible cortical discrimination.
Candidate next factors: far longer training (Izhikevich ran ~24 h simulated for
polychronous groups to form; these runs are 8 s), recurrent rather than
feedforward cortical targets, and larger populations.

## 10. Cortical sparsity, and a first readable Test 3 (PRELIMINARY)

Test 3 (hippocampal lesion -> cortical recall) was previously unaskable: mPFC
fired 120/120 cells, so there was no assembly to cue, and cortex had no
recurrent excitation to complete a pattern with. Three fixes, in order:

**Sparsity.** Cortical volleys were suprathreshold (CA1->EC LII at 2.5x), so
every cell fired ~1.5 ms before feedback inhibition could arbitrate. Rescaling
each stage to just below threshold AND adding `I_e` heterogeneity gives graded
recruitment. Both were needed: subthreshold volleys with uniform `I_e` fired
nobody, and `V_m` heterogeneity alone did nothing — every initial potential
relaxes to the same rest (−70), so V_m spread is a transient, not a standing
excitability difference.

| population | before | after |
|---|---|---|
| EC LII | 87.3 % | **9.5 %** |
| EC LV | 99.9 % | **62.5 %** |
| mPFC | 100 % | **9.8 %** |
| mPFC assembly | 120/120 | **23/120** |

**E/I balance.** mPFC interneurons were at 0.00 Hz, so the winners-take-all loop
was inert and halving its weight was a no-op (2.0 % -> 2.3 %). mPFC was
excitation-starved; raising EC LV->mPFC brought it to 9.8 % with `mpfc_int` at
0.29 Hz.

**Post-lesion priming.** After the lesion mPFC has no input at all, so the test
otherwise asks cortex to self-ignite from silence. A subthreshold tonic drive
holds cells near threshold; the pre-cue baseline is the guard that it is not
firing them itself.

Result at 1 %, 40 % cue of a 23-cell assembly, hippocampus lesioned:

| condition | completion | baseline |
|---|---|---|
| consolidated (plasticity on) | **0.214** | 0.000 |
| control (no plasticity) | 0.071 | 0.000 |
| priming 60 Hz (too weak) | 0.000 | 0.000 |

Directionally this is what systems consolidation predicts — 3x more recall when
the recurrent weights were shaped by replay, against a proper unconsolidated
control and a zero baseline.

**It is not yet evidence.** Those fractions are **3 cells versus 1 cell**, from a
single seed. Poisson noise on counts that small is comparable to the effect, and
this is the same profile as the EC LV result in §9 that failed to replicate
across three seeds. Cortex also still shows no pattern discrimination by
identity, so even a solid positive would show *an* assembly reactivating rather
than *a specific memory* transferring.

Settling it needs 12 % scale, where mPFC is 1440 cells and an assembly ~250, so
recovered counts are ~30 vs ~10 — measurable rather than anecdotal — plus
several seeds.

## 11. The sparsity retune at 12 % — consolidation becomes selective (Job F)

The cortical sparsity retune of §10 changed EC LII/LV and mPFC weights, `I_e`
heterogeneity and E/I balance, and had only ever been run at 1 %. Job F
(`res/2026-08-17/`, `SCALE=12 DG=1 N_PATTERNS=2 N_SWR=14`) is the sanity run at
12 %. The core holds:

| | pre-retune (08-05) | post-retune (08-17) |
|---|---|---|
| replay ρ_fwd / ρ_rev | +0.70 / −0.54 | **+0.63 / −0.66** (both p<0.001) |
| DG granule active per window | 2.16 % | 1.55 % fwd, 1.60 % rev |
| CA1 PYR | ~5 Hz | 4.82 Hz |
| mPFC | 5.73 Hz (302 Hz in 08-07) | **1.19 Hz** |
| EC LII | 6.12 Hz, 12005/12005 firing | 0.40 Hz, 1847/12005 firing |
| L-LTP | 98.0 % of synapses | **7.0 %** |

The L-LTP drop looks like a regression and is the opposite. In the dense regime
*every* EC cell fired in *every* SWR window, so the PRP pool was just an event
counter — `prp_mean` tracked the event index exactly, everything crossed the
threshold of 14 together at event 14, and the final weights had CV 0.003 pinned
at the ceiling. That is the same non-selective saturation that produced the
retracted "engram" claim, seen from the consolidation side.

With a sparse cortex only ~9 % of EC cells fire per window, PRP accumulates at
0.075/event on average, and the cells that cross are the ones that reliably
participate:

```
                consolidated (862 cells)   rest (11,143)
  final PRP           24.0                     0.0        threshold 14
  weight               0.486                   0.305      1.59x
```

862 of 12 005 EC cells (7.2 %) end up consolidated, weight CV 0.170 spread over
0.29–0.74 — a differentiated trace rather than a saturated one. Consolidation is
also **all-or-none per postsynaptic cell**: consolidated cells have ~49.9 of
their ~51 incoming synapses captured. That is what STC predicts, since the PRP
pool is somatic — and it is a property the dense regime could not have revealed.

So the sparsity retune did not cost consolidation; it made consolidation
selective. Whether that 7.2 % assembly is *pattern*-selective is Test 3, which
has not yet run at 12 % (see below).

### Job D timed out — in the lesion, not the simulation

Job D (Test 3 consolidated) hit the 20 h limit, so Job E was never started. The
simulation was not the problem: all 16 epochs finished in **10 758 s (3.0 h)**,
close to the estimate. It then spent >9 h inside `lesion_hippocampus()`, which
called `nest.GetConnections(source=CA1, target=EC)`. Filtering on `source` makes
NEST scan the entire kernel connection table — ~19 M synapses at 12 % once the
Schaffer STDP set exists. `build_ec_lii()` and `build_stc_hook()` both carry
comments warning about exactly this; the lesion path was written later and
missed it.

Fixed by reusing the connection handles the STC hook already fetched (cost zero)
and falling back to `GetConnections(target=)` with numpy source-filtering, which
touches only that population's incoming slots. No parameter changes are needed —
the run fits comfortably in 20 h once the scan is gone.

## 12. Test 3 at 12 % — cortical recall does not survive the lesion (Jobs D + E)

`res/2026-08-19/`, seed 101, 16 epochs, `SCHAFFER_K=200`, `DELAY_JITTER=4.0`.
Both jobs completed: the lesion that consumed >9 h in the previous attempt now
runs from cached handles, and the post-lesion probe window is present in both
outputs. The hippocampal side is bit-identical across D and E (ρ +0.632/−0.656,
L-LTP 10.1 %, DG 1.52 %), as it must be — `NO_MPFC_ASSOC` touches only the
cortical hook — so the comparison is properly controlled.

Recall reconstructed from the saved mPFC spikes (cue at 16 200 ms, 80 ms
scoring window, 40 % of the assembly cued, baseline 16 100–16 180 ms):

| | assembly | uncued | cue_recall | completion | baseline | net |
|---|---|---|---|---|---|---|
| D consolidated | 309 | 185 | 0.992 | 0.049 (9 cells) | 0.022 (4) | **+0.027** |
| E control | 303 | 182 | 1.000 | 0.016 (3 cells) | 0.005 (1) | **+0.011** |

**This is a negative result.** The `[OK]` criterion is 0.25, which at this
assembly size means ~46 of 185 uncued cells; D reactivated 9. Both runs miss it
by roughly an order of magnitude, so the gap is not a matter of statistical
power — cortex does not reconstruct the pattern once the hippocampus is cut.

D is directionally above E, 3x on the raw counts, matching the direction of the
1 % preliminary in §10. It does not survive testing: Fisher exact on 9/185 vs
3/182 gives **p = 0.140**, from a single seed. This is the same profile as the
EC LV effect in §9 that failed to replicate across three seeds, and it should
not be reported as an effect. D against its own pre-cue baseline is p = 0.020,
but that is 9 cells against an expected 4.

**Correction.** An earlier version of this section read the mechanism off the
HDF5 `mpfc_assoc` group, reporting that "the mPFC recurrent hook barely moved —
614 of 28 800 synapses at ceiling". That group is the **feedforward EC LV→mPFC**
projection (`K_eclv_mpfc` 20, `w_init` 1.8), not the recurrent one. The
recurrent hook runs but its weights were never written to the file, so those
numbers said nothing about the cortical attractor. The recurrent weights are now
exported as a separate `mpfc_recurrent` group. The Test 3 verdict above is
unaffected — it is computed from spikes.

What can be said without those weights is structural, and it is enough. The
recurrent projection is `K_rec = 20` random inputs per cell over N = 1440, so an
uncued assembly cell receives on average

    20 x 124/1440 = 1.72

inputs from the 124-cell cue. At the initial `w_rec` 0.9 that is ~1.5 mV against
a ~20 mV rest→threshold gap; even pinning every one of those synapses at a 2.4
ceiling reaches only ~4.1 mV. **Test 3 cannot pass at `K_rec = 20` however well
the learning works** — clearing threshold needs ~8 ceiling-weight inputs from
the cue, i.e. `K_rec` ≳ 97, and that already assumes synchronous arrival, which
does not hold (§10). Note this is the same `K x w` reasoning that misled the E/I
iteration, so treat it as an order-of-magnitude bound, not a prediction.

E/I balance is **not** the lever, despite the analogy to CA3 in §5. During the
post-cue window mPFC interneurons fire 6 of 288 cells at 0.26 Hz — there is
almost no inhibition to rebalance, the same trap that made an earlier attempt to
halve mPFC inhibition a no-op.

So the honest state of the loop: replay, separation, completion, tagging and
selective consolidation all work at 12 %, and the trace reaches cortex as
weights — but it is not yet strong enough to be *read out* without the
hippocampus, which is what systems consolidation requires.

## 13. Do we have engrams? No — and that is the root problem

Scored against the standard criteria (Josselyn & Tonegawa 2020):

| criterion | status |
|---|---|
| sparse | yes — 862/12 005 EC cells consolidate, all-or-none per cell (§11) |
| persistent | yes — L-LTP captured, 1.59x weights |
| **specific** | **no** |
| sufficient | no — Test 3, 0.049 vs a 0.25 criterion (§12) |
| necessary | never tested |

Specificity is the definitional core of an engram, and at 12 % it is not weak,
it is absent. Within-pattern vs between-pattern separation, Job F (14 epochs,
2 patterns alternating):

| population | active | identity sep | timing sep |
|---|---|---|---|
| CA3 SUP | 98.0 % | 0.000 | 0.028 |
| CA1 PYR | 100.0 % | −0.003 | 0.002 |
| EC LII | 18.4 % | +0.055 | −0.027 |
| EC LV | 83.8 % | +0.013 | −0.009 |
| mPFC | 72.6 % | +0.036 | +0.058 |

Nothing anywhere distinguishes pattern A from pattern B.

### Where it is lost: DG is sparse but not selective

§4 reports DG "pattern separation" as an active fraction of 2–4 %. That
measures sparseness only. Selectivity is a different claim, and it fails:

```
DG granule   3.28% active   Jaccard within 0.067   between 0.067   sep +0.000
CA3 SUP     98.0% active    Jaccard within 0.961   between 0.960   sep +0.000
```

0.067 is 4x chance overlap (chance = f/(2−f) = 0.017 at f = 3.28 %), so granule
activity is weakly reproducible — but within-pattern equals between-pattern to
three decimals. Restricting to the 7 epochs that replay the *same* pattern:

```
cells active at least once   14,124   (5.8x the 2,442 active in any one window)
fired in 1/7 windows         11,671   (83%)
fired in 7/7 windows              0
fired in >=6/7                    1
```

There is no core assembly. The same pattern recruits a nearly-disjoint granule
population on every replay. Counts 2–6 do exceed the binomial null (2,019 vs
798; 363 vs 23), so per-cell excitability biases the draw, but noise dominates.

### Why: the pattern-carrying input is 0.6 % of granule drive

Granule cells receive two excitatory sources — the EC LII perforant path, which
carries the replayed pattern, and a heterogeneous Poisson residual standing in
for unmodelled cortex, **resampled independently every window**. Per cell:

| source | mV/s | share |
|---|---|---|
| Poisson residual (noise) | 494.0 | 99.4 % |
| EC LII perforant path (signal) | 3.0 | **0.6 %** |

DG cannot be pattern-specific when 99.4 % of its drive is fresh noise. The
granule code is decided by Poisson shot noise, which is exactly the
participation statistics above.

Two compounding causes. First, `build_dg_module`'s budget comment assumes EC LII
fires at ~3 Hz (`EC 50 * ~3 Hz * 0.15 = 23`); after the §11 cortical sparsity
retune it fires at **0.40 Hz**, so the term is 3.0, not 23 — the retune cut the
signal share from ~4.4 % to 0.6 % without anyone noticing. Second, even the
design point was never enough to build a reproducible assembly.

This is not a parameter tweak, because the two drives are limited by different
things. EC arrives **synchronously** in SWR-locked bursts, so `K x w_ec_dg` must
stay under the 20 mV granule gap or every granule cell detonates (measured:
`K=50 w=0.4` saturates DG to 98.8 % by the second SWR). The Poisson residual
arrives **asynchronously**, so it dominates the mean-rate budget while staying
individually subthreshold. Raising the signal hits the synchrony ceiling long
before it wins the rate competition, and removing the noise reintroduces the
cold-start deadlock the residual exists to solve — DG has no drive until the
loop runs, and the loop cannot start without DG.

### Consequence for Test 3

A cortex that completes a pattern it cannot distinguish from any other pattern
is an attractor, not a memory. Any positive Test 3 obtained while §13 stands
demonstrates that mPFC has recurrent dynamics, **not** that a specific memory
became hippocampus-independent. Specificity has to be settled first; the
`K_rec` work in §12 is downstream of it.

## 14. Six more attempts at DG selectivity — all negative, one real bug found

§13 named the residual Poisson drive as the reason DG cannot be pattern-specific
and described it as "resampled independently every window." That is true of the
generator's *spike output* — a Poisson process, of course, fires different
timestamps each window — but not of its *rate parameter*: `build_dg_module`
draws each granule cell's residual rate **once**, at build time, and never
touches it again. A cell that happens to draw a high rate is the loudest cell
in every window for the rest of the run, which looks like independent
resampling in a raster plot but is not. This section documents finding and
partially fixing that gap, and five follow-on interventions that did not
restore selectivity.

### Neurogenesis, twice, both negative

Phase 8 (age-indexed intrinsic properties: young cells more excitable, less
inhibited, no learning) and Phase 10 (real Hebbian EC LII→GC plasticity,
restricted to neurogenesis cohorts after the full-population version proved
computationally infeasible — its one-time synapse cache cost 14–16 h of a 20 h
MN5 budget) were both tested as candidate DG selectivity mechanisms this
session, on the theory that a cohort tuned by recent experience could respond
differently to a held-back "oddball" pattern than to a familiar one. Neither
moved DG off chance, at 12 % scale, MN5, matched seed:

| condition | DG active | DG identity sep | DG timing sep |
|---|---|---|---|
| no neurogenesis (control) | 4.4 % | +0.001 | −0.002 |
| age-indexed neurogenesis (Phase 8) | 4.4 % | −0.023 | +0.009 |
| neurogenesis + cohort Hebbian learning (Phase 10) | 4.4 % | −0.023 | +0.003 |

An epoch-resolved oddball comparison (novel pattern C's active set vs. every
other pattern's, epoch by epoch) found no pattern-identity signature in either
condition — only a same-cells-recur-nearby-in-time artifact under
neurogenesis, consistent with §13's "no core assembly" finding, not with
novelty detection.

### The residual-rate bug, and what it actually costs

Two MN5 runs with different `--seed` shared 66 % of DG's "always-on" cells
(active in ≥8/14 epochs) — 632 cells in common against ~8 expected by chance.
Two independent causes, both silent regardless of `--seed`:

1. `build_dg_module`'s residual Poisson rate is drawn once, never refreshed
   (this section's headline finding).
2. `build_dg_module`'s own `seed_connect` defaulted to `42` unconditionally —
   the call site never threaded `--seed` through, so every run drew the
   *identical* granule V_m/rate heterogeneity regardless of seed.

Fix: re-randomize the residual rate every epoch
(`run_dg_residual_refresh_hook`, and the equivalent for each neurogenesis
cohort's own residual generator), and thread `--seed` into `seed_connect`.
Verified locally (1 % scale, matched seed, `--het 0.30`, 14 epochs):

| | before fix | after fix |
|---|---|---|
| cells active ≥8/14 epochs | (not measured at 1 % pre-fix) | 565 (expected ~4 by chance) |
| cells active in all 14 epochs | — | 10 (expected ~0) |

565 vs. 4 says the fix did not fully work with heterogeneity on. Repeating
with `--het 0` (no per-cell neuron-parameter draw) eliminated the excess
completely — 0 cells at every threshold, exactly matching chance. So a
**second**, independent source of persistent per-cell bias exists:
`--het`'s one-time `a/b/c/d/I_e` draw (30 % CV) is itself enough to rank
granule cells by fixed excitability for the whole run, and it was not touched
by this fix. This is not a bug — the heterogeneity is deliberate, tuned
elsewhere for realism and robustness (§8–9) — but it has the same practical
effect on DG as the residual-rate bug did.

Even with the stereotypy fully eliminated (`--het 0`), DG identity separation
was **still 0.000**. Removing the bias was necessary but not sufficient: a
fair, per-epoch-independent competition is still a competition decided by
noise, not by which EC LII cells are currently active, unless the signal is
actually strong enough to win it. That is the same 0.6 %-of-drive problem
§13 already named — the residual-rate bug just added a second, deterministic
distortion on top of it.

### Two attempts to shift the SNR, both negative

With the fix in place and `--het 0.30` restored (the tuned config, not
touched), two follow-up interventions tried to give EC LII's signal more
relative weight without disturbing §13's synchrony-ceiling constraint:

| config | DG active | DG identity sep | EC LII discriminates? |
|---|---|---|---|
| baseline (`w_ec_dg=0.6`, `pp_residual=0.9`) | 15.1 %† | −0.009 | timing only |
| `w_ec_dg=1.0` | **46.0 %** | −0.047 | none |
| `pp_residual=0.5` | 16.1 % | −0.007 | none |

† 1 % scale; not comparable to the 12 % 2–4 % target — see caveat below.

`w_ec_dg=1.0` reproduces exactly the failure mode §13 predicted: `K x w` is
close enough to the 20 mV granule gap that the closed EC→DG→CA3→CA1→EC loop
detonates, active fraction runs to 46 %, and EC LII's own established timing
signal is destroyed along with it — strengthening the signal broke the one
thing that was working. Cutting `pp_residual` to shrink the noise floor
instead left DG's active fraction essentially unchanged (15.1 % → 16.1 %),
which suggests the basket-mediated feedback loop clamps the active fraction
by *rate*, largely independent of the residual's amplitude — so shrinking the
residual did not shift the competition toward EC LII the way a simple
signal-vs-noise picture predicts, and it cost the one significant result
(EC LII's timing separation dropped out of significance). Both directions
tried; both failed; the second was actively regressive.

### Analysis: why the system sits here

- **The competition may not be input-selective at all.** Basket feedback
  keeps the active *fraction* in a narrow band, but nothing in its design
  targets which cells specifically fire — it looks like a rate clamp, not a
  content-addressable winner-take-all. `pp_residual=0.5`'s null result (active
  fraction barely moved) is consistent with this: the clamp defends a target
  rate, not a target identity.
- **Two competing, both-deliberate design goals are in tension.** `--het`
  heterogeneity is tuned for realism and robustness elsewhere in the model;
  DG pattern separation needs granule cells to be interchangeable enough that
  input, not fixed identity, decides who wins. Nothing in this session
  reconciles the two — they were tested as alternatives, not combined.
  A DG-scoped heterogeneity toggle (leave CA3/CA1 untouched) is untried.
- **The loop-gain ceiling caps how much signal-boosting is even safe to try.**
  Because EC LII→DG→CA3→CA1→EC LII/EC LV is a closed loop, any static
  increase to the DG-stage gain (`w_ec_dg`) risks amplifying around the whole
  loop, not just at DG — as observed. Any future signal-boosting attempt
  needs to raise EC LII's *effective* pattern-locked drive without raising
  its *loop* gain, e.g. sharpening EC LII's own place-field tuning, or timing
  the perforant-path kick more precisely into the SWR window, rather than
  scaling `w_ec_dg` further.
- **1 % scale is not a clean stand-in for 12 %.** `K_pp=50` is fixed
  regardless of `--scale`, so a granule cell's 50 perforant-path samples cover
  5 % of EC LII's pool at 1 % scale (1,000 cells) vs. 0.4 % at 12 % (12,005
  cells) — a very different effective sampling regime. All four bracket runs
  in this section ran at 1 % scale for turnaround speed; the residual-rate
  fix itself is scale-independent (it is about *when* a rate is drawn, not
  how many cells there are), but the two SNR-tuning results should be treated
  as hypotheses to re-check at 12 %, not confirmed at the scale that matters.

## 15. CA3's theta drive, DG's time-block instability, and a corrected EC LII → GC result

§14 left DG's selectivity failure with an open, mostly-negative picture.
Three more findings this session change that picture: one names a likely
cause of the already-documented weak replay-sequence correlation, one is a
new instability that neither §13 nor §14 accounted for, and the third
*corrects* one of §14's own conclusions — the "no identity signal, even at
the wiring level" reading of the granule-cell drive test does not survive
a closer look.

**Terminology note.** This codebase's "epoch" is not the ML sense (one pass
over a training set). It is one fixed-duration (`--epoch-ms`, 1000 ms
throughout this section) block of simulated time: a theta rhythm runs
across the whole block, and it contains two SWR-like replay events (forward
at +300–420 ms, reverse at +600–720 ms) plus one assigned pattern. This
section identifies each block by its absolute simulation-time range
(`t=9000–9999 ms`, ...) rather than a bare epoch index, to avoid trading on
the overloaded term.

### CA3 is not quiet between "replay events" — it never stops

The model wires an explicit 8 Hz theta drive onto CA3 SUP/DEEP/INT_SUP/
INT_DEEP and CA1 PYR/BASKET/OLM (`replay_scaled.py:1096–1106`,
`theta_hz=8.0`), continuously, every block — not gated by the SWR windows.
Single-block instantaneous rate (not the 14-block average, which smooths
this out) confirms it: CA3 SUP and CA3 DEEP both show a regular ~9–10 Hz
oscillation with peaks of 60–130+ Hz recurring roughly every 100 ms for the
*entire* 1000 ms block — before, during, and after the two labeled SWR
windows alike. This is a deliberate design feature (θ-rhythm modulation,
biologically reasonable), not raw runaway. But the SWR windows are not
distinguishable in amplitude from the surrounding background rhythm — a
plausible, previously-unstated explanation for §7's already-weak,
non-significant replay-sequence Spearman ρ: the sequence-locked signal the
model is trying to produce rides on top of a comparably-large background
rhythm, not standing out from it.

### DG's activity is not stable across blocks — four blocks run 4–9× hot

Restricting to the SWR-forward window specifically (the same window used
for every separation metric in §13–14), DG granule-cell active fraction
across 14 blocks (1 % scale, `replay_output_1pct/phase11_residual_fix_control.h5`):

```
t (ms)          active%   t (ms)          active%
    0 -   999     15.4%    7000 -  7999      7.7%
 1000 -  1999      3.2%    8000 -  8999      6.6%
 2000 -  2999      4.4%    9000 -  9999     32.2%   <-
 3000 -  3999      4.0%   10000 - 10999      5.2%
 4000 -  4999      3.5%   11000 - 11999      5.3%
 5000 -  5999      7.2%   12000 - 12999     33.1%   <-
 6000 -  6999      6.6%   13000 - 13999     27.3%   <-
```

Ten blocks sit in a plausible 3–8 % band. Four (t=0–999, 9000–9999,
12000–12999, 13000–13999 ms) run 4–9× hotter (15–33 %). DG basket cells
track the exact same four blocks (46 %/94 %/99 %/94 % active vs. a 22–33 %
baseline elsewhere) — the sparsifying feedback population is destabilized
right alongside the cells it is supposed to be sparsifying, not correcting
for it.

The four anomalous blocks do not share one cause:
- **t=0–999 ms** is a first-block-only phenomenon shared with CA3: CA3 SUP's
  peak instantaneous rate here (128 Hz) is the highest of any block, mirroring
  the extreme, isolated 239 Hz spike CA3 shows in the first few ms of the run
  — an initialization transient, not a recurring failure mode.
- **t=9000–9999, 12000–12999, 13000–13999 ms** show no corresponding
  anomaly in CA3 SUP/DEEP's own peak rate (all within the normal 65–90 Hz
  range at those same blocks) — whatever destabilizes DG here is not simply
  inherited from an equally-destabilized CA3. EC LII's own active fraction
  (same SWR-forward window) shows a slow upward drift across the run (10 %
  at t=1000–1999 ms rising to 25–30 % by t=6000–8999 ms) consistent with
  the STC hook's ongoing CA1→EC LII potentiation gradually raising EC LII's
  baseline excitability — plausibly making DG's sparse-coding regime
  increasingly prone to tipping into a denser state as the run progresses,
  though not deterministically (t=10000–11999 ms sit between two anomalous
  blocks and stay perfectly normal). Cause not fully isolated; flagged as
  an open item below rather than guessed at further.

### The EC LII → GC single-cell result, corrected

A follow-up to §14, same session, reconstructed the model's actual fixed
EC LII → GC wiring (600,000 synapses, exact GID match against real spike
data — see `reconstruct_connectivity.py`, extracted from that analysis) and
tested whether granule cells receiving more input from currently-pattern-
tuned EC LII cells are more likely to fire. Pooled across all 14 blocks
(168,000 cell-block observations), the result read as flat-to-negative — no
signal. Splitting by the block-health finding above changes that reading:

| grouping | n | active % | point-biserial r | p |
|---|---|---|---|---|
| all 14 blocks (pooled) | 168,000 | 11.56 % | −0.0173 | 1.5×10⁻¹² |
| healthy 10 blocks only | 120,000 | 5.39 % | **+0.0086** | 2.8×10⁻³ |
| anomalous 4 blocks only | 48,000 | 27.00 % | **+0.0165** | 3.1×10⁻⁴ |

Both subsets, analyzed within their own regime, show a small but
statistically real *positive* relationship between a granule cell's actual
wired-in pattern-tuned input and its firing probability (decile 1 → decile
10 of input drive: 5.0 % → 5.6–5.9 % active in healthy blocks; 25.8 % →
27–29 % in anomalous blocks). The pooled analysis inverted this sign — a
Simpson's-paradox artifact from mixing two regimes with very different
baseline activity levels and mean input drive, not evidence the wiring is
blind to identity.

**Corrected conclusion:** the EC LII → GC synapse *does* carry a small,
real, single-cell-level identity signal — it is just far too weak to
survive the coarser population-level Jaccard-overlap separation metric
(§13–14), and naive pooling across an unstable network can hide or even
invert it entirely. This does not overturn §13–14's headline finding (DG's
*aggregate* identity/timing separation is still ≈0.00, EC LII is still the
only population that reliably discriminates) — it refines *why*: the
signal is present but weak, not absent, and any future SNR-tuning attempt
(§14 tried two, both negative) should be evaluated against this smaller,
real effect rather than against a strawman of zero relationship.

### New tool

`reconstruct_connectivity.py` (repo root) extracts a projection's actual
fixed connectivity from `replay_scaled.py`'s own network-builder functions
without paying for a full simulation — drives the real build code up to
(but not through) the first `nest.Simulate()` call, using the same CLI
flags as a real run. Used here for EC LII → GC; documented to extend to
other projections. See its module docstring for the mechanics (namespace
`exec` split at `if __name__`, `nest.__dict__['Simulate']` patch — plain
`nest.Simulate = ...` is blocked by NEST's module `__setattr__`).

## 16. DG's time-block instability is chaotic, not novel-pattern- or schedule-driven

§15 found 3 of 14 blocks (t=9000–9999, 12000–12999, 13000–13999 ms) running
4–9× hotter than the rest and could not fully isolate a cause, noting only
that all three come after the novel pattern's debut at t=8000–8999 ms.
Four follow-up runs (same 1 % scale, same seed 202 unless noted, `--dg
--ec-lii --ec-lv --mpfc --n-swr 14 --stc --het 0.30 --het-wcomp 2.3
--w-ec-dg 0.6 --pp-residual 0.9 --dg-delay-jitter 4.0 --pattern-source
ec-lii --place-field-sigma 0.15 --ec-pattern-base-rate 20
--ec-pattern-peak-rate 800 --ec-pattern-weight 1.5` throughout) test that
hypothesis and overturn it.

**Reseed test.** `--seed 303`, otherwise identical to the original run.
The specific anomalous blocks are not reproduced: seed 303's are
t=7000–7999, 8000–8999, 9000–9999, 11000–11999 ms (30–61 % active) — only
t=9000–9999 ms is anomalous in both seeds. Every anomaly in *either* seed
still falls in t=7000–13999 ms, none in t=1000–6999 ms — at this point
consistent with a real, seed-independent vulnerable window whose exact
tipping block is seed-dependent.

**Two confounded "no novel pattern" attempts.** To test whether that
window is caused by the novel pattern specifically (vs. simply being the
second half of a 14-block run), two more runs were tried and both turned
out to test something else instead:
- `--n-patterns 3`, no `--novel-pattern-onset` flag at all → all three
  patterns cycle from t=0, which is *more* pattern-switching than the
  original run's own t=0–7999 ms (which only alternates 2 of 3 patterns
  before the novel pattern's debut) — not a "no novel pattern" condition.
- `--n-patterns 2` throughout → matches the original's t=0–7999 ms
  *alternation schedule*, but `--n-patterns` also sets how the 10 CA3
  sequence groups are partitioned (`replay_scaled.py:1273–1274`): 2
  patterns means each owns 5 groups instead of 3–4, so every pattern
  presentation recruits more of CA3 than in the original run. Not matched
  either.

Both are still useful as data: the first stayed noisy but bounded
(2.6–23.1 % active, no block over 3× the 9.35 % median); the second was
uniformly *worse* than the original from block 0 onward (4.2–67.3 %,
median 7.18 %, with its own anomalies at t=5000–5999, 6000–6999,
11000–11999, 13000–13999 ms) — more CA3 recruitment per pattern
destabilizes DG throughout the run, not just in some later window.

**The actual clean control.** `--n-patterns 3` (same CA3 group partition
as the original run) with `--novel-pattern-onset 999` — past the 14-block
run length, so the held-back pattern is never shown and the alternate-2-
of-3 schedule the original used for t=0–7999 ms simply continues for all
14 blocks. This is the correct isolation: identical seed, identical
per-block CA3 recruitment, the only difference is that the novel pattern
never happens.

| t (ms) | original (novel@8) | seed 303 | 3-pat, no onset | 2-pat only | **clean control** |
|---|---|---|---|---|---|
| 0–999 | 15.4 | 3.9 | 15.4 | 15.4 | 15.4 |
| 1000–1999 | 3.2 | 3.0 | 3.2 | 4.2 | 3.2 |
| 2000–2999 | 4.4 | 4.5 | 2.6 | 2.6 | 4.4 |
| 3000–3999 | 4.0 | 15.1 | 4.8 | 3.8 | 6.2 |
| 4000–4999 | 3.5 | 7.9 | 3.2 | 11.4 | 4.5 |
| 5000–5999 | 7.2 | 4.2 | 4.1 | **35.3** | 4.9 |
| 6000–6999 | 6.6 | 5.8 | 3.4 | **28.1** | **30.7** |
| 7000–7999 | 7.7 | **52.2** | 20.3 | 4.7 | 3.4 |
| 8000–8999 | 6.7 | **60.7** | 8.0 | 5.4 | 14.3 |
| 9000–9999 | **32.2** | **30.4** | 10.7 | 5.6 | 7.1 |
| 10000–10999 | 5.2 | 5.5 | 23.1 | 6.6 | 5.9 |
| 11000–11999 | 5.3 | **21.4** | 14.6 | **67.3** | 6.9 |
| 12000–12999 | **33.1** | 6.1 | 21.6 | 7.8 | **22.5** |
| 13000–13999 | **27.3** | 9.7 | 14.1 | **28.3** | 12.2 |

(bold = >3× that column's own median; DG GC active %, SWR-forward window,
same metric as §15.)

The clean control's first three blocks (15.4, 3.2, 4.4 %) are bit-for-bit
identical to the original run — same seed, same schedule, no novel
pattern has happened in either one yet, nothing has had a chance to
differ. The two trajectories still diverge starting t=3000–3999 ms (4.0 %
vs. 6.2 %) and by t=6000–6999 ms the clean control has its own runaway
block (30.7 %) exactly where the original stayed normal (6.6 %) — while
the original's own anomalies (t=9000–9999, 13000–13999 ms) are unremarkable
in the clean control (7.1 %, 12.2 %). t=12000–12999 ms is elevated in
both, but that is the only block-level overlap between the two runs
outside the shared initial transient.

**Conclusion.** All five conditions — original, reseed, both confounded
attempts, and the clean control — produce their *own* 2–4 wildly anomalous
blocks, never the same set twice, including between the clean control and
the original run it is nominally identical to for its first several
blocks. A cause that depended on the novel pattern, on pattern count, or
on elapsed time would have to produce the *same* divergence given the
*same* seed and the *same* schedule; it does not. This is the signature of
chaotic sensitivity in a system sitting close to a bistability threshold
(§15's STC-driven EC LII excitability drift already suggested the
threshold is real and slowly approached) — small, uncontrolled numerical
differences (most plausibly floating-point summation order under NEST's
multi-threaded kernel, which is not guaranteed reproducible run-to-run even
at fixed seed) get amplified into qualitatively different block-level
outcomes once the margin to that threshold is small enough. §15's
"possibly related to distance from the novel-pattern block" framing is
superseded by this: the novel pattern is not the trigger, it just happened
to sit near a block index that this run's particular numerical trajectory
tipped on.

This does not mean pattern count is irrelevant — it clearly sets the
*average* proximity to the tipping threshold (2-pattern-only ran hot
almost throughout; 3-pattern conditions mostly did not) — but it does not
determine *which* block tips. Those are two separate effects: a slow,
parameter-controlled one (how close to the edge the run sits on average)
and a fast, uncontrolled one (which specific block crosses it).

## 17. EC LII -> DG identity: found, explained, and restored by clustering the fan-in

§15 found a real but tiny single-cell identity signal at the EC LII -> GC
synapse (point-biserial r=+0.009 to +0.017) — real, but far too weak to
show up by eye in a raster, and the wrong sign once wrongly pooled across
health-heterogeneous blocks. This section explains *why* it is that weak,
and shows a bio-plausible wiring change that substantially restores it.

### Why one projection can lose identity: convergent random sampling, not a bug

The EC LII -> GC perforant path is NEST's native `fixed_indegree`: each of
the 12,000 granule cells independently draws K=50 sources, sampled
**uniformly at random from all 1,000 EC LII cells**, with no relationship
to place field (`replay_scaled.py`, `TARGET_INDEGREE["ec_dg_pp"]=50`, the
`fixed_connect` call in `build_dg_module`). Combined with the pattern's
tuning width (σ=0.15, so ~30-40% of the ring is "hot" for any active
pattern), basic sampling statistics explain the result: any GC's 50 random
sources are, in expectation, a representative cross-section of the *whole*
ring regardless of which pattern is active, so by the law of large numbers
nearly every GC ends up with nearly the same expected total drive no
matter what is being replayed. Only the K=50 sampling noise across
different GCs' specific draws carries any residual identity information —
exactly the tiny effect §15 measured. Average fan-out per EC LII cell:
K·N_post/N_pre = 50×12,000/1,000 = **600** — every EC LII cell talks to
~600 of the 12,000 granule cells, scattered uniformly across the whole
ring.

### A direct visual test, and a real methodology bug caught along the way

Pooling every SWR-forward-window spike across all 14 blocks and measuring
each firing cell's ring-distance from whichever pattern was actually
active at that moment gives a clean, model-free null: if place field had
no bearing on firing, this distribution is uniform on [0, 0.5] (density
2.0 flat) — a direct consequence of place fields being drawn uniformly on
a ring. EC LII's distribution clearly departs from that null (dense near
distance=0, thinning toward 0.5); DG's, pooled across all 14 blocks
(mixing the §16 healthy/anomalous regimes), sits slightly on the *wrong*
side of it (mean distance 0.273 vs. the null's 0.250) — DG cells farther
from the active pattern were, if anything, slightly more likely to fire.

Deriving each GC's own "effective place field" (needed to sort/analyze
DG at all, since GCs don't have one of their own — only their K=50 wired
EC LII inputs do) requires averaging those 50 inputs' place-field values.
The first pass used a plain arithmetic mean, which is wrong on a ring:
values near the 0/1 wraparound point average toward 0.5 instead of toward
0. Caught by checking the derivation directly — the fix (a circular mean,
`atan2(mean(sin θ), mean(cos θ))`) changed 64% of GCs' derived field
values by more than 0.1 ring-units. Every figure and number in this
section uses the corrected circular mean.

### Two experiments, and which one actually explains it

The averaging-washes-out-identity theory above makes a testable
prediction: reduce K so there is less to average over, and DG should
track EC LII's identity much more closely. `--dg-ec-cluster-sigma` is not
that test — reducing indegree from K=50 to K=1 (one random EC LII source
per GC, everything else unchanged, 1% scale, seed 202) is:

| condition | DG mean ring-dist. from active pattern | vs. null (0.250) |
|---|---|---|
| original: random K=50 | 0.273 | worse than chance |
| uniform K=1 (no averaging, still non-topographic) | 0.253 | ≈ chance |
| **clustered K=50** (σ=0.05, biased toward each GC's local neighbourhood) | **0.236** | clearly better than chance |

(EC LII itself: 0.217-0.223 in all three conditions — its own encoding
never changes; only how DG samples it does.)

K=1 mostly just removes the anti-correlation — consistent with "less
averaging, less washing" — but does not produce a real positive signal:
without any spatial bias, a single random sample is no more informative
than 50 averaged ones, just noisier. **Clustering** the sampling —
keeping K=50, but biasing each GC's 50 sources toward EC LII cells near
its own topographic location (a Gaussian kernel, σ=0.05, tighter than the
pattern's own σ=0.15) instead of sampling uniformly — is what actually
restores a real signal, and by a wider margin than K=1 reached. This is
the direct computational analogue of the user's proposed biological
mechanism: axons closer on the ring are more likely to synapse onto the
same dendritic-tree neighbourhood, i.e. real anatomical topography (e.g.
the entorhinal-dentate medial-lateral gradient, Dolorfo & Amaral 1998)
rather than the model's original all-of-EC-is-equally-reachable
assumption. See `figures/dg_diagnostic/clustered_wiring_identity_test.png`
for the full distribution comparison (not just the means above) and
`zoomed_pattern_identity_test.png` / `identity_loss_histograms.png` for
the single-window and pooled diagnostics that led here.

### Now in the codebase: `--dg-ec-cluster-sigma`

Landed as a real CLI option (not a throwaway script): `--dg-ec-cluster-sigma
SIGMA` (default 0.0 = old uniform-random behaviour). Requires `--dg
--ec-lii --pattern-source ec-lii` (needs the real place-field centres,
already computed for the pattern drive — no re-derivation). Implementation:
`clustered_fixed_connect()` (`replay_scaled.py`, next to `fixed_connect`) —
each GC gets a topographic centre via even tiling by creation index, then
draws its K sources with `np.random.choice(..., p=gaussian_kernel)`
instead of NEST's uniform `fixed_indegree`; one `nest.Connect` call per GC
(all_to_all, K sources -> 1 target). ~6-7s for 12,000 GCs at 1% scale;
scales linearly with N_gc (~12,000 GCs/s), so ~80-90s expected at 12%
scale (144,000 GCs) — negligible next to the run's own ~20min wall time,
nothing like the cohort-(-1) cache bug fixed earlier this session.

### The rest of the network has the same exposure — Schaffer collaterals worst of all

The mechanism above (uniform random convergent sampling washing out
identity by averaging) is not special to the perforant path. Every major
projection in the model was checked against the same criterion — how many
sources does a downstream cell sample, out of how many available, and is
the sampling topographically biased at all:

| projection | rule | K (indegree) | pre size (1%) | density | topographic? |
|---|---|---|---|---|---|
| EC LII -> DG GC | fixed_indegree | 50 | 1,000 | 5.0% | **yes, now** (`--dg-ec-cluster-sigma`) |
| DG GC -> CA3 SUP (mossy) | fixed_indegree | 15 | 12,000 | 0.13% | no |
| DG GC -> CA3 DEEP (mossy) | fixed_indegree | 8 | 12,000 | 0.07% | no |
| **CA3 SUP -> CA1 PYR (Schaffer)** | fixed_indegree | 2,640 | 2,640 | **100%** | no |
| **CA3 DEEP -> CA1 PYR (Schaffer)** | fixed_indegree | 660 | 660 | **100%** | no |
| CA3 SUP -> CA1 BASKET | fixed_indegree | 500 | 2,640 | 18.9% | no |
| CA1 PYR -> EC LII | fixed_indegree | 500 | 4,600 | 10.9% | no |
| EC LII -> EC LV | fixed_indegree | 20/cell | 1,000 | 2.0% | no |
| EC LV -> mPFC | fixed_indegree | 20/cell | 600 | 3.3% | no |

Schaffer collaterals stand out: **density 100%** means every CA1 PYR cell
receives from literally every CA3 SUP and CA3 DEEP cell — there is no
sampling at all, so no downstream cell can be distinguished by *which*
CA3 cells it hears from (they all hear from all of them). This is a more
severe version of the same problem DG had, and lines up exactly with the
already-documented collapse of CA3's real, measured spike-timing identity
signal (separation 0.167 ± 0.022) to ~0.00 by CA1 (§9, §13-14): with an
all-to-all projection, identity can *only* survive through timing/delay
heterogeneity, never through connectivity structure — which is why §9's
delay-heterogeneity and delay-aware-STDP attempts were the only kind of
fix that could possibly have worked on this hop, and why they still
weren't enough on their own.

**Not fixed in this pass.** Schaffer's density can't be lowered casually —
it is the dominant source of CA1's excitatory drive, tuned as part of the
model's whole E/I balance, and turning 100% density into a genuinely
sparse+topographic projection is a materially bigger change than DG's
perforant path (which was already explicitly budgeted as a small,
subthreshold contribution). Flagged as the highest-priority next
candidate, not bundled into this session's fix or the MN5 run below —
changing DG and Schaffer in the same run would make it impossible to
attribute any result to either one.

## 18. The clustered fix at full scale (JOB H8, 12%): DG's population-level identity moves off zero for the first time

JOB H8 (`run.sh`) ran §17's fix as a matched A/B pair at 12% scale (144,000
granule cells, 12,005 EC LII cells, same seed 202, otherwise identical
config) on MN5. Both arms finished cleanly, well inside the 20h QOS cap
(control 37,758s ≈ 10.5h; clustered 38,410s ≈ 10.7h) — DG's own build step
cost an extra ~209s for clustering (917s → 1,126s) at this scale, in line
with §17's estimate and nowhere near the cohort-(-1) cache bug (§ JOB H7).
Bulk network statistics (CA1/CA3/DG firing rates, active fractions) are
essentially unchanged between arms — exactly what a targeted, single-
variable intervention should look like.

The pipeline's own population-level Jaccard-overlap `pattern_discrimination`
table (not a reconstruction — printed by every run with `--n-patterns > 1`)
gives the first real answer at this scale:

| population | identity sep (control) | identity sep (clustered) | timing sep (control) | timing sep (clustered) |
|---|---|---|---|---|
| CA3 SUP | -0.000 | -0.000 | 0.001 | -0.001 |
| **DG GC** | **0.000** | **+0.004** | -0.009 | +0.005 |
| CA1 PYR | -0.003 | -0.000 | -0.003 | -0.001 |
| EC LII | 0.146 | 0.186 | 0.281 | 0.321 |
| EC LV | -0.011 | +0.011 | -0.005 | +0.021 |
| mPFC | -0.016 | **+0.046** | -0.021 | -0.031 |

**DG's population-level identity separation moves from exactly 0.000 to
+0.004** — the first time this metric has moved off zero for DG under any
of the seven manipulations tried across §13–17 (neurogenesis, cohort
Hebbian learning, the residual-rate fix, heterogeneity toggling, both
SNR-tuning directions, and now clustering). Small, but the *direction*
and rough scale match what §17's single-cell test predicted — this coarse
Jaccard metric is far less sensitive than that test, so a small nonzero
shift here is consistent with, not contradicted by, a real-but-weak
effect.

**A confound worth stating plainly, not glossing over:** mPFC's
separation flips from -0.016 to +0.046, which reads as exciting — but
EC LII's *own* separation also rose in the same run (0.146 → 0.186), and
since EC LII's construction is identical between the two arms (clustering
only touches the perforant path, downstream of EC LII), that rise is
itself a consequence of the closed EC→DG→CA3→CA1→EC loop echoing whatever
changed in DG back onto EC LII — not independent evidence that DG's fix
specifically is what reached cortex. With one seed per arm, "DG's fix
propagated forward" and "closed-loop amplification of any change,
regardless of its source" cannot yet be told apart. Resolving this needs
either a second seed at 12% (matching §16's seed-robustness method) or the
single-cell reconstruction test directly at 12% scale (below).

### The single-cell test could not be reconstructed locally — expected, not a bug

§17's more sensitive test (each firing cell's ring-distance from the
active pattern, at the single-cell level) needs the model's actual wired
EC LII→GC connectivity, which `reconstruct_connectivity.py` extracts by
building the network up to (not through) `nest.Simulate()`. Attempted
locally at 12% scale: the process died silently after the NEST startup
banner, no traceback, no output file — consistent with an OOM kill, not a
code error. 12% scale needs ~226M total synapses (JOB H8's own build log)
against this machine's 16GB RAM; this is the same class of problem that
moved this whole investigation to MN5 in the first place (see the
swap-thrashing note earlier this session). Not retried locally.

**Follow-up prepared, not yet run:** `run_reconstruct.sh` (new, repo root)
wraps `reconstruct_connectivity.py` in an MN5 sbatch job — same 12% config
as JOB H8, one arm with `--dg-ec-cluster-sigma 0.05` and one without,
`--out-hdf5` pointed at a throwaway path since the network never reaches
`Simulate()`. Network build itself costs ~1,050s (52s CA3/CA1/Schaffer +
~1,000s DG, per JOB H8's own timings), but the subsequent one-time
`nest.GetConnections(target=GC)` call needed to extract the wiring is NOT
well characterized at 144,000 targets — a local 1%-scale sanity check
(12,000 targets) took 288.6s for that call alone, and §16/JOB H7 already
found this exact call's cost scales with target population size in a way
that is not simply linear. `--time` is set generously (4h) for this
uncertainty; either way it is a fraction of JOB H8's ~10.5h full run,
since it only needs the connectivity, not a simulation:
```bash
sbatch --export=ALL run_reconstruct.sh                       # control
sbatch --export=ALL,CLUSTER_SIGMA=0.05 run_reconstruct.sh    # clustered
```

## 19. The single-cell test, replicated at full scale (JOB H9): clustering's effect holds, GID-mapping caveat included

JOB H9 (`run_reconstruct.sh`) finally completed after two fixes — dropping
`--ec-lv/--mpfc/--stc` (unneeded builds that stalled the first attempt for
85+ minutes) and querying `nest.GetConnections(source=ec_pop)` instead of
`target=gc_pop` (the second attempt's own stall: the same cost-tracks-
target-size pattern as the DG cohort-(-1) bug, now fixed by querying the
~12x-smaller EC LII side instead). Both arms finished in ~46 minutes each,
comfortably inside the 4h budget: network build to `Simulate()` ~1,800s,
then the source-query extraction ~950-970s (12,005 sources, 7,199,500
synapses each) — slower than 1%'s 6.5s, as expected at 12x the source
size, but nowhere near the multi-hour target-query cost it replaced.

**A GID-mapping subtlety, caught before trusting the result.** Dropping
`--stc` changes `n_epochs` (14→1), which changes how many place-field-
drive Poisson generators `main()` creates in Phase 7 — *before* the DG
module builds. That shifts every population's absolute NEST GID from
that point on: JOB H9's reconstructed GC population sits at GIDs
823385-967374, while JOB H8's real run has GC at 1135515-1279504 — a
312,130-GID gap, exactly `(14-1) × 2 × 12,005` (13 extra epochs × two
drive windows × EC LII's population size), confirming the mechanism.
EC LII itself is unaffected (nothing before its own creation differs
between the two configs), so its GIDs match exactly. Fix: join the
reconstructed connectivity to the real run's spike data by **positional
index within each population** (`gid - population's own minimum GID`),
not by raw GID — verified safe because both configs create GC as one
contiguous `nest.Create` block of identical size (143,990) immediately
following whatever came before, so the *i*-th GC cell created is the same
logical cell in both runs even though its absolute GID differs.

With that mapping in place, the same distance-from-active-pattern-center
test as §17 (all 14 blocks, SWR-forward window) gives:

| | 1% scale (§17) | 12% scale (JOB H9) |
|---|---|---|
| control (uniform K=50) | 0.273 (worse than null) | **0.251** (≈ null) |
| clustered (K=50, σ=0.05) | 0.236 (below null) | **0.226** (below null) |
| EC LII (reference) | ~0.22 | 0.177 (control) / 0.185 (clustered) |
| null | 0.250 | 0.250 |

The effect **replicates at full scale, and the gap from null is if
anything larger** (0.024 at 12% vs. 0.014 at 1%). It is also more
internally consistent than at 1%: the 12% control condition lands almost
exactly on the null (0.251) rather than measurably past it in the wrong
direction (0.273 at 1%) — closer to what "no identity signal" should
look like, with clustering then producing a clear, visible departure from
it in `figures/dg_diagnostic/clustered_wiring_identity_test_12pct.png`
(same layout as the 1%-scale figure: EC LII's own tuning curve is nearly
identical between arms, as expected — clustering only touches DG's
sampling — while DG's curve visibly rises near distance=0 only under
clustering). This single-cell result and JOB H8's population-level
Jaccard result (§18, 0.000 → +0.004) now agree in direction at the same
scale, from two independently-computed metrics.

## 20. Schaffer collaterals, clustered by CA3 group: a first pilot, no clean win yet

§17's flagged next candidate — Schaffer collaterals (CA3→CA1), 100% dense,
the most severe version of the identity-washing problem in the network —
gets its first real test here. Landed as two new pieces of infrastructure
(`grouped_fixed_connect`, `--schaffer-group-frac`), plus a genuinely
different design choice than DG's fix, explained below, plus a 1%-scale
pilot result that does **not** show a clean win the way §17/§19's DG fix
did — reported honestly rather than reframed as a success.

### Why this needed a different clustering axis than DG's fix

`clustered_fixed_connect` (§17) biases sampling toward a *continuous ring
position* — the right axis for EC LII, whose place fields genuinely sit on
such a ring. CA3 has no equivalent continuous coordinate; its pattern
identity lives in **sequence-group membership**, and those groups are
*deliberately interleaved by neuron index* (`patterns = [list(range(i,
n_seq_groups, n_patterns)) ...]`, chosen specifically so "a contiguous
split would let downstream cells discriminate on gross topography rather
than on assembly identity" — see `build_replay_network`'s own comment).
Clustering Schaffer by neuron-index proximity, the way DG's fix works,
would therefore cluster together cells from *different* patterns by
construction — close to meaningless, and arguably a reintroduction of the
exact shortcut the interleaving was designed to prevent.

`grouped_fixed_connect` (`replay_scaled.py`, next to `clustered_fixed_connect`)
clusters by **group identity** instead: each CA1 PYR cell is assigned a
"home" CA3 sequence group (creation index modulo `n_seq_groups` — CA1 has
no group of its own, so this is an arbitrary but well-defined and evenly-
distributed stand-in), then draws an expected `--schaffer-group-frac`
fraction of its (already-reduced via the existing `--schaffer-k`) in-degree
from that group specifically, the rest uniformly from the other groups.
Reuses `--schaffer-k`'s existing weight-compensation (fewer inputs,
proportionally stronger, preserving mean CA1 drive) — `--schaffer-group-frac`
is a pure sampling-bias layered on top, requires `--schaffer-k` to be set
(meaningless at the default 100% density, where every source is included
regardless of bias).

### A clamping caveat, caught before trusting the pilot

Verified against the real wired connectivity before running anything
further: at 1% scale (`--schaffer-k 500`, 10 groups, 264 cells/group),
requesting `--schaffer-group-frac 0.7` (350 of 500 sources from the home
group) is **unsatisfiable** — a group only has 264 cells to give, full
stop. The achieved purity was 264/500 = 0.528, confirmed via
`nest.GetConnections(target=<one CA1 cell>)` on four sample cells (all
exactly 264/500). Still a real, substantial bias — >5x the ~10% a
uniform-random draw would give a group this size — but the requested 0.7
would only be achievable at a larger scale (12%: 905 cells/group, comfortably
above K=500) or a smaller K.

### The pilot result: no clean population-level win, and larger swings than DG's fix ever showed

Two full 1%-scale runs (seed 202, otherwise identical to every other run
this session), `--schaffer-k 500` alone (uniform, reduced-but-unbiased)
vs. `--schaffer-k 500 --schaffer-group-frac 0.7` (clustered, 0.528 achieved):

| population | identity sep (uniform K=500) | identity sep (clustered, purity 0.528) | active % (uniform) | active % (clustered) |
|---|---|---|---|---|
| CA3 SUP | 0.000 | -0.000 | 99.9% | 99.9% |
| **CA1 PYR** | **-0.008** | **-0.006** | 96.8% | 97.6% |
| DG GC | -0.008 | -0.001 | 15.8% | 24.1% |
| EC LII | 0.048 | 0.003 | 25.7% | 31.8% |
| EC LV | 0.028 | 0.022 | 75.4% | 74.7% |
| mPFC | 0.027 | -0.021 | 40.0% | 32.9% |

CA1 PYR — the population this manipulation directly targets — moved
from -0.008 to -0.006: negligible, and still solidly in "no discrimination"
territory in both arms (both flagged `[FLAG] no population discriminates
the patterns by either code`). Unlike §17/§19's DG result (a clean,
reproducible move off exactly 0.000 at two independent scales), this does
not read as a real effect yet. Two things stand out as worth resolving
before drawing any conclusion, not glossed over:

1. **Active fractions swung far more between arms than DG's fix ever
   produced** — DG GC 15.8%→24.1%, EC LII 25.7%→31.8%, mPFC 40.0%→32.9%
   (§17/19's DG pilot kept every population's rate within ~1% of the
   control across both arms). Schaffer being CA1's dominant excitatory
   drive means reducing its in-degree by >5x (2640→500), even with weight
   compensation preserving the *mean*, plausibly changes higher-order
   statistics (synchrony, burstiness) enough to shift the whole network's
   operating point — a confound this pilot cannot separate from the
   clustering bias itself.
2. **EC LII's own identity separation dropped** (0.048→0.003) despite its
   own construction being unaffected by Schaffer's wiring directly — the
   same closed-loop-feedback signature already flagged as a confound in
   §18 (CA1→EC feedback via the STC hook), now working in the *opposite*
   direction from what would support a "Schaffer clustering helps"
   narrative.

**Not concluded either way.** This could mean the axis or parameters are
wrong (K=500 too aggressive a cut this early, `group_frac` too high once
clamping is accounted for, or CA1 needs its own within-group readout
structure DG didn't need), or it could mean Schaffer's identity-washing
problem needs a fundamentally different fix than "bias the sampling."
Candidates for a next pass, not yet tried: a gentler `--schaffer-k`
bracket (1000-1500 rather than 500) to isolate how much of the active-
fraction swing comes from the indegree cut alone vs. the group bias;
running the SAME uniform-vs-clustered comparison at 12% scale (where 0.7
purity is actually achievable, unlike this clamped pilot); and checking
whether the active-fraction swing itself is real signal or another
instance of the block-to-block chaotic sensitivity documented in §16
(a single whole-run average, exactly what §16 warned not to trust alone).

## Open items

- **Cortical selectivity is unsolved.** Pattern identity is robustly encoded in
  CA3 spike timing (0.167 ± 0.022) but does not reach cortex. Neither delay
  heterogeneity (§8) nor delay-aware Schaffer STDP (§9) transmits it; the one
  apparent positive failed to replicate across three seeds. Untested factors:
  much longer training, recurrent cortical targets, larger scale.
- **Ripple background is still epoch-0 only.** The replay drive (trigger +
  staggered scaffold) now repeats every epoch, so epochs 1…*n*−1 do contain
  real replay. The sharp-wave/ripple *background* does not: repeating it needs
  ~16 k `sinusoidal_poisson_generator`s per window (one per neuron across
  CA3+CA1), and NEST steps every generator at every timestep regardless of its
  start/stop — at 6 epochs that is ~211 k generators and a 1 % run had not
  finished after 3 h. Doing it cheaply needs
  `inhomogeneous_poisson_generator` (one node, scheduled rate profile).
- **EC LII→DG loop gain** is set by synchrony, not mean rate: EC fires in
  SWR-locked bursts, so K·w must stay well under the 20 mV granule
  rest→threshold gap or the loop saturates DG. Confirmed again in §14
  (`w_ec_dg=1.0` → 46 % active, detonation) — the ceiling is real at both
  scales tried.
- **DG selectivity at the population level has moved off zero, and the
  single-cell effect now replicates at full scale — but the cortical read
  is still confounded** (§13–19): six earlier attempts (age-indexed
  neurogenesis, cohort Hebbian learning, the residual-rate fix,
  heterogeneity off, both SNR-tuning directions) never moved DG's
  population-level identity separation off ~0.000. §17's clustered
  perforant-path fan-in does, and §19 confirms it holds at 12% scale with
  an *independently reconstructed* connectivity: DG's mean ring-distance
  from the active pattern goes from 0.251 (control, ≈ null) to 0.226
  (clustered) — the gap from null is if anything larger than at 1% scale
  (0.273→0.236). Population Jaccard (JOB H8) agrees in direction at the
  same scale: 0.000→+0.004. What is NOT yet resolved: JOB H8's downstream
  mPFC separation also improved (-0.016 -> +0.046), but EC LII's own
  separation rose in the same run (0.146 -> 0.186) despite identical
  construction in both arms, meaning the closed EC->DG->CA3->CA1->EC loop
  could be amplifying ANY change, not specifically propagating DG's fix
  forward — resolving this still needs a second seed at 12% or a direct
  cortical-layer version of §19's reconstruction test.
- **Schaffer collaterals (CA3->CA1), clustered by CA3 group: first pilot
  inconclusive, real confounds identified** (§20): a first attempt at
  sparsifying + topographically biasing Schaffer (`--schaffer-k 500
  --schaffer-group-frac 0.7`, clamped to an achieved 0.528 purity at 1%
  scale) did not move CA1 PYR's population-level identity separation in
  any convincing way (-0.008 uniform vs. -0.006 clustered, both still
  flagged as no discrimination) — unlike DG's clean, reproducible result
  (§17/19). Two real confounds, not yet disentangled: active fractions
  swung far more between arms than DG's fix ever produced (DG GC
  15.8%→24.1%, mPFC 40.0%→32.9%), suggesting the >5x in-degree cut alone
  shifts the network's operating point; and EC LII's own identity
  separation dropped (0.048→0.003) via the same closed-loop-feedback
  mechanism already flagged in §18, working against a "clustering helps"
  reading this time. Untried: a gentler `--schaffer-k` bracket to isolate
  the indegree-cut confound from the clustering bias itself, the same
  comparison at 12% scale where the requested purity is actually
  achievable (unlike this clamped 1% pilot), and checking whether the
  active-fraction swing is real signal or another instance of §16's
  block-to-block chaotic sensitivity (this pilot only looked at a
  whole-run average, exactly what §16 warned not to trust alone).
- **DG's activity is unstable across time blocks — confirmed chaotic, not
  novel-pattern- or schedule-driven** (§15–16): 2–4 of 14 blocks run 4–9×
  hotter than the rest in any given run, with DG basket cells destabilized
  in lockstep. §16 ruled out the novel-pattern-timing hypothesis with a
  clean matched-schedule control: even a run identical in seed and
  per-block CA3 drive to the original, differing only in that the novel
  pattern never occurs, still diverges from the original starting
  t=3000–3999 ms and produces its own, differently-placed anomalous
  blocks. Five conditions tested (reseed + 3 pattern-schedule variants +
  the clean control) never reproduce the same anomalous-block set twice —
  consistent with chaotic amplification of run-to-run numerical noise
  (plausibly NEST's multi-threaded, non-reproducible summation order) once
  the DG sparse-coding regime sits close to a bistability threshold set by
  the STC hook's slow EC LII excitability drift. Pattern count/CA3
  recruitment breadth sets how close to that threshold a run sits *on
  average* (2-pattern-only ran hot almost every block); it does not
  determine which specific block tips. Any future DG-focused run should
  report block-by-block activity, not just a whole-run average, and should
  not expect exact reproducibility even at fixed seed and schedule.
- **CA3's SWR windows are not distinguishable from background theta**
  (§15): the model's explicit 8 Hz theta drive onto CA3/CA1 produces
  60–130+ Hz peaks continuously, every ~100 ms, all block long — not just
  during the two labeled SWR windows. A plausible cause of the
  already-documented weak/non-significant replay-sequence ρ (§7). Untried:
  compare replay quality with theta on vs. off, or check whether the SWR
  windows' own drive is strong enough to produce a distinguishably larger
  event against this background.

## Reproducing

```bash
# 1% test: replay + DG, ~1 min
python replay_scaled.py --scale 1 --dg --dg-scale 2 --no-figures

# full stack with consolidation
python replay_scaled.py --scale 1 --dg --ec-lii --stc --ec-lv --mpfc --n-swr 6

# CA3 pattern completion (intact vs ablated)
python replay_scaled.py --scale 1 --pattern-completion

# figures from any output
python plot_consolidation_figure.py --in <file.h5>
python plot_pattern_completion.py  --in <pattern_completion.h5>
```

On MareNostrum 5 (see [`run.sh`](run.sh)):

```bash
sbatch --export=ALL,SCALE=12,DG=1,N_SWR=14 run.sh
```
