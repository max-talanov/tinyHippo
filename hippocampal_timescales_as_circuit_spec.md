# Hippocampal Timescales as a Circuit Specification


## 1. Consolidation timescales and their circuit constraints

Learning in this system runs on eight processes spanning nine orders of magnitude, from a single receptor gating event to a memory trace redistributing across cortex over months. What matters for hardware is not the molecular detail of each stage but three facts: which stages are fast enough to force analog/continuous-time circuitry, which are slow enough that a clocked digital design is free, and where the biological threshold behavior maps directly onto a specific circuit primitive.

![Eight consolidation stages plotted on one logarithmic time axis from 1 ms to 10 years — bar position marks onset, length marks characteristic duration, and stages visibly overlap rather than handing off in strict sequence.](timescale_axis.png)

*Fig. 1 — All eight consolidation stages on one log time axis. Full molecular detail (receptor subunits, kinase cascades, transcription factors) is in the companion consolidation-timescales document; what carries over to the hardware discussion below is just the numbers.*

| Stage | Time constant | What it forces on a circuit |
|---|---|---|
| Neurotransmission (AMPA/NMDA gating) | ~0.5–5 ms | Sets the fastest timescale in the whole system — the floor for any front-end comparator bandwidth. |
| Short-term plasticity | ~10 ms–1 s | NMDAR's slower deactivation is *why* a ±20 ms coincidence window exists at all — a real biophysical constant, not a modeling convenience. |
| Ca²⁺/kinase cascades | ~1 s–5 min | The "weight update decision" happens on this timescale; anything reading it out only needs to sample in the low-second range. |
| AMPAR trafficking / early-LTP | ~1–30 min, decays over 1–3 hr | A real, decaying analog state — this *is* the tag described below. |
| Synaptic tagging & capture | tag set in 1–2 min, decays τ≈1–4 hr | The pivot point: a local, transient, decaying signal that is either captured or lost — directly analogous to a volatile-memristor relaxation state (§2). |
| Transcription / late-LTP | onset ~30–60 min, stable by ~4–8 hr | Once captured, this state should stop decaying — a discrete regime change, not a continuous process. |
| Sharp-wave-ripple replay | ripple 100–300 ms, recurs over hours–days | Compressed replay packs sequence elements only a few milliseconds apart — this is the number that actually sizes circuit timing budgets in §4 of the companion hardware document (`neuron_model_optimization.md`). |
| Systems consolidation | days–weeks (rodent) to months–years (human) | Fully clocked-digital territory; no continuous-time requirement whatsoever. |

## 2. Synaptic tagging and capture as a volatile-memristor circuit

The synaptic tagging-and-capture (STC) mechanism described in the hippocampal plasticity literature has a clean circuit reading: a tag is set by coincident activity, decays with a fixed time constant if nothing happens, and is *captured* — converted to a stable, non-decaying state — only if a separate signal (biologically, a plasticity-related-protein pool) crosses threshold before the tag decays. This is close to the operating description of a **volatile threshold-switching memristor**: a device that transitions to a low-resistance state on a triggering pulse and spontaneously relaxes back over an intrinsic retention time — unless something latches it into a non-volatile state first.

![Synaptic weight versus successive sharp-wave-ripple events. Before event 4 the weight jumps and partly decays each cycle. At event 4 the plasticity-related-protein pool crosses threshold, decay stops, and weight climbs in a clean staircase — normal capture. A dashed counterfactual trace shows the same jumps but with capture blocked: weight oscillates at a flat baseline and never consolidates.](weight_consolidation.png)

*Fig. 2 — The tag-decay/capture dynamic, and the falsification test it implies. Both traces receive identical STDP-driven jumps per event (replay quality is unaffected either way); they diverge only in whether the tag gets captured before it decays.*

The dashed trace is the predicted signature if capture is blocked entirely: the weight distribution stays flat while replay quality (measured independently) stays unchanged, separating the replay mechanism from the consolidation mechanism. This separation defines the test for a candidate memristive tagging device: drive it with the same input statistics and check whether blocking "capture" produces the same flat, non-accumulating signature.

This section specifies the tag's *time-domain* behavior — onset, decay, capture. §4.2 specifies a second, independent axis, tag *amplitude*, using the `V_seg` state variable introduced there.

## 3. The dendritic-convolution pipeline architecture

![Reference circuit topology for the dendritic-convolution pipeline: per-synapse threshold units feed a two-level hierarchy of OR-gated refractory junctions (each level with its own 5–10 ms τ_rf), converging on a soma that also receives a direct, dendritic-bypass inhibitory input, followed by a fixed-delay axon stage.](dendritic_convolution.png)

*Fig. 3 — High-level neuron architecture, shown as two layers over the same physical topology: (A) synapse and dendritic-spike generation, and the timescales of dendritic-spike propagation along the dendritic tree; (B) synaptic homeostasis, short-term plasticity, and neuromodulation-driven tagging — through E-LTP, L-LTP, and replay — to memory consolidation, each mapped to the neuronal compartment where it occurs.*

### 3.1 Layer A — dendritic-spike timescales

Signals move through three physical stages, each with its own characteristic delay: synaptic areas generate dendritic spikes in about 1 ms; two levels of dendritic junctions integrate those spikes over 5–10 ms per level; the soma converts the result into an output spike over a further 5–10 ms. The companion hardware document treats this as a three-stage information-reduction pipeline — postsynaptic threshold, OR-gated refractory junction, somatic summation — and each stage is a candidate site for replacing an expensive analog circuit with a cheap thresholding one (§3.3).

### 3.2 Layer B — learning mechanisms mapped to the architecture

The same three compartments host the learning mechanisms from §1 and §2:

- **Synaptic areas** receive two distinct inputs — excitatory drive and a separate neuromodulatory input — and are where synaptic homeostasis (activity-dependent weight decay), short-term plasticity, and neuromodulation-triggered tagging take place.
- **Junctions** implement the OR-plus-refractory primitive listed in the table below.
- **The soma** is where E-LTP and L-LTP resolve for a single neuron; an ensemble of somas firing together implements sharp-wave-ripple replay, and repeated replay across the network implements systems-level memory consolidation (§4.5).

### 3.3 Stage-by-stage feasibility table

| Stage | Circuit primitive | Feasibility | Main open risk |
|---|---|---|---|
| Postsynaptic threshold (dSpike) | Memristor + local comparator | High — prototype-ready | Conductance drift shifting the threshold near the decision boundary |
| Junction (OR + refractory) | Wired-OR + edge-catcher + counter FSM, or a volatile memristor | High | Volatile-memristor relaxation-time variability, if used in place of a digital counter |
| Synaptic clustering | Non-uniform crossbar floorplan / reconfigurable interconnect | Moderate — the binding constraint | Needs 3D memristive/CMOS integration to be biologically faithful; not yet a mature foundry process |
| Somatic E/I summation | Differential integrate-and-fire | High — already standard (Loihi, BrainScaleS, TrueNorth-class cores) | None specific to this proposal |

Three of the four stages are near off-the-shelf. The junction/tag stage is the exception: it is the hardest stage, and the one where biology and device physics align most directly. §2 and §4.4 specify it in detail.

## 4. Implementation

This section specifies the implementation: a block-by-block description of the reference topology, an extension giving stage 1 a graded rather than purely boolean output, and a materials candidate for the tag/capture element.

### 4.1 Reference circuit topology, block by block

The figure specifies a concrete two-cluster instance, described block by block below.

**Excitatory side.** `Cluster1` groups six excitatory synapses (`ex_syn0`…`ex_syn4`, `ex_synN`); `Cluster2` groups three more (`ex_syn0`, `ex_syn1`, `ex_synN`) — two independent dendritic segments, sized differently, which is itself a statement that the pipeline does not assume uniform fan-in per branch. Every excitatory synapse feeds its own per-synapse `thr` unit in the leftmost, **1 ms** column — the stage-1 postsynaptic threshold from the table above, instantiated once per synapse, not shared. Within `Cluster1`, two of the six threshold outputs (`ex_syn2`, `ex_syn3`) are drawn joining on a short local wire before entering the junction; the other four converge on the same junction along separate wires. This is a wiring-topology detail, not a functional one — all six thresholded outputs feed the same first-tier junction, which is a plain wired-OR (§4.2 of the companion document): the diagram's local pairing of two wires versus four individual ones has no effect on the OR's logic, only on layout.

**Multi-level junction hierarchy.** The junction stage generalizes to `N` tiers: each tier's junctions take the previous tier's `jSpike` outputs as their inputs, applying the same OR-plus-refractory primitive at every level, with tier count set by the dendritic-tree depth to be modeled. The reference diagram specifies the `N = 2` case as a concrete, minimal instance: each cluster's thresholded outputs converge on its *own* first-tier `junction th/rp`, in the first **5–10 ms** column — one junction instance per cluster, each carrying an independently-settable `τ_rf` in that range — and both first-tier junctions' outputs converge on a single second-tier `junction th/rp`, in the second **5–10 ms** column. The same `th/rp` primitive appears three times in this two-tier instance (twice at tier 1, once at tier 2) with identical internal structure and independently tunable timing per instance — the reusable-leaf-cell design specified in the companion document, which extends to any `N` by replicating the leaf cell rather than redesigning it. The worst-case synapse-to-soma latency contributed by the hierarchy is `N` stacked `τ_rf` periods; for the diagram's `N = 2` at 5–10 ms per level, this is 10–20 ms.

**Inhibitory bypass.** `Cluster3` (`in_syn0`, `in_syn1`, `in_synN`) connects directly to the `Soma th/rp` block with a single wire each, passing through no `thr` unit and no junction at all. This is the diagram's explicit statement that fast, proximal/perisomatic inhibition is architecturally exempt from the threshold-and-refractory pipeline that every excitatory input must pass through — consistent with basket-cell-type inhibition acting close to the soma rather than being integrated dendritically.

**Somatic stage and output.** The second-tier junction's output and `Cluster3`'s direct inhibitory lines both arrive at `Soma th/rp`, which is the same threshold-plus-refractory primitive again, just with a different fan-in, a different threshold, and (presumably) a different `τ_rf`, implementing the differential E/I summation from the feasibility table. Its output feeds a final `axon delay` block, drawn as a separate stage outside the soma and outside all `τ_rf`-scale circuitry — a fixed propagation delay applied only after the somatic decision has already been made, decoupling millisecond-scale dendritic computation from simple downstream routing latency.

### 4.2 Graded synaptic-potential state per segment (`V_seg`)

Each `thr` unit reduces its segment's drive to a single bit the instant it crosses threshold, discarding the graded depolarization that led up to that crossing. But NMDA-receptor conductance is voltage-dependent (Mg²⁺-block relief follows a sigmoid of local depolarization; Jahr & Stevens, 1990), so the calcium influx driving tag-setting in §2 scales continuously with subthreshold depolarization rather than with a single all-or-nothing spike — consistent with voltage-based plasticity models (Clopath, Büsing, Vasilaki & Gerstner, *Nat. Neurosci.* 13:344–352, 2010) and the calcium-control hypothesis (Lisman, 1989; Shouval, Bear & Cooper, 2002): induction *strength*, not only induction occurrence, sets the outcome.

`V_seg` is one continuous state per dendritic segment (per `thr` unit, not per synapse), a leaky integration of the segment's synaptic input carried alongside the existing boolean `thr` output:

```
τ_v · dV_seg/dt = −V_seg + Σ_i w_i · s_i(t)
```

with `τ_v` in the 10–30 ms range already assigned to short-term plasticity in §1. A sigmoid gain `g_Ca(V_seg) = 1/(1 + K·exp(−λ·V_seg))`, patterned on the NMDAR unblock curve, converts `V_seg` into a tag-induction amplitude: `tag_amplitude ∝ g_Ca(V_seg) · [dSpike fired]`. This leaves §2's time-domain claims (`τ_tag`, the capture gate, the falsification test) unmodified — only the tag's *initial height* becomes graded instead of fixed.

Two consequences follow. First, because `V_seg` is shared across all synapses on one segment, several individually subthreshold synapses can jointly drive it into the steep part of `g_Ca` without any one of them firing a dSpike — giving the clustering argument in §3.3's table a mechanistic channel, voltage-dependent cooperativity, alongside its wiring-based one. Second, `V_seg` is one added analog block per `thr` unit (a leaky-integrator capacitor plus a differential-pair sigmoid stage, both standard sub-threshold-CMOS primitives) driving a graded-amplitude SET pulse into the same volatile memristor proposed as the tag element in §2, instead of a fixed-height pulse. The open risk is whether that memristor's SET response is graded over a useful amplitude range — parallel to the `τ_rf` variability already flagged for the junction stage, but along the amplitude axis. It predicts a companion falsification test: dose-dependent partial NMDAR blockade should shrink captured-weight step size continuously, rather than switching captured synapses to uncaptured all-or-nothing.

### 4.3 Passive EPSP current versus dendritic spikes in somatic spike generation

Both `thr` and `V_seg` carry dendritic-spike-related current to the soma. A second, parallel pathway is passive electrotonic spread of the EPSP via axial current, independent of any dendritic spike.

A single spine-level passive EPSP reaching the soma is 0.1–2 mV, against a 15–20 mV gap from rest to somatic threshold — roughly 1–5% of what firing requires per event, so reaching threshold this way alone needs summation across tens to hundreds of quasi-coincident inputs. A dendritic spike delivers 5–15 mV per event, one to two orders of magnitude more — which is why the pipeline digitizes the `thr` threshold-crossing event rather than the raw EPSP waveform. Dendritic democratization (Magee & Cook, 2000; Andrasfalvy & Magee, 2001) means unitary conductance scales with distance from the soma, so a passive, non-boosted EPSP stays a graded, spatially-distributed, always-on contribution regardless of whether any given segment's `thr` fires.

In the topology as specified (§4.1), every excitatory synapse routes through its segment's `thr` unit before reaching the junction hierarchy or the soma; `Cluster3`'s inhibitory line is the only bypass, justified by perisomatic targeting, and has no excitatory counterpart. The pipeline represents coincident threshold-crossing input (`thr`, junctions) and cooperative subthreshold drive toward tag induction (`V_seg`, §4.2) — it does not represent subthreshold input summed linearly toward the somatic threshold itself.

For this pipeline's target function — coincidence detection and timing (the ±20 ms STDP window, tag/capture, SWR replay-order fidelity) — the dendritic spike is the dominant driver of somatic firing, and the boolean `thr` abstraction is the right compression. A graded, rate-coded excitatory contribution to the somatic decision, independent of any single coincidence event, is not represented here; adding it would need a third input to `Soma th/rp` — a low-pass-filtered sum of excitatory `V_seg` states — which is outside `V_seg`'s scope as specified in §4.2 (tag induction only).

### 4.4 Tag-element implementations: digital and non-digital

§2 and §4.2 specify the tag element at the level of circuit primitives, independent of device technology. §3.1's junction stage specifies two implementation options for its own `τ_rf` timer — a wired-OR/counter FSM or a volatile memristor — and the tag element admits the same choice. §4.4.1 specifies the digital option; §4.4.2 specifies a non-digital (memristive) candidate under active development.

#### 4.4.1 Digital implementation

A fully digital tag/capture circuit uses: a multi-bit register per tag site, holding the tag amplitude; a down-counter clocked by a slow divider, decrementing the register at a rate set to reach zero after `τ_tag` (an 8–12 bit register decremented roughly once per 1–15 minutes covers the `τ_tag ≈ 1–4 hr` range); and a digital comparator that checks the register against zero before each decrement. Capture is a single conditional write: if the PRP-threshold signal (already sampled at low-second resolution per §1's table, well within reach of a digital sampler) crosses its set point while the register is still nonzero, the register's current value is copied into a separate, non-decrementing capture register and the down-counter is disabled for that site; if the PRP-threshold signal has not crossed by the time the register reaches zero, the tag register clears and no capture occurs. `V_seg`'s graded tag amplitude (§4.2) sets the register's initial loaded value rather than a fixed constant, and the falsification test in §2 (flat weight distribution when capture is blocked, replay quality unaffected) applies unchanged by disabling the conditional-write path. This circuit uses only standard cells — register, down-counter, comparator, clock divider — with no memristor, retention variability, or device characterization; each site's `τ_tag` is set by a clock-divider ratio, deterministic and identical across sites. The cost is area: one register, counter, and comparator per tag site, versus one two-terminal memristor in the non-digital route of §4.4.2.

#### 4.4.2 Non-digital implementation: copper-aspirinate memristors

Caus, Sławek, Mazur, Zawal, Baś, Szaciłowski, Talanov & Abdi (2026, "The memristive implementation of the hippocampus: a hypothesis") report a concrete candidate for the volatile threshold-switching memristor specified as the tag element in §2, built and electrically tested. The material is polycrystalline copper(II) bis-aspirinate, [Cu₂(asp)₄], spin-coated as a thin layer on ITO and capped with a sputtered copper electrode; three axially-ligated derivatives — [Cu₂(asp)₄(py)₂] (pyridine), [Cu₂(asp)₄(bimi)₂] (benzimidazole), and [Cu₂(asp)₄(DABCO)₂] — were prepared from the same parent complex by adding an axial N-donor ligand. All but the DABCO derivative show pinched I–V hysteresis loops under cyclic voltammetry, the standard electrical signature of memristive switching. The axial ligand tunes two properties: conductivity (the pyridine and benzimidazole derivatives conduct roughly two orders of magnitude more than the unligated parent) and retention. In chronoamperometric retention testing, the unligated parent's low-resistance state decayed to the high-resistance state within about 50 minutes (volatile); the benzimidazole derivative's high- and low-resistance states remained stable, essentially noise-free, after 6 hours (non-volatile on the timescale tested).

This volatile-versus-latched split, produced by changing only the axial ligand on one underlying complex, brackets the `τ_tag ≈ 1–4 hr` range in §1's table and provides the two states the tag/capture analogy in §2 requires: a device that relaxes on its own (parent complex, minutes-scale retention) and a device that holds once switched (benzimidazole derivative, hours-plus retention) — a materials-level realization of "tag decays unless captured." The paper's I–V and retention data characterize switching and retention, not conductance-step-versus-pulse-amplitude behavior; the graded-SET behavior specified in §4.2's `V_seg` extension is not demonstrated by this data and remains an open device-characterization question. The same paper's methylammonium-lead-iodide perovskite devices, modified with graphene oxide, fullerenol (C₆₀(OH)), or multiwalled carbon nanotubes, show measurable STDP-like potentiation directly in the memristive response, a second materials route toward stage-1's threshold-plus-plasticity behavior. The same paper also proposes a random-junction stochastic-network framing (percolating Ag–Ag₂S and SWNT/Por-POM meshes) as a physical analogue for the probabilistic, disordered connectivity in the clustering stage.

### 4.5 Memory consolidation: replay as a volatile-to-non-volatile handoff

§1's systems-consolidation row and §4.4.2's device data point at the same architecture from two directions: a hippocampal trace is fast, local, and volatile; a cortical trace is slow to form but effectively permanent. Replay (§1, SWR row) is the mechanism that drives one into the other, and the two memristor technologies proposed as tag elements above map onto the two ends of that handoff by more than analogy.

**Fast tier: organic volatile memristors.** The tag/capture element in §2 and §4.4.2 is deliberately a device that relaxes on its own unless captured — matching Victor Erokhin's organic (polymeric) memristors, which switch and decay on short, biologically-relevant timescales and, being solution-processed, can be additively 3D-printed rather than requiring foundry fabrication. That fabrication route trades device count for reconfigurability: organic memristors are cheap to print in small numbers at synapse-like density, which is exactly the local, per-tag-site role §2 and §4.4 assign them, not a route to the device counts a cortex-scale store needs.

**Slow tier: inorganic non-volatile memristors, at scale.** Systems consolidation redistributes one hippocampal engram across a much larger, distributed population of cortical synapses — a fan-out, not a copy. Themis Prodromakis's inorganic (metal-oxide) memristor crossbars are the complementary device for that side: individually slower to switch and non-volatile once set, but fabricated at silicon densities reaching on the order of 10⁶ devices per chip. What this tier contributes is not per-device sophistication but sheer number — the same property that lets a sparse hippocampal representation be redistributed across a large population of stable cortical sites.

**Replay as the handoff mechanism.** Each SWR event (§1: 100–300 ms, recurring over hours–days) replays a compressed version of a stored sequence, re-driving the same volatile organic tag sites that stored it originally. Repeated replay is the network-level equivalent of the PRP-threshold capture signal in §2: each pass is one more SET pulse toward a target inorganic memristor, and once enough passes have accumulated, that cortical-side device latches into its own non-volatile state — the same discrete regime change §1 assigns to transcription/late-LTP, relocated from a single synapse to a population of chip-scale devices. Before enough replay has occurred, the trace exists only in the fast organic tier and is lost if it decays uncaptured; after, it persists in the slow inorganic tier independent of the organic device's state, which is the hardware reading of hippocampal-to-cortical transfer, not just a shared vocabulary of "volatile" and "non-volatile."

**Process-to-device mapping.** Laid against §1's eight consolidation stages, each process falls on one side of the handoff or on the boundary between them:

| Process (§1) | Time constant | Quick tier — organic volatile memristor (Erokhin) | Slow tier — inorganic non-volatile memristor (Prodromakis) |
|---|---|---|---|
| Neurotransmission (AMPA/NMDA gating) | ~0.5–5 ms | Upstream of any memristive state — sets comparator bandwidth only | — |
| Short-term plasticity | ~10 ms–1 s | Drives `V_seg`/dSpike dynamics (§4.2); faster than any memristor switching used here | — |
| Ca²⁺/kinase cascades | ~1 s–5 min | Sets tag-induction amplitude (`tag_amplitude`, §4.2) — the SET pulse into the device | — |
| AMPAR trafficking / early-LTP | ~1–30 min, decays 1–3 hr | The device's own relaxing low-resistance state (§2, §4.4.2 parent complex) | — |
| Synaptic tagging & capture | tag set 1–2 min, decay τ≈1–4 hr | Same device; PRP-threshold crossing is the capture event | Capture can latch locally into the device's non-volatile derivative (§4.4.2 benzimidazole) — still per-synapse, not yet the cortical array |
| Transcription / late-LTP | onset 30–60 min, stable 4–8 hr | — | Locally captured state is now stable — the first candidate SET pulse toward the inorganic array once replay begins |
| Sharp-wave-ripple replay | 100–300 ms, recurs over hours–days | Re-drives the organic tag site on every pass | Receives one accumulating SET pulse per replay event — the fan-out source |
| Systems consolidation | days–weeks (rodent) to months–years (human) | Organic device's role ends once captured and sufficiently replayed | Non-volatile inorganic crossbar — the permanent, distributed store (10⁶ devices/chip) |

The quick tier covers everything up through capture at a single synapse; the slow tier only enters once replay starts driving a cortical-side device toward its own threshold — which is why systems consolidation (days to years) is orders of magnitude slower than tagging and capture (minutes to hours): it is rate-limited by how often replay revisits a given trace, not by any single device's intrinsic switching speed.

**Open risk.** This is an architectural hypothesis, not a demonstrated circuit: it requires a working interface between an organic volatile array and a much larger inorganic non-volatile array, with the replay-driven SET-accumulation rate on the inorganic side calibrated against the hours-to-days systems-consolidation timescale in §1's table — a two-technology integration problem on top of the single-device characterization risks already flagged in §4.2 and §4.4.2.

---

*Full technical detail: [Consolidation Cascade](https://claude.ai/code/artifact/f879468e-00e8-4ca1-9acc-348b12cc7e33) (molecular timescales) and `neuron_model_optimization.md` (hardware feasibility).*