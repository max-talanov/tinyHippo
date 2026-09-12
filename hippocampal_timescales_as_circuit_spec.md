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

![Reference circuit topology for the 
dendritic-convolution pipeline: per-synapse 
threshold units feed a two-level hierarchy of 
OR-gated refractory junctions (each level with 
its own 5–10 ms τ_rf), converging on a soma 
that also receives a direct, dendritic-bypass inhibitory input, followed by a fixed-delay axon stage.](dendritic_convolution.png)

Fig. 3 - **The high-level architecture** of the 
neuron implementation taking into account: (**A**)
synapses and dendritic spikes generation in 
synaptic area, dendritic spikes propagation 
along dendritic tree with timescales of 
dendritic spikes processing.
(**B**) As well as the synaptic homeostasis, 
STP, neuromodulation → synaptic tagging; E-LTP,
L-LTP, replay/repetitions as we ll as memory 
consolidation and their  identification linked 
to their position in the neuron architecture.

The diagram has two layers
1) **(A)** The temporal dynamics of the 
**dendritic 
   spikes propagation** and through dendritic 
   tree on the level of synaptic areas, 
   dendroids junctions and soma and the 
   timeframes of dendritic spikes processing: 
   synaptic areas (1ms), junctions (5-10ms), 
   generation of somatic spikes (5-10ms).
2) **(B)** The learning mechanisms described in 
   sections: **Consolidation timescales** and 
   **Synaptic tagging** mapped to the neuronal 
   areas. **Synaptic areas**: synaptic homeostasis 
   (weights decay due to time); neuromodulation 
   → synaptic tagging; short term plasticity. 
   **Soma**: E-LTP, L-LTP, replay and memory 
   consolidation on the level of network.  

**(A)** The companion hardware document frames 
dendritic 
processing as a three-stage information-reduction 
pipeline — a postsynaptic threshold, an OR-gated 
refractory junction, and a somatic summation each 
stage a candidate site for replacing an expensive 
analog circuit with a cheap thresholding one.

**(B)** Indicate the two inputs synaptic 
devices with excitatory input and 
neuromodulatory input for the groups of the 
synaptic devices, that triggers the tagging 
processes; junction implemented as 
described above; soma implements the E-LTP, 
LTP, while ensemble of somas implement replays,
and system memory consolidation. 

### 3.1 Stage-by-stage feasibility table

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

Every `thr` unit described above reduces its segment's synaptic drive to a single bit the instant it crosses threshold. The graded depolarization before that crossing does not propagate into the junction hierarchy or into the tag/capture mechanism of §2. NMDA-receptor conductance is voltage-dependent (Mg²⁺-block relief follows a sigmoidal function of local depolarization; Jahr & Stevens, 1990), so the calcium influx driving the tag-setting kinase cascade in §2 scales continuously with subthreshold depolarization rather than with a single all-or-nothing spike event. Voltage-based plasticity models (Clopath, Büsing, Vasilaki & Gerstner, *Nat. Neurosci.* 13:344–352, 2010) specify plasticity magnitude as a continuous function of postsynaptic voltage rather than of spike timing alone, consistent with the calcium-control hypothesis (Lisman, 1989; Shouval, Bear & Cooper, 2002): induction strength, not only induction occurrence, sets the outcome.

`V_seg` is one continuous state per dendritic segment (per `thr` unit in the diagram above, not per synapse and not per junction), defined as:

```
τ_v · dV_seg/dt = −V_seg + Σ_i w_i · s_i(t)
```

a leaky integration of the segment's synaptic input, carried *alongside* — not instead of — the existing boolean `thr` output, with `τ_v` in the same 10–30 ms range already assigned to short-term plasticity in §1's table. A sigmoid gain `g_Ca(V_seg) = 1/(1 + K·exp(−λ·V_seg))`, patterned on the NMDAR unblock curve but fit rather than copied, converts `V_seg` into a tag-induction amplitude: `tag_amplitude ∝ g_Ca(V_seg) · [dSpike fired]`. This changes nothing about §2's time-domain claims — `τ_tag`, the PRP-threshold capture gate, and the flat-weight-distribution falsification test all stand unmodified — it only makes the tag's *initial height* graded by how depolarized the segment was at induction, instead of fixed.

Two consequences follow directly from the diagram above. First, because `V_seg` is shared across all synapses on one segment (e.g., all six of `Cluster1`), several individually subthreshold synapses can jointly drive `V_seg` into the steep part of `g_Ca` without any of them, or their shared `thr` unit, firing a dSpike. This gives the clustering argument in the feasibility table a mechanistic channel — voltage-dependent cooperativity — in addition to its wiring-based one. Second, `V_seg` is implementable as one added analog block per `thr` unit: a shared leaky-integrator capacitor plus a differential-pair sigmoid stage, both standard sub-threshold-CMOS neuromorphic primitives, driving a graded-amplitude SET pulse into the same volatile threshold-switching memristor proposed as the tag element in §2, in place of a fixed-height pulse. The open risk is empirical: whether that memristor's SET response is graded over a wide enough pulse-amplitude range to carry a useful dynamic range of tag amplitudes — a device-characterization question parallel to the `τ_rf` variability already flagged in the junction table, measured along the amplitude axis instead of the timing axis. This predicts a companion falsification test to the one in §2: dose-dependent partial NMDAR blockade (reducing `g_Ca(V_seg)` without eliminating dSpike firing) shrinks captured-weight step size continuously, rather than switching captured synapses to uncaptured all-or-nothing.

### 4.3 Passive EPSP current versus dendritic spikes in somatic spike generation

The boolean `thr` output and the graded `V_seg` both carry dendritic-spike-related current to the soma. A second pathway operates in parallel: passive electrotonic spread of the EPSP via intracellular (axial) longitudinal current, independent of any dendritic spike.

**Relative magnitude.** A single spine-level passive EPSP reaching the soma is 0.1–2 mV, against a depolarization gap from rest to somatic threshold of roughly 15–20 mV — order 1–5% of what firing requires per event. Reaching threshold through this pathway alone requires temporal/spatial summation across tens to hundreds of quasi-coincident inputs. A dendritic spike (an NMDA spike, the event `V_seg`'s `g_Ca` gain term is patterned on) delivers a somatic depolarization of 5–15 mV per event, one to two orders of magnitude more than a single passive EPSP. The pipeline digitizes the threshold-crossing `thr` event, not the raw EPSP waveform, because the `thr` event is the one that moves the soma a meaningful fraction of the way to threshold on its own.

**Location dependence.** Unitary synaptic conductance in real dendrites scales with distance from the soma — dendritic democratization (Magee & Cook, 2000; Andrasfalvy & Magee, 2001) — so a passive, non-boosted EPSP from a distal synapse reaches the soma at an amplitude close to that of a proximal synapse. The passive pathway is a graded, spatially-distributed, always-on contribution to somatic state, present regardless of whether any given segment's `thr` unit fires.

**Coverage in the pipeline as specified.** Every excitatory synapse in §4.1's topology routes through its segment's `thr` unit before reaching the junction hierarchy or the soma; a synapse that does not cross local threshold contributes nothing downstream. `Cluster3`'s inhibitory line bypasses `thr` and the junction hierarchy directly to `Soma th/rp`, on the basis of perisomatic/proximal inhibitory targeting; no equivalent excitatory bypass exists for the passive, distance-compensated contribution described above. The pipeline represents coincident threshold-crossing input (`thr` and the junction hierarchy) and, with `V_seg` (§4.2), cooperative subthreshold drive toward tag induction. It does not represent subthreshold input summed linearly toward the somatic threshold itself. `V_seg` feeds tag induction (§4.2, §2), not the somatic spike decision.

**Scope.** For this pipeline's target function — coincidence detection and timing: the ±20 ms STDP window, tag/capture, SWR replay-order fidelity — the dendritic spike is the dominant driver of somatic firing, and the boolean `thr` abstraction is the correct compression for that function. A graded, rate-coded excitatory contribution to the somatic decision, independent of any single coincidence event, is not represented in the pipeline as specified. Representing it requires a third input to `Soma th/rp`, alongside the junction hierarchy's output and `Cluster3`'s inhibitory line: a low-pass-filtered summation of excitatory `V_seg` states feeding the soma directly. This extension is outside the scope of `V_seg` as specified in §4.2, which feeds tag induction only.

### 4.4 Tag-element implementations: digital and non-digital

§2 and §4.2 specify the tag element at the level of circuit primitives, independent of device technology. §3.1's junction stage specifies two implementation options for its own `τ_rf` timer — a wired-OR/counter FSM or a volatile memristor — and the tag element admits the same choice. §4.4.1 specifies the digital option; §4.4.2 specifies a non-digital (memristive) candidate under active development.

#### 4.4.1 Digital implementation

A fully digital tag/capture circuit uses: a multi-bit register per tag site, holding the tag amplitude; a down-counter clocked by a slow divider, decrementing the register at a rate set to reach zero after `τ_tag` (an 8–12 bit register decremented roughly once per 1–15 minutes covers the `τ_tag ≈ 1–4 hr` range); and a digital comparator that checks the register against zero before each decrement. Capture is a single conditional write: if the PRP-threshold signal (already sampled at low-second resolution per §1's table, well within reach of a digital sampler) crosses its set point while the register is still nonzero, the register's current value is copied into a separate, non-decrementing capture register and the down-counter is disabled for that site; if the PRP-threshold signal has not crossed by the time the register reaches zero, the tag register clears and no capture occurs. `V_seg`'s graded tag amplitude (§4.2) sets the register's initial loaded value rather than a fixed constant, and the falsification test in §2 (flat weight distribution when capture is blocked, replay quality unaffected) applies unchanged by disabling the conditional-write path. This circuit uses only standard cells — register, down-counter, comparator, clock divider — with no memristor, retention variability, or device characterization; each site's `τ_tag` is set by a clock-divider ratio, deterministic and identical across sites. The cost is area: one register, counter, and comparator per tag site, versus one two-terminal memristor in the non-digital route of §4.4.2.

#### 4.4.2 Non-digital implementation: copper-aspirinate memristors

Caus, Sławek, Mazur, Zawal, Baś, Szaciłowski, Talanov & Abdi (2026, "The memristive implementation of the hippocampus: a hypothesis") report a concrete candidate for the volatile threshold-switching memristor specified as the tag element in §2, built and electrically tested. The material is polycrystalline copper(II) bis-aspirinate, [Cu₂(asp)₄], spin-coated as a thin layer on ITO and capped with a sputtered copper electrode; three axially-ligated derivatives — [Cu₂(asp)₄(py)₂] (pyridine), [Cu₂(asp)₄(bimi)₂] (benzimidazole), and [Cu₂(asp)₄(DABCO)₂] — were prepared from the same parent complex by adding an axial N-donor ligand. All but the DABCO derivative show pinched I–V hysteresis loops under cyclic voltammetry, the standard electrical signature of memristive switching. The axial ligand tunes two properties: conductivity (the pyridine and benzimidazole derivatives conduct roughly two orders of magnitude more than the unligated parent) and retention. In chronoamperometric retention testing, the unligated parent's low-resistance state decayed to the high-resistance state within about 50 minutes (volatile); the benzimidazole derivative's high- and low-resistance states remained stable, essentially noise-free, after 6 hours (non-volatile on the timescale tested).

This volatile-versus-latched split, produced by changing only the axial ligand on one underlying complex, brackets the `τ_tag ≈ 1–4 hr` range in §1's table and provides the two states the tag/capture analogy in §2 requires: a device that relaxes on its own (parent complex, minutes-scale retention) and a device that holds once switched (benzimidazole derivative, hours-plus retention) — a materials-level realization of "tag decays unless captured." The paper's I–V and retention data characterize switching and retention, not conductance-step-versus-pulse-amplitude behavior; the graded-SET behavior specified in §4.2's `V_seg` extension is not demonstrated by this data and remains an open device-characterization question. The same paper's methylammonium-lead-iodide perovskite devices, modified with graphene oxide, fullerenol (C₆₀(OH)), or multiwalled carbon nanotubes, show measurable STDP-like potentiation directly in the memristive response, a second materials route toward stage-1's threshold-plus-plasticity behavior. The same paper also proposes a random-junction stochastic-network framing (percolating Ag–Ag₂S and SWNT/Por-POM meshes) as a physical analogue for the probabilistic, disordered connectivity in the clustering stage.

---

*Full technical detail: [Consolidation Cascade](https://claude.ai/code/artifact/f879468e-00e8-4ca1-9acc-348b12cc7e33) (molecular timescales) and `neuron_model_optimization.md` (hardware feasibility).*