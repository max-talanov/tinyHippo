<!--
Standalone snippet — not yet wired into hippocampal_timescales_as_circuit_spec.md.
Drop the image + table below in place of (or right after) Fig. 1 in §1.

GitHub's markdown sanitizer strips raw inline <svg>/<a> markup from rendered
.md files (only the leftover text survives, no shapes) — the diagram below
is therefore a real .svg file, referenced as an ordinary Markdown image, so
it actually renders as a picture. An <img>-embedded SVG can't carry clickable
internal regions in any renderer, so the real click-to-jump behavior lives
in the plain-Markdown table underneath, which works everywhere.

Anchor targets in the table are GitHub-slug anchors computed from the current
headings in hippocampal_timescales_as_circuit_spec.md. If any heading text
in that file changes, regenerate the `#...` targets to match.
-->

## Interactive timescale map

Bar color marks which memristor tier implements the stage (§4.5): **blue** = quick/organic, hippocampal-local; **purple** = replay, the stage that touches both tiers at once; **orange** = slow/inorganic, cortical. Click a stage name in the table below the figure to jump to the section that specifies it.

![Timescale map: eight consolidation stages on a log time axis from 1 ms to 10 years, colored by memristor tier — blue for quick/organic stages from neurotransmission through late-LTP, purple for the replay stage that bridges both tiers, orange for systems consolidation on the slow/inorganic tier.](timescale_interactive_map.svg)

*Fig. 1′ — Static rendering of Fig. 1's timescale axis, colored by memristor tier per §4.5. Use the table below to jump to a stage's section.*

### Fallback: plain-link table

Inline SVG links are stripped by some renderers (notably GitHub.com's default markdown sanitizer). This table carries the same links as plain Markdown, which works everywhere:

| Stage | Time constant | Tier | Jump to |
|---|---|---|---|
| [Neurotransmission (AMPA/NMDA gating)](#31-layer-a--dendritic-spike-timescales) | ~0.5–5 ms | Quick (organic) | [§3.1](#31-layer-a--dendritic-spike-timescales) |
| [Short-term plasticity](#42-graded-synaptic-potential-state-per-segment-v_seg) | ~10 ms–1 s | Quick (organic) | [§4.2](#42-graded-synaptic-potential-state-per-segment-v_seg) |
| [Ca²⁺/kinase cascades](#42-graded-synaptic-potential-state-per-segment-v_seg) | ~1 s–5 min | Quick (organic) | [§4.2](#42-graded-synaptic-potential-state-per-segment-v_seg) |
| [AMPAR trafficking / early-LTP](#2-synaptic-tagging-and-capture-as-a-volatile-memristor-circuit) | ~1–30 min, decays over 1–3 hr | Quick (organic) | [§2](#2-synaptic-tagging-and-capture-as-a-volatile-memristor-circuit) |
| [Synaptic tagging & capture](#2-synaptic-tagging-and-capture-as-a-volatile-memristor-circuit) | tag set 1–2 min, decays τ≈1–4 hr | Quick (organic) | [§2](#2-synaptic-tagging-and-capture-as-a-volatile-memristor-circuit) |
| [Transcription / late-LTP](#45-memory-consolidation-replay-as-a-volatile-to-non-volatile-handoff) | onset ~30–60 min, stable ~2 days | Quick (organic) | [§4.5](#45-memory-consolidation-replay-as-a-volatile-to-non-volatile-handoff) |
| [Sharp-wave-ripple replay](#45-memory-consolidation-replay-as-a-volatile-to-non-volatile-handoff) | ripple 100–300 ms, recurs over hours–days | Bridge (organic → inorganic) | [§4.5](#45-memory-consolidation-replay-as-a-volatile-to-non-volatile-handoff) |
| [Systems consolidation](#45-memory-consolidation-replay-as-a-volatile-to-non-volatile-handoff) | days–weeks (rodent) to months–years (human) | Slow (inorganic) | [§4.5](#45-memory-consolidation-replay-as-a-volatile-to-non-volatile-handoff) |
