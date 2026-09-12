<!--
Standalone snippet — not yet wired into hippocampal_timescales_as_circuit_spec.md.
Drop the SVG block below in place of (or right after) Fig. 1 in §1, and the
fallback table wherever a plain-text version is preferred (e.g. an appendix,
or directly under the figure for renderers that strip inline SVG).

Anchor targets below are GitHub-slug anchors computed from the current
headings in hippocampal_timescales_as_circuit_spec.md. If any heading text
in that file changes, regenerate the `href`/`#...` targets to match.
-->

## Interactive timescale map

Click any stage — in the diagram or the table — to jump to the section that specifies it. Bar color marks which memristor tier implements the stage (§4.5): **blue** = quick/organic, hippocampal-local; **purple** = replay, the stage that touches both tiers at once; **orange** = slow/inorganic, cortical.

<svg viewBox="0 0 980 440" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">
  <!-- legend -->
  <rect x="60" y="10" width="14" height="14" fill="#2b6cb0"/>
  <text x="78" y="21" font-size="11" fill="currentColor">Quick tier — organic volatile memristor (hippocampal)</text>
  <rect x="430" y="10" width="14" height="14" fill="#6b46c1"/>
  <text x="448" y="21" font-size="11" fill="currentColor">Replay — bridges both tiers</text>
  <rect x="650" y="10" width="14" height="14" fill="#c05621"/>
  <text x="668" y="21" font-size="11" fill="currentColor">Slow tier — inorganic non-volatile memristor (cortical)</text>

  <!-- vertical gridlines at each tick -->
  <g stroke="#8888" stroke-width="1" stroke-dasharray="2,3">
    <line x1="91" y1="45" x2="91" y2="400"/>
    <line x1="163" y1="45" x2="163" y2="400"/>
    <line x1="235" y1="45" x2="235" y2="400"/>
    <line x1="307" y1="45" x2="307" y2="400"/>
    <line x1="434" y1="45" x2="434" y2="400"/>
    <line x1="562" y1="45" x2="562" y2="400"/>
    <line x1="661" y1="45" x2="661" y2="400"/>
    <line x1="721" y1="45" x2="721" y2="400"/>
    <line x1="766" y1="45" x2="766" y2="400"/>
    <line x1="844" y1="45" x2="844" y2="400"/>
    <line x1="916" y1="45" x2="916" y2="400"/>
  </g>

  <!-- stage bars: each wrapped in a link to its section -->
  <a href="#31-layer-a--dendritic-spike-timescales">
    <title>Neurotransmission (AMPA/NMDA gating), ~0.5–5 ms — jump to §3.1</title>
    <rect x="70" y="60" width="72" height="24" rx="4" fill="#2b6cb0"/>
    <text x="150" y="76" font-size="12" fill="currentColor">Neurotransmission (AMPA/NMDA gating) — 0.5–5 ms</text>
  </a>

  <a href="#42-graded-synaptic-potential-state-per-segment-v_seg">
    <title>Short-term plasticity, ~10 ms–1 s — jump to §4.2</title>
    <rect x="163" y="102" width="144" height="24" rx="4" fill="#2b6cb0"/>
    <text x="315" y="118" font-size="12" fill="currentColor">Short-term plasticity — 10 ms–1 s</text>
  </a>

  <a href="#42-graded-synaptic-potential-state-per-segment-v_seg">
    <title>Ca²⁺/kinase cascades, ~1 s–5 min — jump to §4.2</title>
    <rect x="307" y="144" width="177" height="24" rx="4" fill="#2b6cb0"/>
    <text x="492" y="160" font-size="12" fill="currentColor">Ca²⁺/kinase cascades — 1 s–5 min</text>
  </a>

  <a href="#2-synaptic-tagging-and-capture-as-a-volatile-memristor-circuit">
    <title>AMPAR trafficking / early-LTP, ~1–30 min, decays over 1–3 hr — jump to §2</title>
    <rect x="434" y="186" width="162" height="24" rx="4" fill="#2b6cb0"/>
    <text x="604" y="202" font-size="12" fill="currentColor">AMPAR trafficking / early-LTP — 1–30 min, decays 1–3 hr</text>
  </a>

  <a href="#2-synaptic-tagging-and-capture-as-a-volatile-memristor-circuit">
    <title>Synaptic tagging &amp; capture — tag set 1–2 min, decays τ≈1–4 hr — jump to §2</title>
    <rect x="434" y="228" width="171" height="24" rx="4" fill="#2b6cb0"/>
    <text x="613" y="244" font-size="12" fill="currentColor">Synaptic tagging &amp; capture — tag 1–2 min, decay τ≈1–4 hr</text>
  </a>

  <a href="#45-memory-consolidation-replay-as-a-volatile-to-non-volatile-handoff">
    <title>Transcription / late-LTP, onset ~30–60 min, stable ~2 days — jump to §4.5</title>
    <rect x="540" y="270" width="142" height="24" rx="4" fill="#2b6cb0"/>
    <text x="690" y="286" font-size="12" fill="currentColor">Transcription / late-LTP — onset 30–60 min, stable ~2 days</text>
  </a>

  <a href="#45-memory-consolidation-replay-as-a-volatile-to-non-volatile-handoff">
    <title>Sharp-wave-ripple replay, ripple 100–300 ms, recurs over hours–days — jump to §4.5</title>
    <rect x="235" y="312" width="497" height="24" rx="4" fill="#6b46c1"/>
    <text x="742" y="328" font-size="12" fill="currentColor">SWR replay — 100–300 ms, recurs hours–days</text>
  </a>

  <a href="#45-memory-consolidation-replay-as-a-volatile-to-non-volatile-handoff">
    <title>Systems consolidation, days–weeks (rodent) to months–years (human) — jump to §4.5</title>
    <rect x="661" y="354" width="255" height="24" rx="4" fill="#c05621"/>
    <text x="925" y="370" font-size="12" fill="currentColor" text-anchor="end">Systems consolidation — days–years</text>
  </a>

  <!-- axis -->
  <line x1="60" y1="400" x2="930" y2="400" stroke="currentColor" stroke-width="1.5"/>
  <g font-size="10" fill="currentColor">
    <text x="91" y="415" text-anchor="end" transform="rotate(-40 91 415)">1 ms</text>
    <text x="163" y="415" text-anchor="end" transform="rotate(-40 163 415)">10 ms</text>
    <text x="235" y="415" text-anchor="end" transform="rotate(-40 235 415)">100 ms</text>
    <text x="307" y="415" text-anchor="end" transform="rotate(-40 307 415)">1 s</text>
    <text x="434" y="415" text-anchor="end" transform="rotate(-40 434 415)">1 min</text>
    <text x="562" y="415" text-anchor="end" transform="rotate(-40 562 415)">1 hr</text>
    <text x="661" y="415" text-anchor="end" transform="rotate(-40 661 415)">1 day</text>
    <text x="721" y="415" text-anchor="end" transform="rotate(-40 721 415)">1 week</text>
    <text x="766" y="415" text-anchor="end" transform="rotate(-40 766 415)">1 month</text>
    <text x="844" y="415" text-anchor="end" transform="rotate(-40 844 415)">1 year</text>
    <text x="916" y="415" text-anchor="end" transform="rotate(-40 916 415)">10 yr</text>
  </g>
  <text x="495" y="435" text-anchor="middle" font-size="11" fill="currentColor">Time (log scale)</text>
</svg>

*Fig. 1′ — Interactive version of Fig. 1's timescale axis. Each bar links to the section that specifies that stage; bar color marks its memristor tier per §4.5.*

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
