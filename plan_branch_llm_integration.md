# Branch plan: LLM integration

Connect the tinyHippo model to an LLM so that the model's defining property,
information that is encoded, replayed during sleep and consolidated to cortex while
keeping its identity, does useful work for the LLM.

- **Branch:** `llm-integration`, created from `main` at the commit where the gate
  below passes.
- **Sibling branch:** `bio-plasticity` ([plan_branch_bio_plasticity.md](plan_branch_bio_plasticity.md)),
  created from the same gate commit.

## 0. Entry condition: pattern identity survives consolidation

The full definition is in the plasticity plan, §0. In short:
- **G1:** the pattern ID is decodable from mPFC and EC LV after consolidation.
- **G2:** a partial cue after the hippocampal lesion reactivates the right pattern.
- **G3:** claims 1–2 still hold.
- All three at 12%, ≥2 seeds, permutation p < 0.01.

The shared scorer `decode_identity.py` is also this branch's recall decoder. Without
G1 and G2 an LLM cannot get back *which* memory was stored, and the integration
would not be meaningful.

**Exception:** step L0 below has no dependency on the model and can start now.

## 1. What "integration" means: the chosen design

| Option | Description | Decision |
|---|---|---|
| **A. The model as episodic memory for an LLM agent** | The LLM calls `remember`. The model encodes the memory, replays it during "sleep" and consolidates it. The LLM calls `recall(cue)` and gets back the memory IDs the model recovers. | **Primary.** It is the only option that uses the project's scientific result, identity surviving consolidation. |
| B. The LLM operates the simulator | An agent that designs runs, submits MN5 jobs and analyses h5 files | Useful tooling, but it does not depend on the model working. Not this branch. |
| C. LLM-guided replay prioritization | The LLM scores the importance of memories, and importance biases which memories are replayed and tagged | **Later phase (L6)**, built on A. It pairs with neuromodulated tagging in the plasticity branch. |

## 2. Architecture (option A)

```
┌────────────┐  tool calls   ┌──────────────────────┐   drive/readout   ┌──────────────┐
│ LLM agent  │ ────────────► │  memory service      │ ────────────────► │ tinyHippo    │
│ (Claude,   │  remember /   │  - content store     │  encode / sleep / │ sim server   │
│  tool use) │ ◄──────────── │    (SQLite: ID→text) │ ◄──────────────── │ (NEST, kept  │
└────────────┘  recall       │  - codec text→code   │   spikes → IDs    │  in memory)  │
                             │  - decoder (G1/G2)   │                   └──────────────┘
                             └──────────────────────┘
```

**The model stores the index, not the content.** This follows hippocampal indexing
theory (Teyler & DiScenna, 1986). The text lives in an ordinary store keyed by
memory ID. The model holds a sparse code for each ID and must return the right ID
from a partial cue, especially after consolidation and after the lesion. This split
is honest about capacity: today the model separates 3 patterns, so it cannot hold
text.

**The memory lifecycle:**

1. **Wake: encode.**
   - `remember(text)` assigns an ID, embeds the text, maps it to a sparse code and
     queues it.
   - The drive enters through EC LII, the model's intended input. Until §21's gap is
     closed, it enters through the CA3 scaffold instead, flagged as a
     less-plausible fallback.
2. **Sleep: consolidate.**
   - An offline batch replays the new memories interleaved with old ones over N
     blocks.
   - STC and cortical plasticity update the weights, and the plastic weights are
     checkpointed.
3. **Recall.**
   - `recall(query)` embeds the query and delivers it as a partial cue to EC LII,
     or to mPFC for "remote" memories.
   - A short simulation runs, the decoder ranks the IDs, and the service returns
     the top-k texts with confidence scores.

**The LLM side:**
- Anthropic Python SDK, Claude API tool use.
- Default model `claude-opus-5-5`, with `output_config.effort` tuned per route.
- Custom tools `remember`, `recall` and `sleep`, run through the SDK's Tool Runner
  (`client.beta.messages.tool_runner` with `@beta_tool`).
- Tool schemas use `strict: true`.
- `tool_choice` stays `auto`: forced tool use is rejected on current models.

**Baselines for every evaluation:** the same agent backed by (i) a vector store and
(ii) Claude's built-in memory tool (`memory_20250818`).

## 3. Hard constraints to plan around

| Constraint | Current number | Consequence |
|---|---|---|
| Speed | 12%: 5.4h per 14 s of sim time. 1%: about 26 min per 14 blocks with the full stack (RESULTS §25–26). | Recall must take seconds to minutes, so it runs at 1% or in a reduced model, inside a **persistent sim server** that builds once and keeps the kernel alive. Sleep runs offline as a batch. |
| Capacity | 3 patterns demonstrated | Measure a capacity and interference curve (L2) before building features on top of it |
| Stochasticity | DG is chaotic from block to block (CLAUDE.md) | Recall returns a ranked list with confidence, and can repeat the cue and aggregate |
| No kernel serialization | A NEST network cannot be saved and reloaded | Checkpoint = seed + flags + plastic weight arrays. Rebuild deterministically, then write the weights back through the hooks' SynapseCollections. |
| Backend | CPU NEST is the validated reference; GPU port is unproven | This branch is the candidate for NEST GPU ([nest_gpu_migration.md](nest_gpu_migration.md) §0). **Conditional:** decided at L3 from Phase 0 numbers (recall latency, hook get/set cost, per-neuron `b`). The `bio-plasticity` branch stays on CPU. Use the `simbackend` shim (`--backend nest|nestgpu`) so the choice is a flag, not a fork. |

## 4. Phases

### L0 — Service skeleton and benchmark with a stub backend (can start now)

1. Create `llm/`:
   - `memory_service.py`: SQLite content store and the backend interface
   - `agent.py`: the Tool Runner agent
   - `backends/stub.py`: exact-match or vector-store backend
   - `bench/`
2. Build the benchmark: a multi-session dialogue in which facts introduced early must
   be recalled many turns and "sleeps" later. Include distractor and near-duplicate
   facts. Metrics: recall@k, false-recall rate, and latency.
3. This lets the interface and evaluation settle while the hippocampal gate is
   still being finished.

### L1 — Arbitrary pattern codes in the simulator

1. Add `--pattern-file codes.npy`, holding N sparse drive vectors over EC LII (or over
   CA3 groups for the fallback), plus a schedule file. This generalizes today's
   ring positions and interleaved groups.
2. Re-run G1 with random codes to show that the gate does not depend on the
   hand-built ring or group layout.

### L2 — Capacity and interference

1. At 1%, sweep N = 3 → 10 → 30 stored codes, then run one 12% point.
2. Measure:
   - G1 accuracy against N;
   - retention of old memories after new ones are added;
   - accuracy against code overlap.
3. This fixes N_max and the sparsity of the codec. It also directly tests DG pattern
   separation on similar inputs (claim 2).

### L3 — Persistent sim server and checkpointing

0. **Backend decision.** Compare recall latency on `--backend nest` and `--backend nestgpu` at 1% and 12% (Phase 0 and 2 results). Choose GPU only if it wins and the G1/G2 gate reproduces on it; otherwise stay on CPU.
1. Build a long-lived process holding the built network, with the endpoints
   `encode(code)`, `sleep(n_blocks)`, `recall(partial_code)` and
   `checkpoint()`/`restore()`.
2. Target recall latency: under 60 s at 1%. Measure first; the result may reshape L5.

### L4 — Text codec

1. Map text → embedding (a sentence-embedding model; a local model keeps the
   pipeline offline and deterministic) → a sparse code (random projection plus
   k-winners-take-all at the sparsity chosen in L2).
2. Similar texts get overlapping codes.
3. **Scientific check:** does DG decorrelate near-duplicate memories? Compare input
   overlap with CA3/CA1 overlap.

### L5 — End to end

1. Swap the stub for the model backend and run the L0 benchmark against both
   baselines.
2. **Expectation:** the model will not beat a vector store on raw accuracy. It is
   evaluated on the behaviours only it has:
   - sleep-dependent consolidation (recall improves after `sleep`);
   - forgetting of memories that are never replayed;
   - interference between similar memories;
   - survival of remote memories after the hippocampal lesion.

### L6 — LLM-guided replay prioritization (option C)

1. The LLM tags each memory with an importance score, which sets that memory's
   replay frequency and its tag or PRP gain.
2. This is the functional analogue of neuromodulated tagging and is shared with
   the plasticity branch.
3. **Test:** important memories should consolidate first and survive the lesion
   more often.

## 5. Risks

| Risk | Mitigation |
|---|---|
| Gate G2 (lesion) takes much longer than G1 | L1–L4 need only G1. G2 gates just the "remote memory" part of L5. |
| Capacity stays near a handful of patterns | Report it as the finding. The index-not-content design still works for small N, and L2 tells us early. |
| Recall latency is too high | Reduced-scale server (1% or smaller, no MN5), cached builds, and short recall windows (one SWR window, not one block) |
| Codec quality dominates the results | Keep the codec fixed across the model and baseline comparisons, and ablate it separately |
| GPU backend does not reproduce the gate, or is no faster at 1% | Stay on CPU NEST; the shim makes this a flag. The decision is taken at L3, not before |
| Coupling to research code | The service depends only on the CLI flags and the checkpoint format, never on `replay_scaled.py` internals |

## 6. First steps

- [ ] Now, on a short-lived branch or `main`: L0 skeleton, stub backend, benchmark
- [ ] Once the gate passes: `git switch -c llm-integration <gate-commit>` and push
- [ ] `--pattern-file` (L1) and the random-code G1 check at 1%
- [ ] Phase 0 NEST GPU spike ([nest_gpu_migration.md](nest_gpu_migration.md)), in parallel with L0
- [ ] Capacity sweep at 1% (L2), plus a RESULTS.md section
