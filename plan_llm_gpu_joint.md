# Joint plan: LLM integration on NEST GPU

Status: proposal, 2026-10-24. Merges [plan_branch_llm_integration.md](plan_branch_llm_integration.md)
(L0-L6) with [nest_gpu_migration.md](nest_gpu_migration.md) (Phases 0-5). Those two stay the
detailed references; this document fixes the **order, the merge points and the go/no-go gates** for the
part of the work that needs both.

Scope reminder (migration doc §0): only `llm-integration` targets NEST GPU. `bio-plasticity` stays on CPU NEST.

## 1. Where things stand (2026-10-24)

- Gate G1/G2/G3 is **not met**. Grouping every projection on the pattern path (`--cortex-group-frac`,
  `--mf-group-frac`, EC LII→CA3) is implemented; the 12% runs that decide it (JOB H15 G1, JOB H16 EC LII
  path, JOB H17 G2 probe) are queued on MN5. Neither branch can fork until they pass.
- CPU runs are **bit-identical at a fixed seed, config and thread count** (RESULTS.md §29). This makes
  the backend shim cheap to verify (hash the HDF5), but a GPU port will not be bit-identical, so GPU
  equivalence is statistical only.
- `decode_identity.py` exists (home-group z, leave-one-block-out G1 decoder) and doubles as the GPU
  equivalence scorer and the LLM recall decoder.
- `run.sh` already submits with `-A uab100 -q acc_resc` (the accelerated partition), so GPU nodes are
  probably reachable on the same account. **[verify]** CPU runs happen there today; confirm that a GPU
  is allocatable and that CUDA/NEST GPU can be built on it.

## 2. What the GPU is for in this project

Be specific, because 1% (~8k neurons) is a small network for a GPU:

| Use | Why it needs speed | Does the GPU plausibly help? |
|---|---|---|
| `recall(cue)` latency in the persistent sim server | Target < 60 s per recall at 1% | Maybe. Build time is paid once; a short SWR-window simulation at 8k neurons is already cheap on CPU. Measure before assuming. |
| Many recalls per query (repeat and aggregate against DG chaos; top-k ranking over N memories) | Recall cost multiplies by repeats and by N | Yes, if trials can share one built network. Cue trials are independent reads. |
| Capacity sweeps (L2: N = 3 → 10 → 30, then a 12% point) | Each point is a full encode + sleep + recall run | Yes at 12%, marginal at 1%. |
| `sleep(n_blocks)` batches | 12% is ~5.4 h per 14 s of sim time on CPU | The main prize at 12%, **if** hooks are fast on-device (connection get/set). |
| Full-scale (780k neurons) | Not reachable in practice on CPU | Long-term. Out of this plan's critical path. |

Consequence: the first decision is **not** "port everything", it is "is a GPU recall server measurably
better than a persistent CPU server?" That is what Phase 0 + D2 below decide.

## 3. Two NEST GPU gaps found since the first plan

1. **No `sinusoidal_poisson_generator` / `inhomogeneous_poisson_generator`.** `tiny.py` builds theta and
   the SWR ripple background from `sinusoidal_poisson_generator` (5 call sites) and `replay_scaled.py`
   has an open item for `inhomogeneous_poisson_generator`. NEST GPU has `poisson_generator` (a `rate`
   parameter, unique train per target) and `spike_generator`. Replacements to test in Phase 0:
   (a) `poisson_generator` with `rate` updated between short `Simulate` slices (1-2 ms), (b) a
   precomputed `spike_generator` train per population or per target. Theta and the ripple envelope shape
   the replay, so the replacement must reproduce replay ρ (+0.6 / -0.6 band), not just mean rates.
2. **Group-clustered wiring and arbitrary memories.** Pattern identity currently travels through
   *structural* group wiring (home group `i % n_groups`, `*-group-frac` on each hop). That fixes the
   number of storable patterns at build time. An LLM `remember()` stream needs new memories after the
   network is built. See §6, decision D3.

## 4. Workstreams and branches

```
main ──●──────── Phase 1: simbackend shim (CPU, bit-identical) ────●── gate commit (G1-G3 pass)
        \                                                            \
         nest-gpu ── Phase 0 spike ── Phase 2 core net ── Phase 3 hooks    llm-integration ── L1 ... L5
                          │                  │                │               ▲
                          D1                 D2               └──── merges ───┘ (D3, D4)
         L0 (stub backend, bench) runs on a short-lived branch / main throughout
```

| Stream | Branch | Depends on | Output |
|---|---|---|---|
| S0: shim | `main` | nothing | `simbackend.py`, `--backend nest` default, HDF5 hash unchanged |
| S1: GPU spike and backend | `nest-gpu` (from `main` after S0) | S0 | `--backend nestgpu`, equivalence report |
| S2: LLM service | `llm-integration` (from the gate commit); L0 earlier on `main` | gate for L1+; none for L0 | `llm/` service, benchmark, codec |
| S3: bio-plasticity | `bio-plasticity` | gate | CPU only, no overlap with S1/S2 except the shim |

Rules to keep the branches mergeable: all backend-specific code behind `simbackend`; new model features
(`--*-group-frac`, hooks) stay behind flags; `main` is merged into `nest-gpu` and `llm-integration`
after every gate-relevant commit.

## 5. Schedule and gates

Durations are estimates, not measurements. Weeks count from the start of S0.

| Week | S0/S1 (GPU) | S2 (LLM) | Gate |
|---|---|---|---|
| 1 | Phase 0 spike on an MN5 GPU node: build NEST GPU, f-I match, per-neuron `b`, sign/delay, theta/ripple generator replacement, connection get/set benchmark, mid-run create test | L0: `llm/` skeleton, SQLite store, stub backend, benchmark dialogue | **D1** end of week 1 (§6) |
| 2-3 | S0: `simbackend` shim, route ~190 `nest.*` calls; bit-identical hash on a 1% run | L0 finished: baselines (vector store, memory tool) scored | S0 gate: identical HDF5 |
| 3-5 | Phase 2: GPU backend, DG → CA3 → CA1 → replay at 1%; scored with `decode_identity.py` | wait for H15-H17; prepare L1 (`--pattern-file`) design | **D2** (§6) |
| 5-8 | Phase 3: hooks (STC, MPFC assoc, Schaffer STDP, homeostasis), neurogenesis pool, cortical recall probe on GPU, 12% point | L1-L2 on CPU: arbitrary codes, capacity curve at 1% | G1/G2/G3 reproduced on GPU at 12%, 2 seeds |
| 8-10 | latency benchmark: CPU persistent server vs GPU server, 1% and 12% | L3: persistent sim server on the winning backend; checkpoint/restore | **D4** (§6) |
| 10-13 | multi-GPU only if 12% sleep is the bottleneck | L4 codec, L5 end-to-end vs both baselines, RESULTS section | |

L2 on CPU first is deliberate: capacity is a model question, not a backend question, and CPU is the
validated reference. GPU capacity runs repeat only the chosen points.

## 6. Decisions

**D1: after Phase 0 (week 1). Is the GPU path viable at all?** Proceed only if all hold:
f-I curves within ~5% after recalibration; per-neuron `b` heterogeneity solved (or a sub-group split
that keeps the spread below `b_crit`); connection get/set for 1M synapses under ~1 min at a 12%-sized
network; theta and ripple generators reproduced well enough to keep replay ρ in band.
Fail → `llm-integration` stays on CPU, S1 stops, and the persistent server is built on CPU NEST.

**D2: after Phase 2 (week 5). Core equivalence.** At 1% and across ≥2 seeds, GPU vs CPU: replay ρ both
directions, DG % active (2-4% target), CA3 completion, home-group z in CA3/CA1. Same bands as `main`,
compared as distributions (GPU will not be bit-identical).
Fail → fix on the GPU side (dt, integrator, delay rounding); two failed rounds → stay on CPU.

**D3: how do memories map onto patterns? (before L1, needs a model owner's call).** Options:
(a) **Fixed bank**: build with `n_groups = N_max`; each memory takes the next free group. Capacity is the
group count; simple, matches the validated wiring; no new memories after N_max.
(b) **Hash-to-group**: codec maps text to one of `n_groups` groups with overlap; several memories share
a group; recall returns a candidate set the content store disambiguates.
(c) **Learned grouping** (the bio-plasticity branch's P4: clustering from plasticity). Most faithful,
but it is research, not integration.
Recommendation: (a) for L1-L3, measure interference in L2, and treat (c) as the long-term route.
The GPU plan is unaffected, except that (a) means `n_groups` ≫ 3 must itself be validated (G1 at N = 10
and 30) before the LLM branch builds on it.

**D4: which backend runs the persistent server? (week 8-10).** GPU only if, on the same recall
benchmark, it beats CPU by a margin that matters (suggested: ≥3x on repeated recall at 1%, or the 12%
sleep batch fits a practical session) **and** the gate reproduces on it. Otherwise CPU server, GPU kept
for batch sleep/capacity runs.

## 7. Design of the GPU-resident memory service

Builds on L3 of the LLM plan; additions specific to NEST GPU:

- **One process, one GPU, network built once.** NEST GPU has no kernel reset, so each experiment
  needing a fresh network (e.g. pattern-completion probes) runs in a subprocess; the server never
  resets.
- **Fixed-at-first-`Simulate` network.** Everything a memory could ever need is preallocated:
  the bank of N_max groups (D3-a), plus a silent granule-cell pool for neurogenesis if that hook is
  used. `remember()` activates resources (sets drive, enables weights); it never calls `Create`/`Connect`.
- **Checkpoint = seed + flags + plastic weight arrays**, as in the LLM plan. On GPU the weight read-back
  goes through cached connection-id arrays (built once after construction, reused every epoch, never
  `GetConnections(source=, target=)`). Restore rebuilds deterministically, then writes weights with
  `SetConnectionStatus`. Validate that restore reproduces recall results within the stochastic band.
- **Recall as batched trials.** A recall is a partial-cue trial of a few hundred ms. Repeat K times in one
  call (as the G2 probe already interleaves cue trials) and aggregate with `decode_identity`-style
  ranking; return top-k IDs with confidence. Trials must be spaced so activity from one does not prime
  the next (the G2 probe uses 250 ms; keep a priming-leak control).
- **Spike readout via the backend shim**: `record_spikes(window)` returns senders/times for the last
  window only, so cumulative GPU buffers do not leak across trials. Cell positions come from
  `gid_first`, never the lowest spiking GID.
- **Determinism**: record seed, thread/GPU count and backend in every result; GPU results are reported
  as distributions over seeds, CPU results may also cite hashes.

## 8. Evaluation matrix

| Axis | Values |
|---|---|
| Backend | CPU NEST (reference), NEST GPU |
| Memory system | tinyHippo (index in the model, text in SQLite), vector store, Claude memory tool (`memory_20250818`) |
| Metric | recall@k, false-recall rate, latency, plus model-only behaviours: recall improves after `sleep`, forgetting of unreplayed memories, interference between similar memories, survival after the hippocampal lesion |
| Scale | 1% (server), 12% (batch capacity and sleep) |
| Seeds | ≥2, permutation p < 0.01 on identity claims |

The model is not expected to beat a vector store on raw accuracy. GPU-vs-CPU numbers are reported next to
the CPU reference in a new RESULTS.md section (do not edit older sections).

## 9. Risks specific to the joint plan

| Risk | Mitigation |
|---|---|
| H15-H17 fail and the gate never passes | S0/S1/L0 do not depend on it, so that work stays useful; revisit the pattern wiring before more LLM work |
| Theta/ripple generators cannot be reproduced on NEST GPU | Fall back to rate-sliced `poisson_generator`; if replay ρ leaves its band, D1 fails and the LLM server stays on CPU |
| Per-neuron `b` impossible on stock `izhikevich` | NESTML / `user_m1` model, or K sub-groups with a coarser `b` spread; check `b_crit` per population |
| 1% is too small for GPU benefit | D4 compares like with like; a CPU server is an acceptable outcome |
| Fixed group bank limits memory count | D3: report the capacity curve as a finding; hash-to-group as the fallback |
| Two backends diverge on features | Every new flag lands behind the shim; CI-style smoke test runs both at 1%, 1 SWR |
| Engineering effort crowds out the open science problem (engram specificity) | S1 and S2 never block `main` science work; the gate commit is whatever `main` has when H15-H17 pass |
| GPU access on `uab100` unconfirmed | Week 1 task; fallback is a Linux/NVIDIA workstation or cloud GPU |

## 10. First steps

- [ ] Confirm a GPU is allocatable on the `uab100` / `acc_resc` account and CUDA + OpenMPI modules exist
- [ ] Build NEST GPU there; record versions in `COMPATIBILITY.md`
- [ ] Phase 0 spike (migration doc §3), including the theta/ripple generator replacement and the
  connection get/set benchmark; write the numbers into a RESULTS.md section
- [ ] L0 skeleton and benchmark (`llm/`), independent of everything else
- [ ] Decide D1 at the end of week 1, then start S0 (shim) regardless of D1: the shim is useful for CPU too
- [ ] Wait for H15-H17 before forking `llm-integration`; do not fork on a guess
