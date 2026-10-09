# Migrating tinyHippo to NEST GPU: options and plan

Status: proposal, 2026-10-09. Based on a survey of how `replay_scaled.py` and `tiny.py` use
NEST, plus a read of the `nest/nest-gpu` source (`pythonlib/nestgpu.py`, `src/`, `doc/`).
Items marked **[verify]** were not confirmed and are the first things the Phase 0 spike must settle.

## 1. What the code needs from the simulator

PyNEST call counts in `replay_scaled.py` (5,873 lines) and `tiny.py`:

| Need | Uses | NEST GPU equivalent | Fit |
|---|---|---|---|
| `izhikevich` neurons | 24 `Create` sites | `izhikevich` (`src/izhikevich.cu`): same equations, `I_e` in pA, spike adds weight to V, group params `a,b,c,d,V_th` | Good |
| Per-neuron heterogeneity (`het_params`, V_m/U_m/I_e arrays) | many | `SetStatus` with arrays or `RandomNormal` distributions | Good. `a,b,c,d` are *group* params in NEST GPU, so per-neuron `b` heterogeneity (a methodological must-have) is **not** possible on the stock model **[verify]** |
| `Connect`, `fixed_indegree` (6), `pairwise_bernoulli` (1) | 86 sites | `one_to_one`, `all_to_all`, `fixed_total_number`, `fixed_indegree`, `fixed_outdegree`, `assigned_nodes` | Good. The one `pairwise_bernoulli` site already has a documented `fixed_indegree` fallback |
| Per-synapse delay jitter | `jittered_delay` | delay arrays via `SetSynSpecFloatPtParam` | Good **[verify]** |
| `static_synapse` + Python plasticity hooks (STC, MPFC assoc, Schaffer STDP, homeostasis) | 4 hooks | `GetConnections` / `GetConnectionFloatParam` / `SetConnectionStatus` between `Simulate` calls | Works in principle. Cost of `GetConnections` at 20M-2B synapses is the key unknown |
| `stdp_synapse` | 6 mentions, deliberately avoided (CPU path runs ~300 syn/s) | single `stdp` synapse group (`tau_plus, tau_minus, lambda, alpha, mu_*, Wmax`), computed on the GPU | Fine for plain pair STDP. No triplet, dopamine or tagging models, so STC stays in Python |
| `poisson_generator` (20) | background + replay drive | `poisson_generator` node, connected like a neuron | Good. Confirm per-target independence **[verify]** |
| `spike_recorder` (23), analysis on `events` | all scoring | `ActivateRecSpikeTimes` / `GetRecSpikeTimes`, or `CreateRecord` | Needs an adapter; cumulative buffer, so windowing per epoch is ours to do |
| `multimeter` (6) | V_m/U_m traces | `CreateRecord` (file or in-memory) | Good |
| `NodeCollection` slicing/arithmetic (26) | everywhere | `NodeSeq` (contiguous ranges only) | Needs a thin wrapper |
| `DGNeurogenesisHook` | creates cells and connects mid-run | network is fixed at the first `Simulate`/`Calibrate` **[verify]** | **Hard.** Needs a preallocated, silent granule-cell pool that the hook activates |
| `nest.ResetKernel()` (3 calls, pattern-completion probe) | | no kernel reset | Run each probe in a fresh process |
| MPI | `local_num_threads`, 1 task x 50 cores | one GPU per rank; `RemoteCreate` / `RemoteConnect`, whole populations per rank | Natural fit for the modular DG/CA3/CA1/EC/mPFC layout |

**Platform constraint.** NEST GPU needs Linux and an NVIDIA GPU with the CUDA toolkit. The current
laptop loop (macOS, 1% runs in 10-20 min) cannot run it. Any plan must keep a CPU path for local work.

## 2. Options

**A. Stay on NEST CPU and optimise.** Zero scientific risk. The 3.9 pain points (STDP `Connect`
serialisation, whole-kernel `GetConnections` scans, 20 h walls) have been worked around, not removed.
Full-rat scale (~780k neurons) remains an MN5 CPU-hours problem.

**B. Port to NEST GPU behind a backend shim (recommended).** Keep PyNEST-style model code, add a
small `simbackend` layer with `nest` and `nestgpu` implementations. Same Python API shape, `Izhikevich`
and STDP exist, multi-GPU via MPI matches the full-scale goal. Risks are per-neuron `b`, no mid-run
neuron creation, recorder/connection-query semantics, and no local GPU.

**C. NEST GPU plus NESTML-generated neuron models.** Only needed if B fails on the neuron model
(per-neuron `a,b,c,d`) or if tag/PRP state should move on-device. NESTML advertises a NEST GPU target,
but I did not verify its maturity **[verify]**. Treat as a fallback inside B, not a separate path.

**D. A different GPU simulator (GeNN, Brian2CUDA).** More flexible custom plasticity, but it is a
rewrite off the PyNEST API, and loses continuity with all of RESULTS.md. Not researched in depth here;
only worth it if B's Phase 0 shows the NEST GPU API cannot host the hooks.

**Recommendation: B, gated by a short Phase 0 spike, with CPU NEST kept as the reference backend.**
Be honest about the benefit: the logs show much of the 12% wall time is Python hooks and
`GetConnections`, not `Simulate`. A GPU helps `Simulate` and construction, but the hooks only
speed up if connection get/set is fast on-device. Profile before promising a speedup.

## 3. Plan

### Phase 0: feasibility spike (go/no-go)
Standalone scripts in `scratchpad`-style `gpu_spike/`, not touching `replay_scaled.py`.
1. Build NEST GPU on a GPU node (MN5 ACC partition or a Linux/NVIDIA box). Record versions in `COMPATIBILITY.md`.
2. Izhikevich f-I check: DC rheobase and f-I curve vs CPU NEST for the DG/CA3 parameter sets
   (`I_rheo = (5-b)²/0.16 - 140 - I_e`), at dt 0.1 ms. Integrator differs (forward Euler vs NEST's
   `consistent_integration`), so recalibrate if the curves differ >5%.
3. Per-neuron `b` heterogeneity: try `SetStatus`/group-param on `b`. If impossible, test splitting each
   population into K sub-groups, or a NESTML/`user_m1` variant.
4. Sign and port semantics: inhibitory weights on the single `izhikevich` port; delay arrays.
5. Benchmark at 12% size (~19M synapses): `Connect` time, `GetConnections(target=)` and `(source=)`
   time, `GetConnectionFloatParam` / `SetConnectionStatus` round trip for 1M synapses, STDP-group
   `Connect` speed, memory (`getCUDAMemHostPeak`).
6. Test whether `Create`/`Connect` is legal after a `Simulate`, and what a pool-activation workaround costs.
7. Spike recording: cumulative-buffer windowing, buffer sizing for SWR bursts.

**Gate:** neuron dynamics match, hook-sized get/set under ~1 min per call at 12%, per-neuron `b` solved.
If not, stay on A and revisit C/D.

### Phase 1: backend shim on CPU (no behaviour change)
- Add `simbackend.py` exposing: `create_izh`, `create_poisson`, `create_spike_gen`, `connect`,
  `set_params`, `simulate`, `record_spikes` (returns senders/times for a window), `record_vm`,
  `get_weights/set_weights(handle)`, `population` wrapper with `.ids`, slicing, `len`.
- Re-route the ~190 direct `nest.*` calls in `replay_scaled.py` and `tiny.py` through it,
  keeping `--backend nest` as default.
- Gate: a fixed-seed 1% run is statistically identical to today (replay rho, DG %active, EC counts;
  not bit-identical, because DG is chaotic).

### Phase 2: NEST GPU backend, single GPU, core network
- Implement `nestgpu` backend. Bring up in order: DG-only, CA3, CA1, then replay with SWR drive.
- Validation at 1% against CPU: replay rho (forward/reverse, SWR window only), DG 2-4% active,
  CA3 completion, population rates. Use rate/z-score readouts, not Jaccard.
- Gate: JOB A-equivalent at 12% on one GPU reproduces replay +0.6/-0.6 band.

### Phase 3: plasticity and cortex
- Port `STCHook`, `MPFCAssocHook`, `SchafferSTDPHook`, homeostasis to cached connection-id arrays
  (build once, reuse every epoch; never source+target `GetConnections`, per the existing rule).
- Replace `DGNeurogenesisHook` creation with pool activation (Phase 0 item 6).
- Run pattern-completion probes in subprocesses.
- Gate: Phase 5 falsification replicates (`--prp-threshold 999`: replay unchanged, consolidation ~0)
  and JOB F (selective EC consolidation, 862/12005 band) on GPU at 12%.

### Phase 4: scale and multi-GPU
- Place populations per rank with `RemoteCreate`/`RemoteConnect` (e.g. DG+CA3 / CA1 / EC+mPFC),
  all ranks calling all construction calls.
- New `run_gpu.sh` (SLURM, one GPU per task, `CUDA_VISIBLE_DEVICES` handled by SLURM). Keep `run.sh` for CPU.
- Scaling ladder: 12% (1 GPU) -> 25% -> 50% -> 100%, recording memory and wall time per epoch at each step.
- Check HDF5 export still works with spikes gathered across ranks.

### Phase 5: docs and housekeeping
- RESULTS.md: new numbered section for the GPU-vs-CPU equivalence and speedup numbers (do not edit older sections).
- Update CLAUDE.md "Running", COMPATIBILITY.md, INSTALL.md, and `run.sh` header.
- Keep CPU path in CI-style smoke test (1%, 1 SWR).

## 4. Risks, ranked

1. Per-neuron `b` (and `a,c,d`) heterogeneity unsupported on stock `izhikevich` (project rule depends on it).
2. No mid-run neuron/synapse creation breaks neurogenesis hook as written.
3. Connection query/set cost, which decides whether hooks speed up at all.
4. No macOS/GPU dev loop; mitigated by the CPU backend and a small GPU box.
5. Numerical differences (integrator, dt, delay rounding) shift calibrated f-I curves and E/I thresholds. CLAUDE.md warns CA1 collapses near its 0.2 Hz E/I threshold, so recalibrate rather than assume.
6. Single-seed pilots: all GPU equivalence claims must be shown at 12% and across seeds.

## 5. Effort (rough)

Phase 0: a few days. Phase 1: ~1 week. Phase 2: 1-2 weeks. Phase 3: 2-3 weeks (hooks, neurogenesis). Phase 4: 1-2 weeks plus queue time.
These are estimates, not measurements.
