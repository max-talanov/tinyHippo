#!/bin/bash -l
#SBATCH --job-name=HIPPO_NEST
#SBATCH --output=Nest_replay_%A_%a.slurmout
#SBATCH --error=Nest_replay_%A_%a.slurmerr
#SBATCH --nodes=8                  # 8 nodes  → ~2 TB RAM total (8 × ~256 GB)
#SBATCH --ntasks=16                # 2 MPI ranks per node  ← KEY FIX (was 8)
#SBATCH --ntasks-per-node=2        # 2 ranks share each node's 256 GB (~128 GB/rank)
#SBATCH --cpus-per-task=6          # 16 × 6 = 96 CPUs — same budget as before
#SBATCH --time=12:00:00
#SBATCH --partition=gp_bsccs       # CPU partition on MN5

# ---------------------------------------------------------------------------
# WHY 16 RANKS?
#
# Previous failure: "Too many connections: at most 134,217,726 per VP"
#
# NEST stores connections per Virtual Process (VP = MPI_ranks × threads).
# If NEST silently falls back to 1 thread, VPs = 8 × 1 = 8, and the
# Schaffer CA3_SUP→CA1_PYR call alone needs 460,000×3,000/8 = 172,500,000
# connections/VP — 28% over the hard 134,217,726 limit.
#
# With 16 ranks worst-case (threads=1): VPs=16 → 86,250,000/VP — safely under.
# With 16 ranks best-case (threads=6): VPs=96 → 14,375,000/VP — well under.
#
# Memory: 2 ranks share one 256 GB node → ~128 GB per rank, ~53k neurons/rank.
# This mirrors the 12pct run that succeeded on 1 node. ✓
#
# ---------------------------------------------------------------------------
# IF THIS STILL FAILS — check the new slurmout for this line:
#   "NEST kernel: N MPI rank(s) × T thread(s) = VP VP(s)"
# and look for "[conn-check]" lines showing cumulative %  of the 134M limit.
#
# FALLBACK OPTIONS:
#
#   If VPs are confirmed low (e.g. threads=1), add more ranks:
#     sbatch --nodes=8 --ntasks=32 --ntasks-per-node=4 --cpus-per-task=3 run_test.sh
#     → 32 VPs worst-case; schaffer_sup = 43,125,000/VP ✓
#
#   If Schaffer indegrees need trimming (replay_scaled.py INDEGREES dict):
#     Safe maximum at VPs=N: K_sup + K_deep < 134,217,726 × N / 460,000
#     E.g. at 16 VPs: max total K < 4,667  (current: 4,000 — already fine)
#     E.g. at  8 VPs: max total K < 2,334  (reduce schaffer_sup_pyr to ≤1700)
# ---------------------------------------------------------------------------

SCALE=${SCALE:-100pct}
SIM_MS=${SIM_MS:-1000}
OUTDIR="results"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export LANG=${LANG:-C.UTF-8}
export LC_ALL=${LC_ALL:-C.UTF-8}
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

unset OMP_NUM_THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "[Slurm] job=$SLURM_JOB_ID nodes=$SLURM_NNODES ntasks=$SLURM_NTASKS cpus-per-task=$SLURM_CPUS_PER_TASK"
echo "[Slurm] scale=$SCALE sim_ms=$SIM_MS outdir=$OUTDIR"

# NEST sanity check — single process, pre-srun (rank 0 only)
python3 - <<'PY'
import nest
ks = nest.GetKernelStatus()
mpi = ks.get("mpi_num_processes", ks.get("num_processes", ks.get("total_num_processes", 1)))
thr = ks.get("local_num_threads", ks.get("num_threads", ks.get("threads", 1)))
print("nest", nest.__version__, "mpi_procs", mpi, "local_threads", thr)
PY

mkdir -p "$OUTDIR"

# env -u OMP_NUM_THREADS: prevents Slurm from re-injecting OMP_NUM_THREADS
# inside each rank (Slurm sets it from cpus-per-task after our unset above).
srun --cpu-bind=cores env -u OMP_NUM_THREADS \
  python3 -u "replay_scaled.py" \
    --scale    "$SCALE" \
    --sim-ms   "$SIM_MS" \
    --threads  "$SLURM_CPUS_PER_TASK" \
    --out-hdf5 "${OUTDIR}/replay_${SCALE}.h5" \
    --no-figures
