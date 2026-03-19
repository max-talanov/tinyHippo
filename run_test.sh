#!/bin/bash -l
#SBATCH --job-name=HIPPO_NEST
#SBATCH --output=Nest_replay_%A_%a.slurmout
#SBATCH --error=Nest_replay_%A_%a.slurmerr
#SBATCH --nodes=256                # 256 nodes  → 256 × ~256 GB = ~65 TB RAM
#SBATCH --ntasks=1024              # 4 MPI ranks per node
#SBATCH --ntasks-per-node=4        # 4 ranks × 64 GB = 256 GB/node; ~64 GB/rank
#SBATCH --cpus-per-task=28         # 1024 × 28 = 28,672 CPUs total
#SBATCH --time=12:00:00
#SBATCH --partition=gp_bsccs       # CPU partition on MN5

# ---------------------------------------------------------------------------
# WHY 256 NODES / 1024 RANKS?
#
# replay_scaled.py SCALE_CONFIGS["100pct"] specifies:
#   slurm_nodes = 256,  slurm_ntasks = 1024,  estimated runtime = 3–5 h
#
# MEMORY per rank (1024 ranks):
#   Schaffer SUP→CA1_PYR : 460k × 3000 synapses × 48 B =  66 GB total → 66 MB/rank
#   Schaffer DEEP→CA1_PYR: 460k × 1000 synapses × 48 B =  22 GB total → 22 MB/rank
#   CA3 E↔I, CA1 local, sequence chain, neurons, generators: ~200 MB/rank
#   Total estimate: ~300–500 MB/rank  ← well within 64 GB/rank
#
# NEST VP limit (134,217,726 per VP):
#   VPs = 1024 ranks × 28 threads = 28,672
#   Schaffer worst-case: 460k × 3000 / 28,672 = 48,120/VP  ← no problem
#
# PREVIOUS FAILURE (job 37934990):
#   Root cause 1 — MPI not recognized: every rank printed
#     "NEST kernel: 1 MPI rank(s) × 6 thread(s) = 6 VP(s)"
#   meaning each rank allocated the full 845k-neuron network solo.
#   Schaffer alone = 460k × 4000 × 48 B = 88 GB per rank → OOM on 128 GB/rank.
#
#   Root cause 2 — Only 8 nodes (16 ranks): even with working MPI the
#   simulation time would be ~200–320 h, far past the 12 h wall limit.
#
# MPI FIX: load the MPI-enabled NEST module (see module check below).
# The serial NEST build silently ignores MPI even when launched with srun.
# Check available modules with:  module spider nest
# ---------------------------------------------------------------------------

SCALE=${SCALE:-100pct}
SIM_MS=${SIM_MS:-1000}
OUTDIR="results"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export LANG=${LANG:-C.UTF-8}
export LC_ALL=${LC_ALL:-C.UTF-8}
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[Slurm] job=$SLURM_JOB_ID nodes=$SLURM_NNODES ntasks=$SLURM_NTASKS cpus-per-task=$SLURM_CPUS_PER_TASK"
echo "[Slurm] scale=$SCALE sim_ms=$SIM_MS outdir=$OUTDIR"

# ---------------------------------------------------------------------------
# MPI + NEST sanity check — runs in a single process before srun.
# This verifies the loaded NEST module was compiled with MPI support.
# If have_mpi=False, switch to the MPI-enabled module and resubmit.
# ---------------------------------------------------------------------------
python3 - <<'PY'
import os, nest

ks = nest.GetKernelStatus()
have_mpi = ks.get("have_mpi", "key_absent")
mpi_procs = ks.get("mpi_num_processes",
              ks.get("num_processes",
              ks.get("total_num_processes", 1)))
threads = ks.get("local_num_threads",
            ks.get("num_threads",
            ks.get("threads", 1)))

print(f"[pre-flight] NEST {nest.__version__}")
print(f"[pre-flight] have_mpi        = {have_mpi}")
print(f"[pre-flight] mpi_num_procs   = {mpi_procs}  (expect 1 here, many after srun)")
print(f"[pre-flight] local_threads   = {threads}")
print(f"[pre-flight] PMI_RANK        = {os.environ.get('PMI_RANK',        'NOT_SET')}")
print(f"[pre-flight] OMPI_COMM_WORLD_RANK = {os.environ.get('OMPI_COMM_WORLD_RANK', 'NOT_SET')}")

if have_mpi is False or have_mpi == False:
    print()
    print("ERROR: NEST was built WITHOUT MPI support.")
    print("       Load the MPI-enabled module, e.g.:")
    print("         module load NEST/3.9.0-foss-2023a-mpi")
    print("       Then resubmit this job.")
    raise SystemExit(1)

print("[pre-flight] MPI check passed — proceeding to srun.")
PY

# Abort the job if the pre-flight check failed (non-zero exit from python3 block)
if [ $? -ne 0 ]; then
    echo "[ERROR] Pre-flight check failed. Job aborted. See slurmout for details."
    exit 1
fi

mkdir -p "$OUTDIR"

# ---------------------------------------------------------------------------
# Launch the simulation.
# --cpu-bind=cores: pins each rank's threads to physical cores on the same NUMA node.
# OMP_NUM_THREADS is exported above so every rank inherits it from the environment;
# replay_scaled.py also receives --threads as an explicit argument for clarity.
# ---------------------------------------------------------------------------
srun --cpu-bind=cores \
  python3 -u "replay_scaled.py" \
    --scale    "$SCALE" \
    --sim-ms   "$SIM_MS" \
    --threads  "$SLURM_CPUS_PER_TASK" \
    --out-hdf5 "${OUTDIR}/replay_${SCALE}.h5" \
    --no-figures
