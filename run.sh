#!/bin/bash -l
#SBATCH --job-name=HIPPO_NEST
#SBATCH --output=Nest_replay_%A_%a.slurmout
#SBATCH --error=Nest_replay_%A_%a.slurmerr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --partition=gp_bsccs        # CPU partition on MN5  (was: acc = GPU partition)

# Scale: 1pct | 12pct | 100pct
# Override at submission: sbatch --export=ALL,SCALE=12pct run.sh
SCALE=${SCALE:-12pct}
SIM_MS=${SIM_MS:-1000}
OUTDIR="results"          # no trailing slash — kept consistent with --out-hdf5
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export LANG=${LANG:-C.UTF-8}
export LC_ALL=${LC_ALL:-C.UTF-8}
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# NEST uses SetKernelStatus for threads, not OMP_NUM_THREADS.
# Unset to suppress the NEST warning.
unset OMP_NUM_THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "[Slurm] job=$SLURM_JOB_ID ntasks=$SLURM_NTASKS cpus-per-task=$SLURM_CPUS_PER_TASK"
echo "[Slurm] scale=$SCALE sim_ms=$SIM_MS outdir=$OUTDIR"

# NEST sanity check
python3 - <<'PY'
import nest
ks = nest.GetKernelStatus()
mpi = ks.get("mpi_num_processes", ks.get("num_processes", ks.get("total_num_processes", 1)))
thr = ks.get("local_num_threads", ks.get("num_threads", ks.get("threads", 1)))
print("nest", nest.__version__, "mpi_procs", mpi, "local_threads", thr)
PY

mkdir -p "$OUTDIR"

# --distribution flag is only meaningful with multiple tasks; removed for clarity.
# For multi-node (12pct/100pct) runs you will want --ntasks > 1 and can restore it.
srun --cpu-bind=cores \
  python3 -u "replay_scaled.py" \
    --scale    "$SCALE" \
    --sim-ms   "$SIM_MS" \
    --threads  "$SLURM_CPUS_PER_TASK" \
    --out-hdf5 "${OUTDIR}/replay_${SCALE}.h5" \
    --no-figures
