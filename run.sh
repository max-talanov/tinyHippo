#!/bin/bash -l
#SBATCH --job-name=BIDIR_REPLAY
#SBATCH --output=Nest_replay_%A_%a.slurmout
#SBATCH --error=Nest_replay_%A_%a.slurmerr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --partition=acc

# Scale to run: 1pct | 12pct | 100pct
# Override at submission time with: sbatch --export=ALL,SCALE=12pct run_replay.sh
SCALE=${SCALE:-1pct}

# Simulation duration in ms
SIM_MS=${SIM_MS:-1000}

# Output directory
OUTDIR="results_${SCALE}/"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export LANG=${LANG:-C.UTF-8}
export LC_ALL=${LC_ALL:-C.UTF-8}
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# OpenMP pinning — important for performance on multi-socket nodes
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

srun --cpu-bind=cores --distribution=block:block \
  python3 -u "$SCRIPT_DIR/replay_scaled.py" \
    --scale    "$SCALE" \
    --sim-ms   "$SIM_MS" \
    --threads  "$SLURM_CPUS_PER_TASK" \
    --no-figures
