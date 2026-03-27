#!/bin/bash -l
#SBATCH --job-name=HIPPO_NEST
#SBATCH --output=Nest_replay_%A_%a.slurmout
#SBATCH --error=Nest_replay_%A_%a.slurmerr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --time=04:00:00      # 2h is plenty for 12% + EC LII K=50; increase for larger K
#SBATCH --partition=gp_bsccs

# Scale and EC options
# Override: sbatch --export=ALL,SCALE=12,EC_LII_K=50 run.sh
SCALE=${SCALE:-12}
SIM_MS=${SIM_MS:-1000}
EC_LII_K=${EC_LII_K:-50}     # K=50 is safe; only raise after confirming fast connect
OUTDIR="results"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export LANG=${LANG:-C.UTF-8}
export LC_ALL=${LC_ALL:-C.UTF-8}
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

unset OMP_NUM_THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "[Slurm] job=$SLURM_JOB_ID  ntasks=$SLURM_NTASKS  cpus-per-task=$SLURM_CPUS_PER_TASK"
echo "[Slurm] scale=${SCALE}%  sim_ms=$SIM_MS  ec_lii_k=$EC_LII_K  outdir=$OUTDIR"

# NEST sanity check
python3 - <<'PY'
import nest
ks = nest.GetKernelStatus()
mpi = ks.get("mpi_num_processes", ks.get("num_processes", ks.get("total_num_processes", 1)))
thr = ks.get("local_num_threads", ks.get("num_threads", ks.get("threads", 1)))
print(f"nest {nest.__version__}  mpi_procs={mpi}  local_threads={thr}")
PY

mkdir -p "$OUTDIR"

srun --cpu-bind=cores \
  python3 -u "replay_scaled.py" \
    --scale      "$SCALE" \
    --sim-ms     "$SIM_MS" \
    --threads    "$SLURM_CPUS_PER_TASK" \
    --out-hdf5   "${OUTDIR}/replay_${SCALE}pct.h5" \
    --ec-lii \
    --ec-lii-k   "$EC_LII_K" \
    --no-figures
