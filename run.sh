#!/bin/bash -l
#SBATCH --job-name=HIPPO_NEST
#SBATCH --output=Nest_replay_%A_%a.slurmout
#SBATCH --error=Nest_replay_%A_%a.slurmerr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --time=04:00:00
#SBATCH --partition=gp_bsccs

# Scale and consolidation options
# sbatch --export=ALL,SCALE=12,N_SWR=7 run.sh
SCALE=${SCALE:-12}
EC_LII_K=${EC_LII_K:-50}
N_SWR=${N_SWR:-7}          # number of SWR epochs for consolidation
EPOCH_MS=${EPOCH_MS:-1000}  # duration of each epoch in ms
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
echo "[Slurm] scale=${SCALE}%  ec_lii_k=$EC_LII_K  n_swr=$N_SWR  epoch_ms=$EPOCH_MS"

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
    --threads    "$SLURM_CPUS_PER_TASK" \
    --out-hdf5   "${OUTDIR}/replay_${SCALE}pct_stc.h5" \
    --ec-lii \
    --ec-lii-k   "$EC_LII_K" \
    --stc \
    --n-swr      "$N_SWR" \
    --epoch-ms   "$EPOCH_MS" \
    --no-figures
