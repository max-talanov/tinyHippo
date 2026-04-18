#!/bin/bash -l
#SBATCH --job-name=HIPPO_NEST
#SBATCH --output=Nest_replay_%A_%a.slurmout
#SBATCH --error=Nest_replay_%A_%a.slurmerr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --time=08:00:00
#SBATCH --partition=gp_bsccs

# Normal run (Phase 2 + Phase 3):
#   sbatch --export=ALL,SCALE=12,N_SWR=14 run.sh
# Phase 3 only (no STC):
#   sbatch --export=ALL,SCALE=12,N_SWR=1,EC_LV=1,MPFC=1,NO_STC=1 run.sh
# Phase 5 falsification:
#   sbatch --export=ALL,SCALE=12,N_SWR=14,PRP_THRESHOLD=999 run.sh

SCALE=${SCALE:-12}
EC_LII_K=${EC_LII_K:-50}
N_SWR=${N_SWR:-14}
EPOCH_MS=${EPOCH_MS:-1000}
PRP_THRESHOLD=${PRP_THRESHOLD:-14.0}
EC_LV=${EC_LV:-1}      # 1=enable Phase 3 EC LV, 0=disable
MPFC=${MPFC:-1}         # 1=enable mPFC module, 0=disable
NO_STC=${NO_STC:-0}     # 1=skip STC hook (useful for Phase 3-only runs)
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
echo "[Slurm] scale=${SCALE}%  n_swr=$N_SWR  epoch_ms=$EPOCH_MS  prp_threshold=$PRP_THRESHOLD"
echo "[Slurm] ec_lv=${EC_LV}  mpfc=${MPFC}  no_stc=${NO_STC}"

python3 - <<'PY'
import nest
ks = nest.GetKernelStatus()
mpi = ks.get("mpi_num_processes", ks.get("num_processes", ks.get("total_num_processes", 1)))
thr = ks.get("local_num_threads", ks.get("num_threads", ks.get("threads", 1)))
print(f"nest {nest.__version__}  mpi_procs={mpi}  local_threads={thr}")
PY

mkdir -p "$OUTDIR"

# Tag output filename with active phases
PHASE_TAG=""
[ "$EC_LV" = "1" ]  && PHASE_TAG="${PHASE_TAG}_lv"
[ "$MPFC"  = "1" ]  && PHASE_TAG="${PHASE_TAG}_mpfc"
[ "${PRP_THRESHOLD%.*}" -gt 100 ] 2>/dev/null && PHASE_TAG="${PHASE_TAG}_ph5"

OUTFILE="${OUTDIR}/replay_${SCALE}pct_stc${PHASE_TAG}.h5"
echo "[Slurm] output → $OUTFILE"

# Build optional flag list
OPTIONAL_FLAGS=""
[ "$NO_STC" != "1" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --stc --n-swr $N_SWR --epoch-ms $EPOCH_MS --prp-threshold $PRP_THRESHOLD"
[ "$EC_LV"  = "1" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --ec-lv"
[ "$MPFC"   = "1" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --mpfc"

srun --cpu-bind=cores \
  python3 -u "replay_scaled.py" \
    --scale       "$SCALE" \
    --threads     "$SLURM_CPUS_PER_TASK" \
    --out-hdf5    "$OUTFILE" \
    --ec-lii \
    --ec-lii-k    "$EC_LII_K" \
    $OPTIONAL_FLAGS \
    --no-figures
