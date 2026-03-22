#!/bin/bash -l
#SBATCH --job-name=HIPPO_NEST
#SBATCH --output=Nest_replay_%A_%a.slurmout
#SBATCH --error=Nest_replay_%A_%a.slurmerr
#SBATCH --nodes=1
#SBATCH --ntasks=1          # 1 MPI task — NEST on MN5 is not MPI-compiled;
                            # multiple tasks all report rank=0 and race to write
                            # the same HDF5 file.  Use OpenMP threads instead.
#SBATCH --cpus-per-task=50  # all 50 physical cores → NEST threads
#SBATCH --time=06:00:00
#SBATCH --partition=gp_bsccs        # CPU partition on MN5

# Scale: 1pct | 12pct | 100pct
# Override at submission: sbatch --export=ALL,SCALE=12pct run.sh
SCALE=${SCALE:-12pct}
SIM_MS=${SIM_MS:-1000}
OUTDIR="results"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export LANG=${LANG:-C.UTF-8}
export LC_ALL=${LC_ALL:-C.UTF-8}
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# HDF5 file locking can fail on Lustre/GPFS parallel filesystems.
# Disabling it is safe for single-writer workflows (one rank writes).
export HDF5_USE_FILE_LOCKING=FALSE

# NEST uses SetKernelStatus for threads, not OMP_NUM_THREADS.
unset OMP_NUM_THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "[Slurm] job=$SLURM_JOB_ID ntasks=$SLURM_NTASKS cpus-per-task=$SLURM_CPUS_PER_TASK"
echo "[Slurm] scale=$SCALE sim_ms=$SIM_MS outdir=$OUTDIR"
echo "[Slurm] NOTE: 1 MPI task + $SLURM_CPUS_PER_TASK OpenMP threads"
echo "[Slurm] (NEST on MN5 is not MPI-compiled; ntasks>1 causes HDF5 write races)"

# NEST sanity check
python3 - <<'PY'
import nest
ks = nest.GetKernelStatus()
mpi = ks.get("mpi_num_processes", ks.get("num_processes", ks.get("total_num_processes", 1)))
thr = ks.get("local_num_threads", ks.get("num_threads", ks.get("threads", 1)))
print("nest", nest.__version__, "mpi_procs", mpi, "local_threads", thr)
if mpi == 1:
    print("NOTE: NEST reports mpi_procs=1 -- OpenMP-only mode (expected on this build)")
PY

mkdir -p "$OUTDIR"

srun --cpu-bind=cores \
  python3 -u "replay_scaled.py" \
    --scale    "$SCALE" \
    --sim-ms   "$SIM_MS" \
    --threads  "$SLURM_CPUS_PER_TASK" \
    --out-hdf5 "${OUTDIR}/replay_${SCALE}.h5" \
    --no-figures
