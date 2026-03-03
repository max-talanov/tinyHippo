#!/bin/bash
# =============================================================================
# run_replay_mn5.sh  —  submit with: sbatch run_replay_mn5.sh
# =============================================================================
# SLURM job script for bidirectional_replay_watson2025_scaled.py
# Target: MareNostrum5 (BSC), gpp partition
#   Node spec: 2x Intel Xeon Platinum 8480+ (Sapphire Rapids)
#              112 cores/node  |  256 GB RAM/node
#
# NEST parallelism: 4 MPI ranks/node x 28 OMP threads/rank = 112 cores/node
#
# QUICK SUBMISSION GUIDE
# ----------------------
#   1pct  (test, 1 node,   <30 min):
#     sbatch --nodes=1   --ntasks=4    --time=00:30:00 run_replay_mn5.sh
#
#   12pct (dev,  16 nodes, ~2 h):
#     sbatch --nodes=16  --ntasks=64   --time=02:00:00 run_replay_mn5.sh
#
#   100pct (prod, 256 nodes, ~24 h):
#     sbatch --nodes=256 --ntasks=1024 --time=24:00:00 run_replay_mn5.sh
#
#   Override scale or sim duration at submission time:
#     sbatch --export=ALL,SCALE=100pct,SIM_MS=5000 --nodes=256 --ntasks=1024 run_replay_mn5.sh
# =============================================================================

# ---- Default resources (12% scale; change per quick guide above) -----------
#SBATCH --nodes=16
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=28
#SBATCH --time=02:00:00

# ---- Cluster config --------------------------------------------------------
#SBATCH --partition=gpp
#SBATCH --account=YOUR_ACCOUNT        # <-- replace with your BSC project account
#SBATCH --job-name=bidir_replay
#SBATCH --output=logs/replay_%j.out
#SBATCH --error=logs/replay_%j.err
#SBATCH --exclusive

# ---- Optional email --------------------------------------------------------
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your@email.com

# =============================================================================
# Runtime variables  (override any of these via --export=ALL,VAR=val)
# =============================================================================

# Scale: 1pct | 12pct | 100pct
SCALE=${SCALE:-12pct}

# Simulation duration in ms
SIM_MS=${SIM_MS:-1000}

# Set to 1 to skip figure generation (recommended for 12pct and 100pct)
NO_FIGURES=${NO_FIGURES:-1}

# OpenMP threads per MPI rank = cpus-per-task (auto-set by SLURM)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-28}

# OpenMP pinning — critical for performance on dual-socket MN5 nodes
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Reduce memory fragmentation with many threads
export MALLOC_ARENA_MAX=4

# =============================================================================
# Modules
# =============================================================================

module purge

# Adjust module names to your MN5 software environment:
module load nest/3.9.0
module load python/3.10

# If using a conda/venv instead of modules, comment the lines above and use:
# source $HOME/envs/nest_env/bin/activate

# Sanity check
python3 -c "import nest; print('NEST OK:', nest.version())" || {
    echo "ERROR: NEST not importable. Check module or venv."
    exit 1
}

# =============================================================================
# Paths  — adjust SCRIPT_DIR to your project location on MN5
# =============================================================================

SCRIPT_DIR=$HOME/memHippo/tinyHippo
SCRIPT=$SCRIPT_DIR/bidirectional_replay_watson2025_scaled.py
mkdir -p $SCRIPT_DIR/logs

# =============================================================================
# Run info
# =============================================================================

echo "========================================================"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Scale       : $SCALE"
echo "  Nodes       : $SLURM_JOB_NUM_NODES"
echo "  MPI ranks   : $SLURM_NTASKS"
echo "  Threads/rank: $OMP_NUM_THREADS"
echo "  Sim [ms]    : $SIM_MS"
echo "  No figures  : $NO_FIGURES"
echo "  Start       : $(date)"
echo "========================================================"

# =============================================================================
# Launch
# =============================================================================

ARGS="--scale $SCALE --sim-ms $SIM_MS"
[ "$NO_FIGURES" -eq 1 ] && ARGS="$ARGS --no-figures"

srun --mpi=pmix python3 $SCRIPT $ARGS
EXIT=$?

echo "========================================================"
echo "  End    : $(date)"
echo "  Status : $EXIT"
echo "========================================================"

exit $EXIT
