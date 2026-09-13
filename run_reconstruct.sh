#!/bin/bash -l
#SBATCH --job-name=HIPPO_RECONSTRUCT
#SBATCH --output=Reconstruct_%A_%a.slurmout
#SBATCH --error=Reconstruct_%A_%a.slurmerr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --time=04:00:00
#SBATCH --partition=gp_bsccs

# JOB H9 -- lightweight 12% EC LII->GC connectivity reconstruction, matched to
# JOB H8's A/B pair (run.sh). Wraps reconstruct_connectivity.py, which drives
# replay_scaled.py's own network-builder functions up to (not through) the
# first nest.Simulate() call -- no full ~10.5h run needed, just the network
# build (52s CA3/CA1/Schaffer + ~1,000s DG, per JOB H8's own timings) PLUS
# one nest.GetConnections(target=GC) call to extract the wiring. That call's
# own cost is NOT well characterized at 144,000 targets -- it took 288.6s at
# 12,000 targets (1% scale) in a local sanity check, and RESULTS.md SS16/JOB H7
# already found this exact call scales with target population size in a way
# that is not simply linear (a 100x larger target cost only ~4x more once
# before). --time is set generously (4h) for this uncertainty; if it finishes
# in the first 20-30min, that is the expected case, not a sign of failure.
#
# Motivation (RESULTS.md SS18): JOB H8's population-level Jaccard metric
# showed DG's identity separation move off zero for the first time at 12%
# scale (0.000 -> +0.004) under --dg-ec-cluster-sigma, but that metric is far
# less sensitive than SS17's single-cell test (each firing cell's ring-distance
# from the active pattern), which is what actually distinguished "clustering
# restores identity" from "K=1 fan-in cut does not" at 1% scale. Reproducing
# that single-cell test at 12% needs the model's real wired connectivity,
# which could not be reconstructed locally (apparent OOM -- 12% scale needs
# ~226M total synapses against a 16GB machine).
#
# Run BOTH arms (control + clustered), same seed, matching JOB H8 exactly
# apart from --dg-ec-cluster-sigma:
#  sbatch --export=ALL run_reconstruct.sh                       # control
#  sbatch --export=ALL,CLUSTER_SIGMA=0.05 run_reconstruct.sh    # clustered
#
# Output: results/ec_gc_connectivity_12pct[_clu<SIGMA>].npz -- an EC LII->GC
# source/target GID array pair (see reconstruct_connectivity.py's docstring),
# analyze with the same "distance from active pattern center" method as
# figures/dg_diagnostic/identity_loss_histograms.png and
# clustered_wiring_identity_test.png.

CLUSTER_SIGMA=${CLUSTER_SIGMA:-}

OUT_SUFFIX=""
EXTRA_FLAGS=""
if [ -n "$CLUSTER_SIGMA" ]; then
  OUT_SUFFIX="_clu${CLUSTER_SIGMA}"
  EXTRA_FLAGS="--dg-ec-cluster-sigma $CLUSTER_SIGMA"
fi

OUTDIR="results"
mkdir -p "$OUTDIR"
OUT="${OUTDIR}/ec_gc_connectivity_12pct${OUT_SUFFIX}.npz"

echo "[Slurm] job=$SLURM_JOB_ID  cpus-per-task=$SLURM_CPUS_PER_TASK"
echo "[Slurm] reconstructing EC LII->GC connectivity at 12% scale -> $OUT"
echo "[Slurm] cluster_sigma=${CLUSTER_SIGMA:-<none, uniform random>}"

export LANG=${LANG:-C.UTF-8}
export LC_ALL=${LC_ALL:-C.UTF-8}
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

unset OMP_NUM_THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores

srun --cpu-bind=cores \
  python3 -u reconstruct_connectivity.py \
    --target ec_lii_gc \
    --out "$OUT" \
    --replay-args \
      --scale 12 --dg --ec-lii --ec-lv --mpfc \
      --n-patterns 3 --n-swr 14 --stc \
      --het 0.30 --het-wcomp 2.3 \
      --w-ec-dg 0.6 --pp-residual 0.9 --dg-delay-jitter 4.0 \
      --pattern-source ec-lii --place-field-sigma 0.15 \
      --ec-pattern-base-rate 20 --ec-pattern-peak-rate 800 \
      --ec-pattern-weight 1.5 --novel-pattern-onset 8 \
      --seed 202 --threads "$SLURM_CPUS_PER_TASK" \
      --out-hdf5 "${OUTDIR}/throwaway_reconstruct${OUT_SUFFIX}.h5" --no-figures \
      $EXTRA_FLAGS

echo "[Slurm] done."
