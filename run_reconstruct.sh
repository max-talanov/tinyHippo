#!/bin/bash -l
#SBATCH --job-name=HIPPO_RECONSTRUCT
#SBATCH --output=Reconstruct_%A_%a.slurmout
#SBATCH --error=Reconstruct_%A_%a.slurmerr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --time=04:00:00
#SBATCH --partition=gp_bsccs

# JOB H9 -- lightweight 12% EC LII->GC connectivity reconstruction. Wraps
# reconstruct_connectivity.py, which drives replay_scaled.py's own network-
# builder functions up to (not through) the first nest.Simulate() call -- no
# full ~10.5h run needed.
#
# FIRST ATTEMPT (2026-09-13, job 45812883) hit its 4h wall without ever
# reaching the EC LII->GC extraction step: --replay-args mirrored JOB H8
# exactly (--ec-lv --mpfc --stc), which drove main() through the STC hook's
# one-time CA1->EC synapse scan (3,179.6s) and EC LV's lesion cache
# (1,907.3s) -- 85+ minutes on TWO builds this reconstruction does not need
# at all. EC LII->GC's own wiring is already complete by the time
# build_dg_module() returns, which happens BEFORE EC LV, mPFC, or the STC
# hook are built (see main()'s Phase ordering) -- none of those three flags
# can affect the connectivity this job extracts.
#
# FIX: --ec-lv/--mpfc/--stc dropped entirely below. This also means the
# reconstructed connectivity is a DIFFERENT (not bit-identical) draw from
# whatever JOB H8's real run used for the CONTROL arm specifically --
# --stc changes n_epochs (14 -> 1), which changes how many scaffold/SWR
# generators build_replay_network creates BEFORE build_dg_module runs,
# which can shift NEST's shared kernel RNG's position by the time the
# perforant path's fixed_indegree draw happens. That is fine for this
# job's purpose: we need a STATISTICALLY representative K=50 uniform-random
# sample under the same seed-derived EC LII place-field centers (which come
# from an INDEPENDENT numpy RNG, seed+13, untouched by any of this) to run
# SS17's distance-from-pattern-center test -- not JOB H8's literal specific
# synapse list. The CLUSTERED arm is unaffected either way: its sampling
# uses its own independent RNG (seed+97), never the shared kernel one.
#
# Expected cost now: ~17min through DG module build (50.5s CA3/CA1/Schaffer
# + 5.2s EC LII + 932.8s DG, all confirmed from the first attempt's own log
# before it got to EC LV) PLUS one nest.GetConnections(target=GC) call whose
# cost at 144,000 targets is still not characterized (288.6s at 12,000
# targets / 1% scale locally; RESULTS.md SS16/JOB H7 found this call does
# NOT scale simply with target size. Fixed 2026-09-14 (reconstruct_connectivity.py):
# query by source (EC LII, ~12,005 cells) instead of target (GC, ~30M
# incoming synapses across every DG projection) -- dropped from 196-289s to
# 6.5s at 1% scale in a local check. --time set to 4h regardless, as a
# safety margin against this call's cost at 144,000-scale still not being
# fully characterized end-to-end.
#
# Run BOTH arms (control + clustered), same seed, matching JOB H8 exactly
# apart from --dg-ec-cluster-sigma (and now --ec-lv/--mpfc/--stc, dropped
# from both arms identically, so the A/B comparison stays apples-to-apples):
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
      --scale 12 --dg --ec-lii \
      --n-patterns 3 \
      --het 0.30 --het-wcomp 2.3 \
      --w-ec-dg 0.6 --pp-residual 0.9 --dg-delay-jitter 4.0 \
      --pattern-source ec-lii --place-field-sigma 0.15 \
      --ec-pattern-base-rate 20 --ec-pattern-peak-rate 800 \
      --ec-pattern-weight 1.5 --novel-pattern-onset 8 \
      --seed 202 --threads "$SLURM_CPUS_PER_TASK" \
      --out-hdf5 "${OUTDIR}/throwaway_reconstruct${OUT_SUFFIX}.h5" --no-figures \
      $EXTRA_FLAGS

echo "[Slurm] done."
