#!/bin/bash -l
#SBATCH --job-name=HIPPO_NEST
#SBATCH --output=Nest_replay_%A_%a.slurmout
#SBATCH --error=Nest_replay_%A_%a.slurmerr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --time=20:00:00
#SBATCH --partition=gp_bsccs

# Standard Phase 2+3 run (default)
#  sbatch --export=ALL,SCALE=12,N_SWR=14 run.sh

# Phase 3 only (no STC consolidation, fast run for testing the loop)
#  sbatch --export=ALL,SCALE=12,N_SWR=1,NO_STC=1 run.sh

# Phase 5 falsification with Phase 3
#  sbatch --export=ALL,SCALE=12,N_SWR=14,PRP_THRESHOLD=999 run.sh

# Phase 4 single-alpha homeostasis
#  sbatch --export=ALL,SCALE=25,N_SWR=14,PRP_THRESHOLD=3.5,HOMEOSTASIS=1,HOMEO_ALPHA=0.75 run.sh

# Phase 4 alpha-sweep (3 alphas in one job, ~5h vs ~15h for 3 separate jobs)
#  sbatch --export=ALL,SCALE=25,N_SWR=14,PRP_THRESHOLD=3.5,HOMEOSTASIS=1,ALPHA_SWEEP=0.50,0.75,0.90 run.sh

# ---- Phase 6.2 validation at 12% (three new capabilities) -------------------
# JOB A — bidirectional replay + DG pattern separation, no consolidation stack
#   (isolates the two so-far-1%-only results at scale; fastest, cleanest read)
#  sbatch --export=ALL,SCALE=12,DG=1,NO_STC=1,EC_LII=0,EC_LV=0,MPFC=0 run.sh
#
# JOB B — CA3 pattern completion probe (separate run mode; ignores STC/cortex)
#  sbatch --export=ALL,SCALE=12,PATTERN_COMPLETION=1 run.sh
#
# JOB C — full integrated stack WITH the real DG (run only after A+B look good)
#  sbatch --export=ALL,SCALE=12,DG=1,N_SWR=14 run.sh

# ---- Test 3 at 12% : hippocampus-independent cortical recall ---------------
# The 1% result (consolidated 0.214 vs control 0.071) is 3 cells vs 1 cell on
# one seed -- too small to trust. At 12% mPFC is 1440 cells and an assembly
# ~250, so recovered counts are ~30 vs ~10.
#
# JOB F — sanity after the cortical sparsity retune.  DONE, PASSED (2026-08-17):
#   replay +0.63/-0.66, DG 1.55%, mPFC 1.19 Hz, and consolidation is now
#   SELECTIVE (862/12005 EC cells, weight CV 0.17) instead of saturated at 98%.
#  sbatch --export=ALL,SCALE=12,DG=1,N_PATTERNS=2,N_SWR=14 run.sh
#
# JOB D — Test 3, consolidated (repeat for SEED=101,202,303)
#   The 2026-08-17 attempt hit the 20 h wall AFTER finishing all 16 epochs in
#   3.0 h: >9 h went into lesion_hippocampus()'s GetConnections(source=,target=),
#   a whole-kernel scan over ~19M synapses. Fixed (the lesion now reuses the STC
#   hook's handles), so these settings are unchanged and now fit in ~4 h.
#  sbatch --export=ALL,SCALE=12,DG=1,N_PATTERNS=2,TRAIN_PATTERN=0,N_SWR=16,\
#SCHAFFER_K=200,DELAY_JITTER=4.0,SCHAFFER_STDP=1,CORTICAL_RECALL=1,SEED=101 run.sh
#
# JOB E — Test 3 control, no cortical plasticity (same seeds as D)
#  sbatch --export=ALL,SCALE=12,DG=1,N_PATTERNS=2,TRAIN_PATTERN=0,N_SWR=16,\
#SCHAFFER_K=200,DELAY_JITTER=4.0,SCHAFFER_STDP=1,CORTICAL_RECALL=1,NO_MPFC_ASSOC=1,SEED=101 run.sh
#
# ---- JOB H : does heterogeneity make DG SELECTIVE? (the engram question) ----
# Background (RESULTS.md §13): the model has no engram because DG is sparse but
# NOT selective -- over 7 epochs replaying the SAME pattern, 83% of granule
# cells fire in exactly 1 of 7 windows and none in all 7. Cause: the only
# pattern-carrying input (EC LII perforant path) supplies 0.6% of granule drive,
# the other 99.4% being a Poisson residual resampled every window. Raising the
# perforant weight was blocked by a synchrony ceiling -- with scalar weights and
# delays a granule cell's K=50 inputs land in one instant, so K*w must stay
# under the 20 mV gap.
#
# --het gives every cell and every synapse a distribution, which lifts that
# ceiling. H1 tests heterogeneity alone; H2 additionally spends the headroom on
# the perforant path. Compare both against JOB F (homogeneous, same settings).
#
# NOT rate-matched: five attempts to calibrate --het-wcomp at 1% each uncovered
# a different failure mode, and the 1% homogeneous baseline cannot anchor the
# comparison anyway (rho_rev -0.079 there vs -0.656 at 12%). Read H1/H2 for
# whether DG becomes SELECTIVE, and treat rate differences as expected.
#
# JOB H1 — heterogeneity alone
#  sbatch --export=ALL,SCALE=12,DG=1,N_PATTERNS=2,N_SWR=14,HET=0.30,HET_WCOMP=2.3 run.sh
#
# JOB H2 — heterogeneity + perforant path through the door
#  sbatch --export=ALL,SCALE=12,DG=1,N_PATTERNS=2,N_SWR=14,HET=0.30,HET_WCOMP=2.3,\
#W_EC_DG=1.2,PP_RESIDUAL=0.5,DG_DELAY_JITTER=4.0 run.sh
#
# JOB H3 — H2 rerun after the DG background fix.  H1/H2 (2026-08-22) produced a
#   real CORE SET for the first time (64-69 granule cells firing in >=6/7
#   same-pattern windows, against 1 in the homogeneous model) but ZERO pattern
#   selectivity (Jaccard within-between -0.002). Cause: the compensated
#   mossy-cell background put DG baskets at 22.3 Hz and clamped granule cells to
#   0.45% active, so the perforant path could not influence WHICH cells fire --
#   H2's 8x perforant weight changed nothing. With GC clamped, the cells that
#   escape are the intrinsically most excitable ones, which is why the core set
#   is reproducible but identical across patterns (H1/H2 core overlap 32 cells
#   despite an 8x input difference). Now that no DG background drive is
#   compensated, DG should return to 2-4% and the perforant path gets a say.
#  sbatch --export=ALL,SCALE=12,DG=1,N_PATTERNS=2,N_SWR=14,HET=0.30,HET_WCOMP=2.3,\
#W_EC_DG=1.2,PP_RESIDUAL=0.5,DG_DELAY_JITTER=4.0,SEED=202 run.sh
#
# READ FIRST in both: DG active fraction must be 2-4%. If it is 0.00% the
# granule population has been extinguished by basket feedback (seen at 1% when
# the basket background drive was wrongly compensated) and nothing else in the
# run means anything. Then check whether granule cells develop a CORE SET --
# cells firing in >=6/7 same-pattern windows, versus 1 observed today.
#
# CHECK BEFORE BELIEVING ANY RECALL NUMBER: the printed pre-cue baseline must be
# ~0. If it is not, the priming is firing cells by itself and completion is
# meaningless -- lower CR_PRIME_RATE (120 works at 1%; 250 did not).
#
# JOB H4 — H3 rerun after two more fixes (2026-08-23):
#   (1) `b` is a bifurcation parameter, not a scalable-heterogeneity gain --
#       a symmetric spread on `b` was pushing cells past their rheobase
#       saddle-node into tonic firing with zero input. DG basket max
#       189 -> 31.5 Hz, mean 21.5 -> 5.2 Hz at 1%.
#   (2) `--w-cv` was shadowed since 1266ba3 (CLI default 0.0 overrode the
#       function default of None), so H1/H2/H3 all ran with cv=0.0 despite
#       --het 0.30 -- no DG synapse ever had real weight heterogeneity.
#   Also: `pattern_discrimination()` (the within/between Jaccard table) never
#   included DG GC -- only CA3/CA1/EC-LII/EC-LV/mPFC were checked, so H1-H3's
#   "DG selectivity" numbers all came from an offline one-off script, not the
#   pipeline. DG GC is now in that table automatically.
#
#   HET_WCOMP must stay 2.3 explicitly -- the CLI default flipped to 1.0 on
#   2026-08-19 (0b71908) for unrelated reasons, and 1.0 starves the WHOLE
#   network (heterogeneity's gain loss is compensated model-wide by this
#   knob; the DG-background bug from H3 is fixed separately via a hardcoded
#   compensate=False on those synapses, so this flag no longer needs to stay
#   low for DG's sake). Confirmed locally at 1%: wcomp=1.0 crashed CA1 PYR
#   9.6 -> <1 Hz and silenced EC LII (0 spikes in the scored SWR windows,
#   i.e. no perforant-path signal could reach DG regardless of tuning).
#   At wcomp=2.3 with the real EC LII input restored, W_EC_DG=1.2 (H2/H3's
#   value) over-drove DG to 13%/4% active; W_EC_DG=0.6 landed back in band
#   locally (3.05% fwd / 4.30% rev, CA1 PYR 9.4 Hz) -- starting point below,
#   re-bracket at 12% since 1%<->12% is not rate-matched (see JOB H header).
#  sbatch --export=ALL,SCALE=12,DG=1,N_PATTERNS=2,N_SWR=14,HET=0.30,HET_WCOMP=2.3,\
#W_EC_DG=0.6,PP_RESIDUAL=0.9,DG_DELAY_JITTER=4.0,SEED=202 run.sh
#
# JOB H5 — the encoding direction (Phase 7), not just the replay direction.
#   All of JOB H1-H4 injected the pattern directly into CA3 SUP -- the
#   REPLAY/consolidation direction (CA3->CA1->EC->cortex), which is what this
#   whole codebase was built to study. It is NOT the encoding direction the
#   user actually asked about (sensory/cortex->EC->DG->CA3), and DG can never
#   be tested for its classical pattern-separation role while CA3 is being
#   handed the answer directly -- confirmed structurally: EC LII's only
#   afferent is CA1 (build_ec_lii), so DG's perforant path only ever echoes
#   what CA3 already decided, never an independent input.
#
#   PATTERN_SOURCE=ec-lii reverses this: the pattern-defining drive attaches
#   to EC LII instead, as place-field-like tuning curves on a ring (each EC
#   LII cell gets a fixed random preferred location; each pattern is a ring
#   location; PLACE_FIELD_SIGMA controls how much adjacent patterns overlap
#   -- the "similar but not identical" input DG pattern separation is
#   actually supposed to be tested against, not disjoint groups). CA3 gets
#   NO direct injection in this mode -- it only ever learns the pattern via
#   the existing EC LII->DG perforant path -> mossy fibre route. No new
#   plasticity was added: the mossy fibre is already a fixed, sparse,
#   high-weight "detonator" projection, so if DG genuinely produces a
#   pattern-specific granule code, different CA3 cells get detonated purely
#   from the existing wiring.
#
#   EC_PATTERN_BASE_RATE / EC_PATTERN_PEAK_RATE / PLACE_FIELD_SIGMA are all
#   UNCALIBRATED -- bracket locally (NEST now runs locally, see
#   [[phase-plan-memhippo]] memory) before spending an MN5 allocation, the
#   same way W_EC_DG/PP_RESIDUAL were bracketed for JOB H4. Read the
#   `pattern_discrimination` table's EC LII row as a positive control first
#   (it should show strong separation since it is pattern-locked by
#   construction) -- if EC LII itself doesn't discriminate, the drive is
#   miscalibrated and the DG/CA3 rows below it mean nothing yet.
#  sbatch --export=ALL,SCALE=1,DG=1,N_PATTERNS=2,N_SWR=8,PATTERN_SOURCE=ec-lii,\
#PLACE_FIELD_SIGMA=0.15,EC_PATTERN_BASE_RATE=20,EC_PATTERN_PEAK_RATE=200 run.sh
#
# JOB H6 — Phase 8: does DG neurogenesis make DG selective? At 1% the EC LII
#   drive was too weak against the existing CA1->EC pathway (K=1 vs K=50) --
#   bracketed locally to EC_PATTERN_PEAK_RATE=800, EC_PATTERN_WEIGHT=1.5,
#   which finally gave EC LII its own real separation (sep +0.045 to +0.16
#   depending on run) -- the first positive signal anywhere in this whole
#   investigation. DG still did not inherit it (sep ~0, noise-level).
#
#   Neurogenesis (new hyperexcitable, under-inhibited GC cohort each epoch,
#   see build_dg_neurogenesis_hook) was added specifically to test whether
#   that closes the gap. Matched-seed local comparison (1%, N_SWR=8, seed
#   202, otherwise identical): DG sep went -0.007/+0.018 (id/timing) ->
#   -0.021/-0.017 with neurogenesis on -- nominally worse on both axes, no
#   directional improvement, verdict unchanged (identity NONE, timing EC
#   LII only). Read as noise, not a real effect, AT THIS SCALE.
#
#   Local is 12,000 granule cells over 8 epochs (120 new cells/epoch) -- thin
#   for both the heterogeneity distributions and the cohort-vs-cohort
#   statistics this mechanism depends on. N_SWR bumped to 14 (7 epochs/
#   pattern, matching JOB H1-H5) for real statistical power. Run the control
#   (no neurogenesis) and the neurogenesis arm as a matched pair, same seed:
#  sbatch --export=ALL,SCALE=12,DG=1,N_PATTERNS=2,N_SWR=14,HET=0.30,HET_WCOMP=2.3,\
#W_EC_DG=0.6,PP_RESIDUAL=0.9,DG_DELAY_JITTER=4.0,PATTERN_SOURCE=ec-lii,\
#PLACE_FIELD_SIGMA=0.15,EC_PATTERN_BASE_RATE=20,EC_PATTERN_PEAK_RATE=800,\
#EC_PATTERN_WEIGHT=1.5,SEED=202 run.sh
#  sbatch --export=ALL,SCALE=12,DG=1,N_PATTERNS=2,N_SWR=14,HET=0.30,HET_WCOMP=2.3,\
#W_EC_DG=0.6,PP_RESIDUAL=0.9,DG_DELAY_JITTER=4.0,PATTERN_SOURCE=ec-lii,\
#PLACE_FIELD_SIGMA=0.15,EC_PATTERN_BASE_RATE=20,EC_PATTERN_PEAK_RATE=800,\
#EC_PATTERN_WEIGHT=1.5,DG_NEUROGENESIS=1,NEUROGENESIS_RATE=0.01,\
#NEUROGENESIS_MATURATION_EPOCHS=4,SEED=202 run.sh
#
# JOB H7 — Phase 10: DG perforant-path Hebbian plasticity (EC LII->GC),
#   testing whether a REAL experience-dependent learning rule (not just
#   age-indexed intrinsic properties, see JOB H6 above) lets young
#   neurogenesis cohorts respond differently to a held-back novel pattern
#   ("oddball" design, --novel-pattern-onset).
#
#   FIRST ATTEMPT (2026-09-02, jobs 45305940/45305945) gave ALL GCs this
#   plasticity, including a synthetic "cohort -1" for the entire original
#   population -- BOTH runs were killed at the 20h wall-time limit without
#   completing a single epoch of simulation. Root cause: cohort -1's
#   one-time GetConnections(target=GC) cache over 143,990 cells / ~28.65M
#   synapses cost 50,130-56,536s (14-16h) BY ITSELF -- confirmed NOT a
#   function of total kernel size (mpfc_assoc_hook's much cheaper cache,
#   ~28,800 synapses / 3.5-4h, runs at essentially the same cumulative
#   kernel size, ~5.2M conns/VP, per the slurmout conn-check log) but of
#   the target population's own size. Re-touching 28.65M synapses via
#   SetStatus every epoch would not have scaled either.
#
#   FIX: cohort -1 removed entirely (see build_dg_neurogenesis_hook /
#   run_dg_perforant_plasticity_hook docstrings). Plasticity now applies
#   ONLY to neurogenesis cohorts (hundreds of cells/epoch, cheap to cache
#   and update) -- the original mature GC population keeps fixed baseline
#   perforant-path weights, no learning. --dg-perforant-stdp now REQUIRES
#   --dg-neurogenesis (validated in argument parsing). The A/B is
#   therefore: no mechanism at all (control) vs. neurogenesis + Hebbian
#   cohort plasticity (test) -- both with the same oddball schedule.
#
#   Attempted locally first at 1% before the first MN5 attempt (matched
#   pair, same seed/schedule as below) and it thrashed: two parallel runs
#   pushed a 16GB laptop's swap to 11.6/12GB used, ~17% CPU duty cycle,
#   15h41m wall time with zero forward progress past NEST kernel init.
#   Killed both -- moved straight to MN5 12% instead of fighting local
#   memory contention, which is how the cohort -1 cost above went
#   undetected until it hit MN5. This account's QOS hard-caps wall time at
#   20h regardless (QOSMaxWallDurationPerJobLimit rejected a 36h request
#   outright) -- with cohort -1 gone, both arms below should easily fit.
#  sbatch --export=ALL,SCALE=12,DG=1,N_PATTERNS=3,N_SWR=14,HET=0.30,HET_WCOMP=2.3,\
#W_EC_DG=0.6,PP_RESIDUAL=0.9,DG_DELAY_JITTER=4.0,PATTERN_SOURCE=ec-lii,\
#PLACE_FIELD_SIGMA=0.15,EC_PATTERN_BASE_RATE=20,EC_PATTERN_PEAK_RATE=800,\
#EC_PATTERN_WEIGHT=1.5,NOVEL_PATTERN_ONSET=8,SEED=202 run.sh
#  sbatch --export=ALL,SCALE=12,DG=1,N_PATTERNS=3,N_SWR=14,HET=0.30,HET_WCOMP=2.3,\
#W_EC_DG=0.6,PP_RESIDUAL=0.9,DG_DELAY_JITTER=4.0,PATTERN_SOURCE=ec-lii,\
#PLACE_FIELD_SIGMA=0.15,EC_PATTERN_BASE_RATE=20,EC_PATTERN_PEAK_RATE=800,\
#EC_PATTERN_WEIGHT=1.5,NOVEL_PATTERN_ONSET=8,DG_PERFORANT_STDP=1,\
#DG_NEUROGENESIS=1,NEUROGENESIS_RATE=0.01,NEUROGENESIS_MATURATION_EPOCHS=4,\
#SEED=202 run.sh
#
# JOB H8 — Phase 11: clustered (topographic) EC LII->GC perforant-path fan-in.
#   RESULTS.md §17: at 1% scale, replacing the perforant path's uniform-random
#   K=50 sampling (every GC draws 50 sources from anywhere in EC LII, no
#   relation to place field) with LOCAL sampling (--dg-ec-cluster-sigma,
#   Gaussian-weighted toward each GC's own topographic neighbourhood) restores
#   a real single-cell identity signal that neither the original wiring nor a
#   naive K=1 fan-in cut produced (DG mean ring-distance from the active
#   pattern: 0.273 uniform K=50 -> 0.253 uniform K=1 -> 0.236 clustered K=50,
#   vs EC LII's own ~0.22 and a chance null of 0.25).
#
#   Same base config as JOB H7's CONTROL arm (no neurogenesis/plasticity, so
#   this isolates clustering as the only variable) -- A/B pair, same seed:
#   arm 1 = today's baseline wiring (control), arm 2 = clustered (test).
#   Wiring cost is small: ~7s/12,000 GCs locally -> ~85s expected for the
#   144,000 GCs at 12% scale (linear in N_gc), negligible next to the run's
#   own wall time -- nothing like the cohort-(-1) cache bug JOB H7 hit.
#  sbatch --export=ALL,SCALE=12,DG=1,N_PATTERNS=3,N_SWR=14,HET=0.30,HET_WCOMP=2.3,\
#W_EC_DG=0.6,PP_RESIDUAL=0.9,DG_DELAY_JITTER=4.0,PATTERN_SOURCE=ec-lii,\
#PLACE_FIELD_SIGMA=0.15,EC_PATTERN_BASE_RATE=20,EC_PATTERN_PEAK_RATE=800,\
#EC_PATTERN_WEIGHT=1.5,NOVEL_PATTERN_ONSET=8,SEED=202 run.sh
#  sbatch --export=ALL,SCALE=12,DG=1,N_PATTERNS=3,N_SWR=14,HET=0.30,HET_WCOMP=2.3,\
#W_EC_DG=0.6,PP_RESIDUAL=0.9,DG_DELAY_JITTER=4.0,PATTERN_SOURCE=ec-lii,\
#PLACE_FIELD_SIGMA=0.15,EC_PATTERN_BASE_RATE=20,EC_PATTERN_PEAK_RATE=800,\
#EC_PATTERN_WEIGHT=1.5,NOVEL_PATTERN_ONSET=8,DG_EC_CLUSTER_SIGMA=0.05,\
#SEED=202 run.sh
#
# JOB H10 — Phase 12: clustered (by CA3 group) Schaffer collaterals, at 12%.
#   RESULTS.md §20: a 1%-scale pilot of --schaffer-k 500 --schaffer-group-frac
#   0.7 was INCONCLUSIVE, not a clean win like JOB H8's DG fix -- CA1 PYR's
#   identity separation barely moved (-0.008 uniform -> -0.006 clustered,
#   still flagged no-discrimination in both arms), and active fractions swung
#   much more between arms than JOB H8 ever showed (DG GC 15.8%->24.1%, mPFC
#   40.0%->32.9%). One confound specifically identified: at 1% scale (10
#   groups, 264 CA3 SUP cells/group) the requested group_frac=0.7 (350 of 500
#   sources) is UNSATISFIABLE -- a group only has 264 cells to give -- so the
#   pilot only ever achieved 0.528 purity, confirmed via GetConnections on
#   sample cells. At 12% scale (35 groups, 905 CA3 SUP cells/group) the same
#   --schaffer-k 500 --schaffer-group-frac 0.7 IS satisfiable (350 < 905) --
#   this reruns the EXACT same test unclamped, to see whether the 1% result
#   was masked by that ceiling or is a real property of this approach.
#
#   Same base config as JOB H8's arms (DG wiring left at its DEFAULT, i.e.
#   NOT combined with JOB H8's --dg-ec-cluster-sigma fix -- one variable at a
#   time, same reasoning as JOB H8/H9's own isolation). Schaffer's own wiring
#   cost is small regardless (7.2s at 1% scale/4,600 CA1 cells -> ~86s
#   expected at 12%/55,195 cells, linear) -- the ~10-11h wall time is the
#   same full-run cost as every other JOB H8-scale run, not new overhead.
#  sbatch --export=ALL,SCALE=12,DG=1,N_PATTERNS=3,N_SWR=14,HET=0.30,HET_WCOMP=2.3,\
#W_EC_DG=0.6,PP_RESIDUAL=0.9,DG_DELAY_JITTER=4.0,PATTERN_SOURCE=ec-lii,\
#PLACE_FIELD_SIGMA=0.15,EC_PATTERN_BASE_RATE=20,EC_PATTERN_PEAK_RATE=800,\
#EC_PATTERN_WEIGHT=1.5,NOVEL_PATTERN_ONSET=8,SCHAFFER_K=500,SEED=202 run.sh
#  sbatch --export=ALL,SCALE=12,DG=1,N_PATTERNS=3,N_SWR=14,HET=0.30,HET_WCOMP=2.3,\
#W_EC_DG=0.6,PP_RESIDUAL=0.9,DG_DELAY_JITTER=4.0,PATTERN_SOURCE=ec-lii,\
#PLACE_FIELD_SIGMA=0.15,EC_PATTERN_BASE_RATE=20,EC_PATTERN_PEAK_RATE=800,\
#EC_PATTERN_WEIGHT=1.5,NOVEL_PATTERN_ONSET=8,SCHAFFER_K=500,\
#SCHAFFER_GROUP_FRAC=0.7,SEED=202 run.sh
#
# JOB H11 — Schaffer group clustering with CA3 groups that carry the pattern.
#   RESULTS.md §21: JOB H10 was null BY CONSTRUCTION -- under
#   PATTERN_SOURCE=ec-lii CA3 gets no group-specific drive, so its groups
#   carry no pattern (group z ~0). §23: the same pair at 1% with the default
#   PATTERN_SOURCE=ca3 (scaffold drives the pattern's CA3 groups, z ~ +1.0)
#   gives the first pattern identity in CA1: home-group z +0.648 grouped vs
#   +0.056 uniform (13/14 blocks, perm p=0.0002), rate-vector sep -0.022 ->
#   +0.051. But purity was clamped to 0.528 at 1%, 1 seed only. This is the
#   12% replication, unclamped (0.7 of K=500 from a 905-cell group). Needs
#   the STC-hook fix (commit 6c8a503) -- scp replay_scaled.py too.
#   PATTERN_SOURCE=ca3 set explicitly (it is also the default): the EC-LII
#   place-field flags and NOVEL_PATTERN_ONSET are ec-lii-only and dropped.
#   Wall time: JOB H10 took 10.9h, of which 5.3h was one-time connection
#   lookups (mPFC assoc 3.7h, STC 1.0h, EC LV lesion cache 0.6h). Those now
#   query source-side (get_conns_between, minutes). Expected ~5.5h at 50
#   threads (5.2h simulation + DG build); --time=08:00:00 for margin
#   (overrides the 20h header default). cpus-per-task=112 was tried and
#   rejected by gp_bsccs ("Requested node configuration is not available").
#  sbatch --time=08:00:00 --export=ALL,SCALE=12,DG=1,N_PATTERNS=3,N_SWR=14,HET=0.30,\
#HET_WCOMP=2.3,W_EC_DG=0.6,PP_RESIDUAL=0.9,DG_DELAY_JITTER=4.0,PATTERN_SOURCE=ca3,\
#SCHAFFER_K=500,SEED=202 run.sh
#  sbatch --time=08:00:00 --export=ALL,SCALE=12,DG=1,N_PATTERNS=3,N_SWR=14,HET=0.30,\
#HET_WCOMP=2.3,W_EC_DG=0.6,PP_RESIDUAL=0.9,DG_DELAY_JITTER=4.0,PATTERN_SOURCE=ca3,\
#SCHAFFER_K=500,SCHAFFER_GROUP_FRAC=0.7,SEED=202 run.sh
#
# JOB H12 — CA1->EC LII group clustering at 12%.
#   RESULTS.md SS26: JOB H11 replicated SS23 at 12% -- CA1 home-group z
#   +0.788 (14/14 blocks, p=0.0002) vs +0.089 uniform, and the group rate
#   modulation doubled vs 1% (+21.0% vs +10.8%). SS25's 1% pilot of
#   --ca1-ec-group-frac failed because CA1's modulation was only +10.8%; at
#   12% it is twice that, so the next hop gets its real test here. Control =
#   JOB H11's grouped arm (same seed, same build except CA1->EC wiring).
#   Output now records each population's gid_first, needed to assign EC LII
#   home groups when not every EC LII cell spikes.
#  sbatch --time=08:00:00 --export=ALL,SCALE=12,DG=1,N_PATTERNS=3,N_SWR=14,HET=0.30,\
#HET_WCOMP=2.3,W_EC_DG=0.6,PP_RESIDUAL=0.9,DG_DELAY_JITTER=4.0,PATTERN_SOURCE=ca3,\
#SCHAFFER_K=500,SCHAFFER_GROUP_FRAC=0.7,CA1_EC_GROUP_FRAC=0.7,SEED=202 run.sh
#
# JOB H13 — JOB H11 on a second seed (CLAUDE.md: single-seed = pilot).
#  sbatch --time=08:00:00 --export=ALL,SCALE=12,DG=1,N_PATTERNS=3,N_SWR=14,HET=0.30,\
#HET_WCOMP=2.3,W_EC_DG=0.6,PP_RESIDUAL=0.9,DG_DELAY_JITTER=4.0,PATTERN_SOURCE=ca3,\
#SCHAFFER_K=500,SEED=303 run.sh
#  sbatch --time=08:00:00 --export=ALL,SCALE=12,DG=1,N_PATTERNS=3,N_SWR=14,HET=0.30,\
#HET_WCOMP=2.3,W_EC_DG=0.6,PP_RESIDUAL=0.9,DG_DELAY_JITTER=4.0,PATTERN_SOURCE=ca3,\
#SCHAFFER_K=500,SCHAFFER_GROUP_FRAC=0.7,SEED=303 run.sh

SCALE=${SCALE:-25}
EC_LII=${EC_LII:-1}     # 1=add EC LII/III cortical target (default on)
EC_LII_K=${EC_LII_K:-50}
N_SWR=${N_SWR:-14}
EPOCH_MS=${EPOCH_MS:-1000}
PRP_THRESHOLD=${PRP_THRESHOLD:-14.0}
EC_LV=${EC_LV:-1}      # 1=enable Phase 3 EC LV, 0=disable
MPFC=${MPFC:-1}         # 1=enable mPFC module, 0=disable
NO_STC=${NO_STC:-0}     # 1=skip STC hook (useful for Phase 3-only runs)
HOMEOSTASIS=${HOMEOSTASIS:-0}  # 1=enable Phase 4 synaptic homeostasis
HOMEO_ALPHA=${HOMEO_ALPHA:-0.75}  # downscaling factor (default 0.75)
ALPHA_SWEEP=${ALPHA_SWEEP:-}      # comma-sep list, e.g. "0.50,0.75,0.90"; overrides HOMEO_ALPHA
# ---- Phase 6.2 dentate gyrus + pattern completion ---------------------------
HET=${HET:-0}                  # model-wide heterogeneity CV: weights, delays AND
                               # per-cell a/b/c/d/I_e. 0 = the historical
                               # homogeneous model (every cell a copy).
HET_WCOMP=${HET_WCOMP:-1.0}    # excitatory-onto-principal weight scale, to offset
                               # the gain heterogeneity costs. NOT rate-matched at
                               # 12% -- see the JOB H notes in the header.
W_EC_DG=${W_EC_DG:-}           # perforant path EC LII->GC weight (default 0.15)
PP_RESIDUAL=${PP_RESIDUAL:-}   # scale on the Poisson stand-in drive to DG
DG_DELAY_JITTER=${DG_DELAY_JITTER:-}   # extra ms jitter on the DG pathway
DG=${DG:-0}                    # 1=add the real DG (Phase 6.2), replaces Poisson proxy
# ---- Phase 7: encoding-direction pattern drive (see JOB H5 below) ---------
PATTERN_SOURCE=${PATTERN_SOURCE:-}   # ca3 (default) or ec-lii
PLACE_FIELD_SIGMA=${PLACE_FIELD_SIGMA:-}     # ec-lii only: ring-unit field width
EC_PATTERN_BASE_RATE=${EC_PATTERN_BASE_RATE:-}  # ec-lii only: Poisson floor Hz
EC_PATTERN_PEAK_RATE=${EC_PATTERN_PEAK_RATE:-}  # ec-lii only: Poisson peak-gain Hz
EC_PATTERN_WEIGHT=${EC_PATTERN_WEIGHT:-}     # ec-lii only: place-field synapse weight
DG_EC_CLUSTER_SIGMA=${DG_EC_CLUSTER_SIGMA:-}   # Phase 11: >0 = topographically clustered
                                               # EC LII->GC fan-in (see JOB H8, RESULTS.md SS17)
# ---- Phase 8: DG neurogenesis ---------------------------------------------
DG_NEUROGENESIS=${DG_NEUROGENESIS:-0}   # 1=new hyperexcitable GC cohort each epoch
NEUROGENESIS_RATE=${NEUROGENESIS_RATE:-}   # fraction of original N_gc born/epoch
NEUROGENESIS_MATURATION_EPOCHS=${NEUROGENESIS_MATURATION_EPOCHS:-}
NEUROGENESIS_YOUNG_IE=${NEUROGENESIS_YOUNG_IE:-}
NEUROGENESIS_YOUNG_INHIB_SCALE=${NEUROGENESIS_YOUNG_INHIB_SCALE:-}
NEUROGENESIS_YOUNG_PP_SCALE=${NEUROGENESIS_YOUNG_PP_SCALE:-}
# ---- Phase 10: DG perforant-path plasticity --------------------------------
DG_PERFORANT_STDP=${DG_PERFORANT_STDP:-0}   # 1=Hebbian EC LII->GC learning
DG_ASSOC_A=${DG_ASSOC_A:-}           # mature-baseline potentiation per event
DG_ASSOC_A_HETERO=${DG_ASSOC_A_HETERO:-}    # heterosynaptic depression per event
DG_ASSOC_W_MAX=${DG_ASSOC_W_MAX:-}
DG_ASSOC_W_MIN=${DG_ASSOC_W_MIN:-}
DG_SCALE=${DG_SCALE:-$SCALE}   # DG scale %; defaults to SCALE
PATTERN_COMPLETION=${PATTERN_COMPLETION:-0}  # 1=run the CA3 completion probe INSTEAD
PC_CUE_FRACS=${PC_CUE_FRACS:-0.1,0.2,0.3,0.5,0.7,1.0}
PC_CUE_WEIGHT=${PC_CUE_WEIGHT:-2.5}
# ---- multi-pattern / temporal-code / Test-3 knobs -------------------------
N_PATTERNS=${N_PATTERNS:-1}        # >1 splits CA3 groups into interleaved assemblies
TRAIN_PATTERN=${TRAIN_PATTERN:-}   # replay ONLY this pattern index (A-only vs B-only)
NOVEL_PATTERN_ONSET=${NOVEL_PATTERN_ONSET:-}   # Phase 9 oddball: epoch index where a held-back pattern first appears
SEED=${SEED:-}                     # sets BOTH the NEST kernel and numpy seeds
SCHAFFER_K=${SCHAFFER_K:-}         # CA3->CA1 in-degree override (weights auto-scaled)
SCHAFFER_GROUP_FRAC=${SCHAFFER_GROUP_FRAC:-}   # Phase 12: bias reduced Schaffer
                                               # in-degree toward each CA1 cell's
                                               # home CA3 group (requires SCHAFFER_K)
CA1_EC_GROUP_FRAC=${CA1_EC_GROUP_FRAC:-}       # RESULTS SS25: group-cluster CA1->EC LII by CA1 home group (needs SCHAFFER_GROUP_FRAC)
SCHAFFER_STDP=${SCHAFFER_STDP:-0}  # 1 = delay-aware STDP on CA3->CA1
DELAY_JITTER=${DELAY_JITTER:-0}    # per-synapse axonal delay jitter (ms)
NO_MPFC_ASSOC=${NO_MPFC_ASSOC:-0}  # 1 = no cortical plasticity (Test-3 control)
CORTICAL_RECALL=${CORTICAL_RECALL:-0}   # 1 = lesion + cue + measure (Test 3)
CR_CUE_FRAC=${CR_CUE_FRAC:-0.4}
CR_PRIME_RATE=${CR_PRIME_RATE:-120}     # keep subthreshold: baseline must stay ~0
CR_PRIME_WEIGHT=${CR_PRIME_WEIGHT:-1.0}
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
echo "[Slurm] ec_lv=${EC_LV}  mpfc=${MPFC}  no_stc=${NO_STC}  homeostasis=${HOMEOSTASIS}  homeo_alpha=${HOMEO_ALPHA}  alpha_sweep=${ALPHA_SWEEP:-<none>}"
echo "[Slurm] dg=${DG}  dg_scale=${DG_SCALE}  pattern_completion=${PATTERN_COMPLETION}"
echo "[Slurm] het=${HET}  het_wcomp=${HET_WCOMP}  w_ec_dg=${W_EC_DG:-<default>}  pp_residual=${PP_RESIDUAL:-<default>}  dg_delay_jitter=${DG_DELAY_JITTER:-<default>}"
echo "[Slurm] pattern_source=${PATTERN_SOURCE:-ca3}  place_field_sigma=${PLACE_FIELD_SIGMA:-<default>}  ec_pattern_base_rate=${EC_PATTERN_BASE_RATE:-<default>}  ec_pattern_peak_rate=${EC_PATTERN_PEAK_RATE:-<default>}"
echo "[Slurm] dg_neurogenesis=${DG_NEUROGENESIS}  neurogenesis_rate=${NEUROGENESIS_RATE:-<default>}  maturation_epochs=${NEUROGENESIS_MATURATION_EPOCHS:-<default>}  young_ie=${NEUROGENESIS_YOUNG_IE:-<default>}  young_inhib_scale=${NEUROGENESIS_YOUNG_INHIB_SCALE:-<default>}  young_pp_scale=${NEUROGENESIS_YOUNG_PP_SCALE:-<default>}"
echo "[Slurm] dg_perforant_stdp=${DG_PERFORANT_STDP}  dg_assoc_a=${DG_ASSOC_A:-<default>}  dg_assoc_a_hetero=${DG_ASSOC_A_HETERO:-<default>}  dg_assoc_w_max=${DG_ASSOC_W_MAX:-<default>}  dg_assoc_w_min=${DG_ASSOC_W_MIN:-<default>}"
echo "[Slurm] novel_pattern_onset=${NOVEL_PATTERN_ONSET:-<none>}"
echo "[Slurm] dg_ec_cluster_sigma=${DG_EC_CLUSTER_SIGMA:-<none, uniform random>}"
echo "[Slurm] schaffer_k=${SCHAFFER_K:-<default, 100% dense>}  schaffer_group_frac=${SCHAFFER_GROUP_FRAC:-<none, uniform random>}"
echo "[Slurm] ca1_ec_group_frac=${CA1_EC_GROUP_FRAC:-<none, uniform random>}"

python3 - <<'PY'
import nest
ks = nest.GetKernelStatus()
mpi = ks.get("mpi_num_processes", ks.get("num_processes", ks.get("total_num_processes", 1)))
thr = ks.get("local_num_threads", ks.get("num_threads", ks.get("threads", 1)))
print(f"nest {nest.__version__}  mpi_procs={mpi}  local_threads={thr}")
PY

mkdir -p "$OUTDIR"

# ---- Pattern-completion probe: separate run mode, short-circuits here --------
# The probe builds an isolated CA3 twice (intact + sup_local-ablated), ignores
# STC / EC / homeostasis, and exits after writing its own HDF5. Runs on the
# same node config; it is lighter than the full consolidation run.
if [ "$PATTERN_COMPLETION" = "1" ]; then
  PC_OUT="${OUTDIR}/pattern_completion_${SCALE}pct.h5"
  echo "[Slurm] PATTERN COMPLETION mode → $PC_OUT"
  srun --cpu-bind=cores \
    python3 -u "replay_scaled.py" \
      --scale             "$SCALE" \
      --threads           "$SLURM_CPUS_PER_TASK" \
      --pattern-completion \
      --pc-cue-fracs      "$PC_CUE_FRACS" \
      --pc-cue-weight     "$PC_CUE_WEIGHT" \
      --out-hdf5          "$PC_OUT" \
      --no-figures
  echo "[Slurm] pattern-completion done."
  exit 0
fi

# Tag output filename with active phases
PHASE_TAG=""
[ "$DG" = "1" ] && PHASE_TAG="${PHASE_TAG}_dg"
# tag heterogeneity so H1/H2 cannot overwrite the homogeneous JOB F output
[ "$HET" != "0" ] && PHASE_TAG="${PHASE_TAG}_het${HET}"
[ -n "$W_EC_DG" ] && PHASE_TAG="${PHASE_TAG}_pp${W_EC_DG}"
[ "$EC_LV" = "1" ]  && PHASE_TAG="${PHASE_TAG}_lv"
[ "$MPFC"  = "1" ]  && PHASE_TAG="${PHASE_TAG}_mpfc"
[ "${PRP_THRESHOLD%.*}" -gt 100 ] 2>/dev/null && PHASE_TAG="${PHASE_TAG}_ph5"
if [ "$HOMEOSTASIS" = "1" ]; then
  if [ -n "$ALPHA_SWEEP" ]; then
    # Sweep mode: tag with #alphas and a hash of the list (e.g. _ph4sw3_050075090)
    SWEEP_HASH=$(echo "$ALPHA_SWEEP" | tr -d '.,' | tr ' ' '_')
    SWEEP_N=$(echo "$ALPHA_SWEEP" | tr ',' '\n' | grep -c .)
    PHASE_TAG="${PHASE_TAG}_ph4sw${SWEEP_N}_${SWEEP_HASH}"
  else
    PHASE_TAG="${PHASE_TAG}_ph4_a${HOMEO_ALPHA//./}"
  fi
fi

[ "$SCHAFFER_STDP" = "1" ]  && PHASE_TAG="${PHASE_TAG}_stdp"
[ "$CORTICAL_RECALL" = "1" ] && PHASE_TAG="${PHASE_TAG}_recall"
[ "$NO_MPFC_ASSOC" = "1" ]   && PHASE_TAG="${PHASE_TAG}_noplast"
[ -n "$TRAIN_PATTERN" ]      && PHASE_TAG="${PHASE_TAG}_p${TRAIN_PATTERN}"
# tag pattern-source/neurogenesis so a JOB H6-style A/B pair (same SCALE,
# HET, W_EC_DG, SEED -- differing only in these flags) cannot collide on the
# same output file when run in parallel.
[ "$PATTERN_SOURCE" = "ec-lii" ] && PHASE_TAG="${PHASE_TAG}_eclii"
[ "$DG_NEUROGENESIS" = "1" ]     && PHASE_TAG="${PHASE_TAG}_neurogen"
[ "$DG_PERFORANT_STDP" = "1" ]   && PHASE_TAG="${PHASE_TAG}_pstdp"
[ -n "$NOVEL_PATTERN_ONSET" ]    && PHASE_TAG="${PHASE_TAG}_novel${NOVEL_PATTERN_ONSET}"
[ -n "$DG_EC_CLUSTER_SIGMA" ]    && PHASE_TAG="${PHASE_TAG}_clu${DG_EC_CLUSTER_SIGMA}"
[ -n "$SCHAFFER_K" ]             && PHASE_TAG="${PHASE_TAG}_schk${SCHAFFER_K}"
[ -n "$SCHAFFER_GROUP_FRAC" ]    && PHASE_TAG="${PHASE_TAG}_schgrp${SCHAFFER_GROUP_FRAC}"
[ -n "$CA1_EC_GROUP_FRAC" ]      && PHASE_TAG="${PHASE_TAG}_ca1ecgrp${CA1_EC_GROUP_FRAC}"
[ -n "$SEED" ]               && PHASE_TAG="${PHASE_TAG}_s${SEED}"
OUTFILE="${OUTDIR}/replay_${SCALE}pct_stc${PHASE_TAG}.h5"
echo "[Slurm] output → $OUTFILE"

# STC requires EC LII (the model errors otherwise) — force it on if STC is active.
[ "$NO_STC" != "1" ] && EC_LII=1

# Build optional flag list
OPTIONAL_FLAGS=""
[ "$EC_LII" = "1" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --ec-lii --ec-lii-k $EC_LII_K"
[ "$N_PATTERNS" != "1" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --n-patterns $N_PATTERNS"
[ -n "$TRAIN_PATTERN" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --train-pattern $TRAIN_PATTERN"
[ -n "$NOVEL_PATTERN_ONSET" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --novel-pattern-onset $NOVEL_PATTERN_ONSET"
[ -n "$SEED" ]          && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --seed $SEED"
[ -n "$SCHAFFER_K" ]    && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --schaffer-k $SCHAFFER_K"
[ -n "$SCHAFFER_GROUP_FRAC" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --schaffer-group-frac $SCHAFFER_GROUP_FRAC"
[ -n "$CA1_EC_GROUP_FRAC" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --ca1-ec-group-frac $CA1_EC_GROUP_FRAC"
[ "$SCHAFFER_STDP" = "1" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --schaffer-stdp"
[ "$DELAY_JITTER" != "0" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --delay-jitter $DELAY_JITTER"
[ "$NO_MPFC_ASSOC" = "1" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --no-mpfc-assoc"
if [ "$CORTICAL_RECALL" = "1" ]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --cortical-recall --cr-cue-frac $CR_CUE_FRAC"
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --cr-prime-rate $CR_PRIME_RATE --cr-prime-weight $CR_PRIME_WEIGHT"
fi
[ "$DG"     = "1" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --dg --dg-scale $DG_SCALE"
[ "$HET" != "0" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --het $HET --het-wcomp $HET_WCOMP"
[ -n "$W_EC_DG" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --w-ec-dg $W_EC_DG"
[ -n "$PP_RESIDUAL" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --pp-residual $PP_RESIDUAL"
[ -n "$DG_DELAY_JITTER" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --dg-delay-jitter $DG_DELAY_JITTER"
[ -n "$PATTERN_SOURCE" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --pattern-source $PATTERN_SOURCE"
[ -n "$PLACE_FIELD_SIGMA" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --place-field-sigma $PLACE_FIELD_SIGMA"
[ -n "$EC_PATTERN_BASE_RATE" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --ec-pattern-base-rate $EC_PATTERN_BASE_RATE"
[ -n "$EC_PATTERN_PEAK_RATE" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --ec-pattern-peak-rate $EC_PATTERN_PEAK_RATE"
[ -n "$EC_PATTERN_WEIGHT" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --ec-pattern-weight $EC_PATTERN_WEIGHT"
[ -n "$DG_EC_CLUSTER_SIGMA" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --dg-ec-cluster-sigma $DG_EC_CLUSTER_SIGMA"
[ "$DG_NEUROGENESIS" = "1" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --dg-neurogenesis"
[ -n "$NEUROGENESIS_RATE" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --neurogenesis-rate $NEUROGENESIS_RATE"
[ -n "$NEUROGENESIS_MATURATION_EPOCHS" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --neurogenesis-maturation-epochs $NEUROGENESIS_MATURATION_EPOCHS"
[ -n "$NEUROGENESIS_YOUNG_IE" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --neurogenesis-young-ie $NEUROGENESIS_YOUNG_IE"
[ -n "$NEUROGENESIS_YOUNG_INHIB_SCALE" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --neurogenesis-young-inhib-scale $NEUROGENESIS_YOUNG_INHIB_SCALE"
[ -n "$NEUROGENESIS_YOUNG_PP_SCALE" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --neurogenesis-young-pp-scale $NEUROGENESIS_YOUNG_PP_SCALE"
[ "$DG_PERFORANT_STDP" = "1" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --dg-perforant-stdp"
[ -n "$DG_ASSOC_A" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --dg-assoc-a $DG_ASSOC_A"
[ -n "$DG_ASSOC_A_HETERO" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --dg-assoc-a-hetero $DG_ASSOC_A_HETERO"
[ -n "$DG_ASSOC_W_MAX" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --dg-assoc-w-max $DG_ASSOC_W_MAX"
[ -n "$DG_ASSOC_W_MIN" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --dg-assoc-w-min $DG_ASSOC_W_MIN"
[ "$NO_STC" != "1" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --stc --n-swr $N_SWR --epoch-ms $EPOCH_MS --prp-threshold $PRP_THRESHOLD"
[ "$EC_LV"  = "1" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --ec-lv"
[ "$MPFC"   = "1" ] && OPTIONAL_FLAGS="$OPTIONAL_FLAGS --mpfc"
if [ "$HOMEOSTASIS" = "1" ]; then
  if [ -n "$ALPHA_SWEEP" ]; then
    OPTIONAL_FLAGS="$OPTIONAL_FLAGS --homeostasis --alpha-sweep $ALPHA_SWEEP"
  else
    OPTIONAL_FLAGS="$OPTIONAL_FLAGS --homeostasis --homeo-alpha $HOMEO_ALPHA"
  fi
fi

srun --cpu-bind=cores \
  python3 -u "replay_scaled.py" \
    --scale       "$SCALE" \
    --threads     "$SLURM_CPUS_PER_TASK" \
    --out-hdf5    "$OUTFILE" \
    $OPTIONAL_FLAGS \
    --no-figures
