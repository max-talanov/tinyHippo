#!/usr/bin/env python3
"""Extract a projection's ACTUAL fixed connectivity from replay_scaled.py's
network builder, without paying for a full simulation.

Motivation
----------
replay_scaled.py's connectivity (EC LII->GC, Schaffer collaterals, etc.) is
generated once at build time by NEST's own kernel RNG, seeded from --seed,
and never touched again during the run. That means the exact source/target
pairs for ANY projection are, in principle, fully reproducible from the CLI
flags alone -- no need to re-run the (possibly hours-long) simulation to ask
"which cells does this granule cell actually receive input from?".

This script drives replay_scaled.py's own network-builder functions (same
code, not a reimplementation) up through the module you ask for, then exits
before the expensive epoch-simulation loop starts. It was built to test
whether DG granule cells with more input from currently-pattern-tuned EC LII
cells are more likely to fire (see RESULTS.md SS15) -- reuse it for any
other "does the model's actual wiring predict X" question.

How it works
------------
1. Splits replay_scaled.py's source at `if __name__ == "__main__":` and
   execs the head in a private namespace -- this defines every function/
   class (build_ec_lii, build_dg_module, ...) without running the CLI block.
2. Wraps build_ec_lii / build_dg_module (or add more names to CAPTURE_FROM
   below for other modules) in that namespace so their return values are
   captured as a side effect of the real call -- main()'s own CLI logic
   calls these by bare name, which resolves via the namespace dict at call
   time, so rebinding them there is enough; no source patching needed.
3. Patches nest.Simulate to raise once execution reaches the epoch loop.
   NEST's module object blocks plain `nest.Simulate = ...` (custom
   __setattr__) -- use `nest.__dict__['Simulate'] = ...` to bypass it.
4. Execs the tail (the `if __name__ == "__main__":` block, with __name__
   forced to "__main__" in the namespace) inside a try/except for the abort
   exception. Every population up to the point you stopped at now exists in
   the live NEST kernel, exactly as it would in a real run with these flags.
5. Reads off nest.GetConnections() specifying whichever side of the
   projection is SMALLER (its cost tracks that side's own connection count,
   not the other side's or the kernel's total -- confirmed twice now: the
   DG cohort-1 cache cost 14-16h scanning a 144,000-cell TARGET, and JOB
   H9's first two 12%-scale attempts stalled past 2h the same way before
   this was fixed to query by the ~50x-smaller SOURCE side instead), then
   post-filters by the other side in Python. Never GetConnections(source=,
   target=) with BOTH specified -- see build_homeostasis_hook's docstring
   in replay_scaled.py for why that variant is catastrophically slow (it
   walks the FULL connectome, not just one side's slots).

Usage
-----
    python reconstruct_connectivity.py --target ec_lii_gc \\
        --replay-args --scale 1 --dg --ec-lii --ec-lv --mpfc \\
        --n-patterns 3 --n-swr 14 --stc --het 0.30 --het-wcomp 2.3 \\
        --w-ec-dg 0.6 --pp-residual 0.9 --dg-delay-jitter 4.0 \\
        --pattern-source ec-lii --place-field-sigma 0.15 \\
        --ec-pattern-base-rate 20 --ec-pattern-peak-rate 800 \\
        --ec-pattern-weight 1.5 --novel-pattern-onset 8 --seed 202 \\
        --out-hdf5 /tmp/throwaway.h5 --no-figures

Everything after --replay-args is passed straight through as if it were
`python replay_scaled.py ...` -- it MUST match the config of whatever run
you're comparing against (same --scale, --seed, and every flag that affects
population sizes or connectivity, since NEST's RNG draws are order- and
seed-dependent). Output: an .npz with the projection's source/target GID
arrays plus each population's full GID list, at --out (default
/tmp/reconstructed_connectivity.npz).

Only "ec_lii_gc" (EC LII -> DG granule cells) is implemented today. To add
another projection: add its module name(s) to CAPTURE_FROM, and a branch in
extract() that knows which two populations to intersect.
"""
import argparse
import sys
import time

import numpy as np

REPLAY_SCALED_PATH = "replay_scaled.py"

# function names whose return value main()'s CLI block produces that we need
# a handle on afterwards -- add to this (and to extract() below) to support
# reconstructing other projections.
CAPTURE_FROM = ["build_ec_lii", "build_dg_module"]


class _StopBuild(Exception):
    """Raised by the patched nest.Simulate to abort before the epoch loop."""


def build_network(replay_args):
    """Run replay_scaled.py's own CLI logic up to the first nest.Simulate()
    call, using the exact flags in replay_args, and return the captured
    module objects (e.g. ec_module, dg_module) plus the namespace they live
    in (for anything else you might want to reach into, e.g. net["..."])."""
    import nest

    nest.__dict__["Simulate"] = lambda *a, **kw: (_ for _ in ()).throw(_StopBuild())

    src = open(REPLAY_SCALED_PATH).read()
    marker = 'if __name__ == "__main__":'
    idx = src.index(marker)
    head, tail = src[:idx], src[idx:]

    ns = {"__name__": "replay_scaled_reconstruct", "__file__": REPLAY_SCALED_PATH}
    exec(compile(head, REPLAY_SCALED_PATH, "exec"), ns)

    captured = {}
    for name in CAPTURE_FROM:
        orig = ns[name]

        def make_wrapper(orig=orig, name=name):
            def wrapper(*a, **kw):
                r = orig(*a, **kw)
                captured[name] = r
                return r

            return wrapper

        ns[name] = make_wrapper()

    sys.argv = ["replay_scaled.py"] + list(replay_args)
    ns["__name__"] = "__main__"

    t0 = time.perf_counter()
    try:
        exec(compile(tail, REPLAY_SCALED_PATH, "exec"), ns)
        raise RuntimeError(
            "build finished without ever reaching nest.Simulate() -- "
            "did replay_args disable the module(s) you wanted?"
        )
    except _StopBuild:
        print(f">>> build stopped at first Simulate() call, {time.perf_counter()-t0:.1f}s elapsed")

    return captured, ns


def extract_ec_lii_gc(captured):
    import nest

    ec_module = captured["build_ec_lii"]
    dg_module = captured["build_dg_module"]
    ec_pop, gc_pop = ec_module.population, dg_module.GC
    print(f"EC LII population size: {len(ec_pop)}")
    print(f"DG GC population size: {len(gc_pop)}")

    # Query by SOURCE (EC LII), not target (GC): GetConnections(target=X)'s
    # cost tracks X's own INCOMING connection count (RESULTS.md SS16/JOB H7 --
    # the same pattern that made the DG cohort-1 cache cost 14-16h at a
    # 144,000-cell target). GC receives ~30M incoming synapses total across
    # every DG projection (perforant path + basket/mossy-cell feedback);
    # walking that to find the ~600K EC LII ones is what stalled JOB H9's
    # first two 12%-scale attempts past a 2h wall. EC LII's OUTGOING slots
    # are only the perforant path itself (600,000 at K=50) whenever --ec-lv
    # is not also active (its only other possible target) -- querying by
    # source instead walks a set ~50x smaller for the same information.
    t1 = time.perf_counter()
    out_conns = nest.GetConnections(source=ec_pop)  # source-only, NOT target=GC
    src = np.array(nest.GetStatus(out_conns, "source"), dtype=np.int64)
    tgt = np.array(nest.GetStatus(out_conns, "target"), dtype=np.int64)
    print(f"GetConnections(source=EC LII): {len(out_conns):,} synapses in {time.perf_counter()-t1:.1f}s")

    gc_id_set = set(gc_pop.tolist())
    mask = np.array([t in gc_id_set for t in tgt])
    print(f"EC LII -> GC synapses: {mask.sum():,} (expect {len(gc_pop)*50:,} at K=50)")

    return dict(
        src=src[mask], tgt=tgt[mask],
        ec_ids=np.array(sorted(ec_pop.tolist())), gc_ids=np.array(gc_pop.tolist()),
    )


EXTRACTORS = {
    "ec_lii_gc": extract_ec_lii_gc,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", choices=sorted(EXTRACTORS), required=True)
    p.add_argument("--out", default="/tmp/reconstructed_connectivity.npz")
    p.add_argument("--replay-args", nargs=argparse.REMAINDER, required=True,
                    help="everything after this flag is passed to replay_scaled.py verbatim")
    args = p.parse_args()

    captured, _ns = build_network(args.replay_args)
    result = EXTRACTORS[args.target](captured)
    np.savez(args.out, **result)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
