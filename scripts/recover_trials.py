#!/usr/bin/env python
"""Rebuild a missing ``sess_XXX_trials.json`` from the per-turn transcript beside it.

WHY THIS EXISTS. Until 2026-08-15 run.py wrote a session's trials only when its whole BATCH
finished, while the per-turn ``.jsonl`` was flushed every turn. A batch killed by the 3 h wall
therefore threw away complete sessions whose every trial was already on disk in the transcript:
fourteen of them, at 74 to 96 trials each. Checkpointing now prevents the loss going forward; this
recovers what was already lost.

WHAT MAKES IT SAFE. Every field of a ``done_trials`` entry is derivable from the turn records:

    block, alignment  recorded on each turn
    acquired          the trial's last turn
    moves_used        the sum of the per-turn ``n_moves`` (each is what run_lockstep applied)
    angular_errors    the per-turn lists concatenated, with the None that stood for NaN dropped

so this is a re-derivation, not an estimate. A trial is emitted ONLY when it is provably complete,
meaning it was acquired or it exhausted the move budget; a trial still in flight when the job died
is dropped rather than recorded as a short one, which would look like a fast failure.

The reconstruction is verified against every session that has BOTH files before anything is
written: if a single one disagrees, nothing is recovered.

    python scripts/recover_trials.py --dir outputs/v6g_d9 --budget 48
    python scripts/recover_trials.py --all --budget 48 --write
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rebuild(jsonl: Path, budget: int) -> list[dict]:
    rows = []
    for line in jsonl.open():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:                                            # noqa: BLE001
            continue                                                 # truncated final write
    out, cur, idx = [], [], None
    def flush():
        if not cur:
            return
        acq = bool(cur[-1].get("acquired"))
        moves = sum(int(r.get("n_moves") or 0) for r in cur)
        if not (acq or moves >= budget):
            return                                                   # still in flight; drop it
        errs = [e for r in cur for e in (r.get("angular_errors") or []) if e is not None]
        out.append({"trial": idx, "block": cur[0].get("block"),
                    "alignment": cur[0].get("alignment"), "acquired": acq,
                    "moves_used": moves, "angular_errors": errs})
    for r in rows:
        if r.get("trial") != idx:
            flush(); cur, idx = [], r.get("trial")
        cur.append(r)
    flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", default=None)
    ap.add_argument("--all", action="store_true", help="every outputs/v6* sweep")
    ap.add_argument("--budget", type=int, default=48)
    ap.add_argument("--write", action="store_true", help="actually write; default is a dry run")
    a = ap.parse_args()

    dirs = a.dir or []
    if a.all or not dirs:
        dirs = sorted({str(Path(p).parent.parent) for p in
                       glob.glob(str(ROOT / "outputs/v6*/*/sess_*.jsonl"))})

    # ---- verify against every session that has BOTH files, before touching anything
    checked = mismatch = 0
    for d in dirs:
        for j in sorted(glob.glob(f"{d}/*/sess_*.jsonl")):
            t = Path(j.replace(".jsonl", "_trials.json"))
            if not t.exists():
                continue
            got, want = rebuild(Path(j), a.budget), json.loads(t.read_text())
            checked += 1
            if got != want:
                mismatch += 1
                if mismatch <= 3:
                    print(f"  MISMATCH {j}: rebuilt {len(got)} trials vs {len(want)} recorded")
                    for x, y in zip(got, want):
                        if x != y:
                            print(f"     first differing trial:\n       rebuilt  {x}\n"
                                  f"       recorded {y}")
                            break
    print(f"[recover] verification: {checked} sessions with both files, {mismatch} mismatched")
    if mismatch:
        raise SystemExit("[recover] reconstruction does not reproduce known-good files; "
                         "refusing to write anything")

    # ---- recover the orphans, ONE PER SESSION INDEX
    # A session that never completed was retried by later chains, so the same index has a transcript
    # in several shard dirs. analyze_grid counts one row per trials.json it finds anywhere in the
    # sweep, so writing all of them would enter the same session two to five times and silently
    # weight it that many times in every aggregate. Keep only the longest attempt.
    n = tot = dropped = 0
    for d in dirs:
        best = {}
        for j in sorted(glob.glob(f"{d}/*/sess_*.jsonl")):
            t = Path(j.replace(".jsonl", "_trials.json"))
            if t.exists():
                best[Path(j).stem] = None            # already recorded; never touch it
                continue
            i = Path(j).stem
            if best.get(i, "missing") is None:
                continue
            tr = rebuild(Path(j), a.budget)
            if not tr:
                continue
            if i not in best or len(tr) > len(best[i][1]):
                if i in best:
                    dropped += 1
                best[i] = (t, tr)
            else:
                dropped += 1
        for i, v in sorted(best.items()):
            if v is None:
                continue
            t, tr = v
            pert = sum(1 for x in tr if x["block"] == "perturbation")
            n += 1; tot += len(tr)
            print(f"  {'WRITE' if a.write else 'would write'} {t.relative_to(ROOT)}  "
                  f"{len(tr)} trials ({pert} perturbation, "
                  f"{'USABLE' if pert >= 40 else 'too short to analyse'})")
            if a.write:
                t.write_text(json.dumps(tr, indent=2, default=float))
    if dropped:
        print(f"[recover] {dropped} shorter duplicate attempts of the same session index skipped")
    print(f"[recover] {n} orphaned sessions, {tot} trials "
          f"{'recovered' if a.write else 'recoverable (dry run; pass --write)'}")


if __name__ == "__main__":
    main()
