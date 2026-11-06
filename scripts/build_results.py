"""Collect a small, shareable copy of the results into results/.

What goes in (everything else stays on scratch: raw activations, per-turn transcripts, per-GPU logs):
  results/summaries/          every *.json in artifacts/ (fits, screens, label checks, grid and replication summaries)
  results/sessions/<run>.tar.gz   meta.json + sess_*_trials.json for every run directory in outputs/
  results/trials.csv.gz       one row per trial across all runs, for quick analysis
  results/logs/               campaign status files and job stdout (*.status, *.out, *.txt)

Usage: python scripts/build_results.py
"""
import csv
import gzip
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"


def copy_summaries():
    dst = OUT / "summaries"
    for f in sorted((ROOT / "artifacts").rglob("*.json")):
        if f.is_symlink():
            continue
        rel = f.relative_to(ROOT / "artifacts")
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst / rel)


def session_files(run):
    for f in sorted(run.rglob("*")):
        if f.name == "meta.json" or f.name.endswith("_trials.json"):
            yield f


def pack_sessions():
    dst = OUT / "sessions"
    dst.mkdir(parents=True, exist_ok=True)
    for run in sorted(p for p in (ROOT / "outputs").iterdir() if p.is_dir()):
        with tarfile.open(dst / f"{run.name}.tar.gz", "w:gz") as tar:
            for f in session_files(run):
                tar.add(f, arcname=str(f.relative_to(ROOT / "outputs")))


def trials_table():
    cols = ["run", "shard", "session", "draw", "alignment", "theta_deg", "severity_deg", "magnitude",
            "block", "trial", "acquired", "moves_used", "mean_angular_error_rad", "n_moves"]
    with gzip.open(OUT / "trials.csv.gz", "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for run in sorted(p for p in (ROOT / "outputs").iterdir() if p.is_dir()):
            for meta_f in sorted(run.rglob("meta.json")):
                shard = meta_f.parent
                try:
                    metas = json.loads(meta_f.read_text())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(metas, dict):
                    metas = [metas]
                by_sess = {m.get("session"): m for m in metas if isinstance(m, dict)}
                for tf in sorted(shard.glob("sess_*_trials.json")):
                    sid = int(tf.name.split("_")[1])
                    m = by_sess.get(sid, {})
                    pert = m.get("perturbation") or {}
                    try:
                        trials = json.loads(tf.read_text())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    for t in trials:
                        errs = t.get("angular_errors") or []
                        w.writerow([run.name, shard.name, sid, m.get("draw"), pert.get("alignment"),
                                    pert.get("theta_deg"), pert.get("dev_deg"), pert.get("mag"),
                                    t.get("block"), t.get("trial"), t.get("acquired"), t.get("moves_used"),
                                    sum(errs) / len(errs) if errs else None, len(errs)])


def copy_logs():
    dst = OUT / "logs"
    dst.mkdir(parents=True, exist_ok=True)
    for pat in ("*.status", "*.out", "*.txt"):
        for f in sorted((ROOT / "logs").glob(pat)):
            shutil.copy2(f, dst / f.name)


if __name__ == "__main__":
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    copy_summaries()
    pack_sessions()
    trials_table()
    copy_logs()
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"wrote {OUT} ({total / 1e6:.1f} MB)")
