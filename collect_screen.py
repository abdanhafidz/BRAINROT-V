"""Download a set of runs, measure them, and compare arms within seed.

Seed is a blocking factor in this project: the 10 October screen found seed 1
roughly four times harder than seeds 0 and 2 for *every* arm. Comparing unpaired
means across arms throws that structure away and needs far more runs for the
same power, so this reports per-seed values and paired win counts rather than a
single mean.

    python collect_screen.py --prefix llm --suffix 30s \
        --arms m0-baseline m1-snapkv-pool m2-tova-last m3-scissor-decay m4-adakv-entropy \
        --seeds 0 1 2 --baseline m0-baseline
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
MODAL_PY = REPO / ".modal-client" / "Scripts" / "python.exe"
VOLUME = "brainrot-v-outputs"


def parse_args():
    p = argparse.ArgumentParser(description="Collect, measure and compare runs")
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--prefix", default="llm", help="run-id prefix")
    p.add_argument("--suffix", default="30s", help="run-id suffix")
    p.add_argument("--baseline", default=None, help="arm to compare the others against")
    p.add_argument("--out-dir", default=None)
    # Window the damage metrics are averaged over. Second 0 has no predecessor
    # for flicker, and the tail of a clip is often past the point of interest.
    p.add_argument("--window", nargs=2, type=int, default=[1, 28],
                   metavar=("START", "END"))
    return p.parse_args()


def run_id_for(prefix, arm, seed, suffix):
    return f"{prefix}-{arm}-s{seed}-{suffix}"


def fetch(run_id, ext, dest):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [str(MODAL_PY), "-m", "modal", "volume", "get", VOLUME,
         f"{run_id}/brainrot_v.{ext}", str(dest), "--force"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, cwd=str(REPO))
    return result.returncode == 0


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else REPO / "output" / f"{args.prefix}-{args.suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    videos, walls, missing = {}, {}, []
    for arm in args.arms:
        for seed in args.seeds:
            run_id = run_id_for(args.prefix, arm, seed, args.suffix)
            mp4 = out_dir / f"{run_id}.mp4"
            manifest = out_dir / f"{run_id}.manifest.json"
            if not (fetch(run_id, "mp4", mp4) and fetch(run_id, "manifest.json", manifest)):
                missing.append(run_id)
                continue
            record = json.loads(manifest.read_text(encoding="utf-8"))
            if record.get("status") != "succeeded":
                missing.append(f"{run_id} ({record.get('status')})")
                continue
            videos.setdefault(arm, {})[seed] = mp4
            walls.setdefault(arm, []).append(record.get("wall_seconds", float("nan")))

    if missing:
        print("not available:")
        for item in missing:
            print(f"  {item}")
    paths = [str(p) for per_seed in videos.values() for p in per_seed.values()]
    if not paths:
        print("nothing to measure")
        return

    metrics_file = out_dir / "screen-metrics.json"
    subprocess.run([sys.executable, str(REPO / "video_metrics.py"),
                    str(metrics_file), *paths], cwd=str(REPO), check=True)
    measured = {r["file"]: r for r in json.loads(metrics_file.read_text(encoding="utf-8"))}

    lo, hi = args.window
    def flicker(arm, seed):
        series = measured[videos[arm][seed].name]["metrics"]["flicker"]["per_second"]
        return float(np.mean(series[lo:hi]))

    print(f"\nmean flicker over seconds {lo}-{hi} (lower is better)\n")
    header = f"{'arm':<20}" + "".join(f"{'seed ' + str(s):>10}" for s in args.seeds)
    print(header + f"{'wall s':>10}")
    for arm in args.arms:
        if arm not in videos:
            continue
        cells = "".join(f"{flicker(arm, s):>10.2f}" if s in videos[arm] else f"{'-':>10}"
                        for s in args.seeds)
        print(f"{arm:<20}{cells}{np.mean(walls[arm]):>10.0f}")

    base = args.baseline
    if base and base in videos:
        print(f"\npaired against {base}, seed by seed:\n")
        for arm in args.arms:
            if arm == base or arm not in videos:
                continue
            shared = [s for s in args.seeds if s in videos[arm] and s in videos[base]]
            if not shared:
                continue
            wins = sum(1 for s in shared if flicker(arm, s) < flicker(base, s))
            detail = " ".join(
                f"s{s}:{'win ' if flicker(arm, s) < flicker(base, s) else 'loss'}"
                f"({flicker(arm, s) / flicker(base, s):.2f}x)" for s in shared)
            print(f"  {arm:<20} {wins}/{len(shared)} paired wins   {detail}")
        print("\n  Gate: an arm needs a clear majority of paired wins to go forward.")
        print("  Two wins in three is what a coin produces half the time.")

    summary = {arm: {str(s): flicker(arm, s) for s in videos[arm]} for arm in videos}
    (out_dir / "screen-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {metrics_file} and {out_dir / 'screen-summary.json'}")


if __name__ == "__main__":
    main()
