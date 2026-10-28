"""Run one generation locally, producing the same record a Modal run produced.

The Modal worker's value was never the GPU alone: every run left a streamed log,
a manifest carrying the exact configuration, and a verifiable checksum, which is
what makes the experiment record in abdanslog auditable. Moving to a local GPU
should not cost that, so this writes the identical artifacts under identical
names -- `video_metrics.py`, `subject_metrics.py` and the documentation workflow
keep working unchanged.

    python run_local.py --run-id my-run --seed 0 --duration-seconds 60 \
        --config-path configs/brainrot_v_framewise.yaml
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = REPO / "output" / "local-runs"


def parse_args():
    p = argparse.ArgumentParser(description="Local Brainrot-V run with a full record")
    p.add_argument("--run-id", required=True, help="One safe path component")
    p.add_argument("--mode", default="brainrot_v", choices=("brainrot_v", "baseline"))
    p.add_argument("--prompt-path", default="prompts/brainrot_v_long_horizon_60s.txt")
    p.add_argument("--config-path", default="configs/brainrot_v_framewise.yaml")
    p.add_argument("--checkpoint-path", default="checkpoints/framewise/ar_diffusion.pt")
    p.add_argument("--duration-seconds", type=float, default=60.0)
    p.add_argument("--sampling-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return p.parse_args()


def latent_frames_for(duration_seconds: float) -> int:
    """Match the frame arithmetic the Modal entrypoints use, exactly."""
    target_pixel_frames = math.ceil(duration_seconds * 16)
    return math.ceil((target_pixel_frames - 1) / 4) + 1


def gpu_name() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def torch_version() -> str | None:
    try:
        import torch
        return torch.__version__
    except Exception:
        return None


def stream_to_log(command, log_path: Path) -> int:
    """Run the command, tee its output to the log and to this console."""
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command, cwd=str(REPO), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return process.wait()


def main() -> int:
    args = parse_args()
    if not args.run_id or any(part in args.run_id for part in ("/", "\\", "..")):
        raise SystemExit("run_id must be one safe path component")

    prompt_file = REPO / args.prompt_path
    config_file = REPO / args.config_path
    for required in (prompt_file, config_file):
        if not required.exists():
            raise SystemExit(f"missing: {required}")
    if not prompt_file.read_text(encoding="utf-8").strip():
        raise SystemExit(f"prompt file is empty: {prompt_file}")

    run_dir = Path(args.output_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    destination = run_dir / f"{args.mode}.mp4"
    destination.unlink(missing_ok=True)
    log_path = run_dir / f"{args.mode}.log"
    log_path.unlink(missing_ok=True)
    manifest_path = run_dir / f"{args.mode}.manifest.json"

    num_frames = latent_frames_for(args.duration_seconds)
    prompt_text = prompt_file.read_text(encoding="utf-8")
    started = time.monotonic()
    manifest = {
        "run_id": args.run_id,
        "mode": args.mode,
        "seed": args.seed,
        "latent_frames": num_frames,
        "actual_duration_seconds": ((num_frames - 1) * 4 + 1) / 16,
        "sampling_steps": args.sampling_steps,
        "prompt_path": args.prompt_path,
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "prompt_section_count": len([l for l in prompt_text.splitlines() if l.strip()]),
        "config_path": args.config_path,
        "config": config_file.read_text(encoding="utf-8"),
        "gpu": gpu_name(),
        "torch_version": torch_version(),
        "host": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME"),
        "runner": "local",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    command = [
        # -u so the log is written as the run proceeds, not in buffered chunks.
        sys.executable, "-u", "brainrot_inference.py",
        "--prompt_file", str(prompt_file),
        "--config_path", str(config_file),
        "--checkpoint_path", args.checkpoint_path,
        "--output_file", str(destination),
        "--num_output_frames", str(num_frames),
        "--sampling_steps", str(args.sampling_steps),
        "--seed", str(args.seed),
        "--mode", args.mode,
    ]
    manifest["command"] = " ".join(command)

    try:
        returncode = stream_to_log(command, log_path)
        manifest["exit_code"] = returncode
        if returncode != 0:
            manifest["status"] = "failed"
            return returncode
        if not destination.exists():
            manifest["status"] = "failed"
            manifest["error"] = "exited 0 but wrote no video"
            return 1
        payload = destination.read_bytes()
        manifest.update({
            "status": "succeeded",
            "output_path": str(destination),
            "output_bytes": len(payload),
            "output_sha256": hashlib.sha256(payload).hexdigest(),
        })
        schedule = destination.with_suffix(".schedule.json")
        if schedule.exists():
            manifest["schedule"] = json.loads(schedule.read_text(encoding="utf-8"))
        return 0
    finally:
        manifest["wall_seconds"] = round(time.monotonic() - started, 1)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        if manifest["status"] == "running":
            manifest["status"] = "failed"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\n[{manifest['status']}] {args.run_id} "
              f"in {manifest['wall_seconds']}s -> {run_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
