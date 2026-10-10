"""Modal.com runner for one-prompt BRAINROT-V inference."""

from pathlib import Path
import subprocess
import sys

import modal

sys.path.insert(0, "/root/causal_forcing")
from modal_app import MODEL_ROOT, REMOTE_REPO, image, model_volume


app = modal.App("brainrot-v-inference")
OUTPUT_ROOT = "/outputs"
output_volume = modal.Volume.from_name(
    "brainrot-v-outputs", create_if_missing=True)


def _download_models() -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download

    wan_dir = Path(MODEL_ROOT) / "wan_models" / "Wan2.1-T2V-1.3B"
    if not (wan_dir / "Wan2.1_VAE.pth").exists():
        snapshot_download(repo_id="Wan-AI/Wan2.1-T2V-1.3B", local_dir=wan_dir)
    checkpoint = Path(MODEL_ROOT) / "checkpoints" / "framewise" / "ar_diffusion.pt"
    if not checkpoint.exists():
        hf_hub_download(
            repo_id="zhuhz22/Causal-Forcing",
            filename="framewise/ar_diffusion.pt",
            local_dir=Path(MODEL_ROOT) / "checkpoints")
    model_volume.commit()
    return checkpoint


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=90 * 60,
    volumes={MODEL_ROOT: model_volume},
)
def generate(prompt: str, seed: int = 0, num_frames: int = 21,
             sampling_steps: int = 50, mode: str = "brainrot_v") -> bytes:
    if mode not in {"brainrot_v", "baseline"}:
        raise ValueError("mode must be brainrot_v or baseline")
    checkpoint = _download_models()
    repo = Path(REMOTE_REPO)
    wan_link = repo / "wan_models"
    if not wan_link.exists():
        wan_link.symlink_to(Path(MODEL_ROOT) / "wan_models", target_is_directory=True)
    output = Path("/tmp/brainrot-v-output.mp4")
    output.unlink(missing_ok=True)
    subprocess.run([
        "python", "brainrot_inference.py",
        "--prompt", prompt,
        "--checkpoint_path", str(checkpoint),
        "--output_file", str(output),
        "--num_output_frames", str(num_frames),
        "--sampling_steps", str(sampling_steps),
        "--seed", str(seed),
        "--mode", mode,
    ], cwd=repo, check=True)
    return output.read_bytes()


COMMIT_INTERVAL_SECONDS = 120


def _stream_to_log(command, cwd, log_path: Path) -> int:
    """Run ``command``, tee its output to ``log_path``, and commit as it goes.

    Committing periodically is the point: a run that is preempted or hits the
    function timeout still leaves the log that explains how far it got.
    """
    import time

    last_commit = time.monotonic()
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="")
            now = time.monotonic()
            if now - last_commit >= COMMIT_INTERVAL_SECONDS:
                output_volume.commit()
                last_commit = now
        return process.wait()


@app.function(
    image=image,
    gpu="A100-80GB",
    # A 60-second 50-step rollout is ~241 latent frames x 50 steps x 2 forwards.
    # The old 90-minute ceiling left no headroom for a slower cold start.
    timeout=6 * 60 * 60,
    volumes={MODEL_ROOT: model_volume, OUTPUT_ROOT: output_volume},
)
def generate_persistent(prompt: str, run_id: str, seed: int = 0,
                        num_frames: int = 241, sampling_steps: int = 50,
                        mode: str = "brainrot_v",
                        config_path: str = "configs/brainrot_v_framewise.yaml") -> str:
    """Generate directly into a Volume so client disconnects cannot lose output."""
    import datetime
    import hashlib
    import json
    import time

    import torch

    if mode not in {"brainrot_v", "baseline"}:
        raise ValueError("mode must be brainrot_v or baseline")
    if not run_id or any(part in run_id for part in ("/", "\\", "..")):
        raise ValueError("run_id must be one safe path component")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    run_dir = Path(OUTPUT_ROOT) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    destination = run_dir / f"{mode}.mp4"
    destination.unlink(missing_ok=True)
    log_path = run_dir / f"{mode}.log"
    log_path.unlink(missing_ok=True)
    manifest_path = run_dir / f"{mode}.manifest.json"

    started = time.monotonic()
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest = {
        "run_id": run_id,
        "mode": mode,
        "seed": seed,
        "latent_frames": num_frames,
        "sampling_steps": sampling_steps,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_section_count": len([line for line in prompt.splitlines()
                                     if line.strip()]),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "started_at": started_at,
        "status": "running",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output_volume.commit()

    checkpoint = _download_models()
    repo = Path(REMOTE_REPO)
    wan_link = repo / "wan_models"
    if not wan_link.exists():
        wan_link.symlink_to(Path(MODEL_ROOT) / "wan_models", target_is_directory=True)
    prompt_file = Path("/tmp/brainrot-v-prompt.txt")
    prompt_file.write_text(prompt, encoding="utf-8")
    command = [
        # -u: without it the child block-buffers into the pipe and the streamed
        # log lags the run by kilobytes, which defeats the point of committing it.
        "python", "-u", "brainrot_inference.py",
        "--prompt_file", str(prompt_file),
        "--checkpoint_path", str(checkpoint),
        "--output_file", str(destination),
        "--num_output_frames", str(num_frames),
        "--sampling_steps", str(sampling_steps),
        "--seed", str(seed),
        "--mode", mode,
        "--config_path", config_path,
    ]
    manifest["config_path"] = config_path
    manifest["config"] = (repo / config_path).read_text(encoding="utf-8")
    manifest["command"] = " ".join(command)

    try:
        returncode = _stream_to_log(command, repo, log_path)
        manifest["exit_code"] = returncode
        if returncode != 0:
            manifest["status"] = "failed"
            raise subprocess.CalledProcessError(returncode, command)
        if not destination.exists():
            manifest["status"] = "failed"
            raise RuntimeError(f"{mode} exited 0 but wrote no video")
        payload = destination.read_bytes()
        manifest.update({
            "status": "succeeded",
            "output_path": str(destination),
            "output_bytes": len(payload),
            "output_sha256": hashlib.sha256(payload).hexdigest(),
        })
        schedule_path = destination.with_suffix(".schedule.json")
        if schedule_path.exists():
            manifest["schedule"] = json.loads(
                schedule_path.read_text(encoding="utf-8"))
        return str(destination)
    finally:
        manifest["wall_seconds"] = round(time.monotonic() - started, 1)
        manifest["finished_at"] = datetime.datetime.now(
            datetime.timezone.utc).isoformat()
        if manifest["status"] == "running":
            manifest["status"] = "failed"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        output_volume.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=90 * 60,
    volumes={MODEL_ROOT: model_volume},
)
def generate_pair(prompt: str, seed: int = 0, num_frames: int = 21,
                  sampling_steps: int = 50) -> dict[str, bytes]:
    """Generate baseline and BRAINROT-V with identical prompt/noise settings."""
    checkpoint = _download_models()
    repo = Path(REMOTE_REPO)
    wan_link = repo / "wan_models"
    if not wan_link.exists():
        wan_link.symlink_to(Path(MODEL_ROOT) / "wan_models", target_is_directory=True)

    results = {}
    for mode in ("baseline", "brainrot_v"):
        output = Path(f"/tmp/{mode}.mp4")
        output.unlink(missing_ok=True)
        subprocess.run([
            "python", "brainrot_inference.py",
            "--prompt", prompt,
            "--checkpoint_path", str(checkpoint),
            "--output_file", str(output),
            "--num_output_frames", str(num_frames),
            "--sampling_steps", str(sampling_steps),
            "--seed", str(seed),
            "--mode", mode,
        ], cwd=repo, check=True)
        results[mode] = output.read_bytes()
    return results


@app.local_entrypoint()
def main(
    prompt: str = "A lone astronaut walking through a neon market in the rain",
    output: str = "output/brainrot-v.mp4",
    seed: int = 0,
    num_frames: int = 21,
    sampling_steps: int = 50,
    mode: str = "brainrot_v",
):
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(
        generate.remote(prompt, seed, num_frames, sampling_steps, mode))
    print(f"Video saved to {destination}")


@app.local_entrypoint()
def compare(
    prompt_path: str = "prompts/brainrot_v_quality.txt",
    output_dir: str = "output/brainrot-v-comparison",
    duration_seconds: float = 5.0,
    sampling_steps: int = 50,
    seed: int = 0,
):
    import math

    prompt_file = Path(prompt_path).expanduser().resolve()
    prompt = prompt_file.read_text(encoding="utf-8")
    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    target_pixel_frames = math.ceil(duration_seconds * 16)
    num_frames = math.ceil((target_pixel_frames - 1) / 4) + 1
    actual_duration = ((num_frames - 1) * 4 + 1) / 16

    outputs = generate_pair.remote(
        prompt, seed, num_frames, sampling_steps)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "baseline-causal-forcing.mp4").write_bytes(outputs["baseline"])
    (destination / "proposed-brainrot-v.mp4").write_bytes(outputs["brainrot_v"])
    (destination / "evaluation.txt").write_text(
        f"prompt_file={prompt_file}\nseed={seed}\n"
        f"requested_duration_seconds={duration_seconds}\n"
        f"actual_duration_seconds={actual_duration}\n"
        f"latent_frames={num_frames}\nsampling_steps={sampling_steps}\n"
        f"prompt={prompt}\n",
        encoding="utf-8")
    print(f"Comparison saved to {destination}")


@app.local_entrypoint()
def long_compare(
    prompt_path: str = "prompts/brainrot_v_long_horizon_60s.txt",
    output_dir: str = "output/brainrot-v-long-horizon-60s",
    duration_seconds: float = 60.0,
    sampling_steps: int = 50,
    seed: int = 0,
):
    """Run the rolling-window baseline and BRAINROT-V concurrently on two A100s."""
    import math

    prompt_file = Path(prompt_path).expanduser().resolve()
    prompt = prompt_file.read_text(encoding="utf-8")
    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    target_pixel_frames = math.ceil(duration_seconds * 16)
    num_frames = math.ceil((target_pixel_frames - 1) / 4) + 1
    actual_duration = ((num_frames - 1) * 4 + 1) / 16

    calls = {
        mode: generate.spawn(prompt, seed, num_frames, sampling_steps, mode)
        for mode in ("baseline", "brainrot_v")
    }
    outputs = {mode: call.get() for mode, call in calls.items()}

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "baseline-causal-forcing-60s.mp4").write_bytes(outputs["baseline"])
    (destination / "proposed-brainrot-v-60s.mp4").write_bytes(outputs["brainrot_v"])
    (destination / "evaluation.txt").write_text(
        f"prompt_file={prompt_file}\nseed={seed}\n"
        f"requested_duration_seconds={duration_seconds}\n"
        f"actual_duration_seconds={actual_duration}\n"
        f"latent_frames={num_frames}\nsampling_steps={sampling_steps}\n"
        f"baseline_window_frames=21\nbrainrot_window_frames=12\n"
        f"brainrot_keep_tokens_per_block=256\nbrainrot_bank_capacity=2048\n"
        f"prompt={prompt}\n",
        encoding="utf-8")
    print(f"Long-horizon comparison saved to {destination}")


@app.local_entrypoint()
def submit_long(
    prompt_path: str = "prompts/brainrot_v_long_horizon_60s.txt",
    run_id: str = "long-horizon-60s-seed0-v2",
    duration_seconds: float = 60.0,
    sampling_steps: int = 50,
    seed: int = 0,
):
    """Submit persistent concurrent workers and return without streaming them."""
    import math

    prompt_file = Path(prompt_path).expanduser().resolve()
    prompt = prompt_file.read_text(encoding="utf-8")
    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    target_pixel_frames = math.ceil(duration_seconds * 16)
    num_frames = math.ceil((target_pixel_frames - 1) / 4) + 1
    actual_duration = ((num_frames - 1) * 4 + 1) / 16

    calls = {
        mode: generate_persistent.spawn(
            prompt, run_id, seed, num_frames, sampling_steps, mode)
        for mode in ("baseline", "brainrot_v")
    }
    print(f"Submitted detached run {run_id}")
    print(f"latent_frames={num_frames} actual_duration={actual_duration}")
    for mode, call in calls.items():
        print(f"{mode}_call_id={call.object_id}")
    print("Outputs will persist in Modal Volume brainrot-v-outputs")


@app.local_entrypoint()
def submit_one_long(
    mode: str,
    prompt_path: str = "prompts/brainrot_v_long_horizon_60s.txt",
    run_id: str = "long-horizon-60s-seed0-v2",
    duration_seconds: float = 60.0,
    sampling_steps: int = 50,
    seed: int = 0,
    config_path: str = "configs/brainrot_v_framewise.yaml",
):
    """Submit one persistent worker; intended for reliable detached execution."""
    import math

    if mode not in {"baseline", "brainrot_v"}:
        raise ValueError("mode must be baseline or brainrot_v")
    prompt_file = Path(prompt_path).expanduser().resolve()
    prompt = prompt_file.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    target_pixel_frames = math.ceil(duration_seconds * 16)
    num_frames = math.ceil((target_pixel_frames - 1) / 4) + 1
    actual_duration = ((num_frames - 1) * 4 + 1) / 16
    call = generate_persistent.spawn(
        prompt, run_id, seed, num_frames, sampling_steps, mode, config_path)
    print(f"Submitted detached {mode} call_id={call.object_id} config={config_path}")
    print(f"run_id={run_id} latent_frames={num_frames} "
          f"actual_duration={actual_duration}")
    print(f"Outputs, log and manifest will persist under "
          f"brainrot-v-outputs/{run_id}/")
