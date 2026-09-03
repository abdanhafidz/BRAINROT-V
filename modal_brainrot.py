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


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=90 * 60,
    volumes={MODEL_ROOT: model_volume, OUTPUT_ROOT: output_volume},
)
def generate_persistent(prompt: str, run_id: str, seed: int = 0,
                        num_frames: int = 241, sampling_steps: int = 50,
                        mode: str = "brainrot_v") -> str:
    """Generate directly into a Volume so client disconnects cannot lose output."""
    if mode not in {"brainrot_v", "baseline"}:
        raise ValueError("mode must be brainrot_v or baseline")
    if not run_id or any(part in run_id for part in ("/", "\\", "..")):
        raise ValueError("run_id must be one safe path component")
    checkpoint = _download_models()
    repo = Path(REMOTE_REPO)
    wan_link = repo / "wan_models"
    if not wan_link.exists():
        wan_link.symlink_to(Path(MODEL_ROOT) / "wan_models", target_is_directory=True)
    destination = Path(OUTPUT_ROOT) / run_id / f"{mode}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    prompt_file = Path("/tmp/brainrot-v-prompt.txt")
    prompt_file.write_text(prompt, encoding="utf-8")
    command = [
        "python", "brainrot_inference.py",
        "--prompt_file", str(prompt_file),
        "--checkpoint_path", str(checkpoint),
        "--output_file", str(destination),
        "--num_output_frames", str(num_frames),
        "--sampling_steps", str(sampling_steps),
        "--seed", str(seed),
        "--mode", mode,
    ]
    subprocess.run(command, cwd=repo, check=True)
    output_volume.commit()
    return str(destination)


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
):
    """Submit one persistent worker; intended for reliable detached execution."""
    import math

    if mode not in {"baseline", "brainrot_v"}:
        raise ValueError("mode must be baseline or brainrot_v")
    prompt_file = Path(prompt_path).expanduser().resolve()
    prompt = prompt_file.read_text(encoding="utf-8")
    target_pixel_frames = math.ceil(duration_seconds * 16)
    num_frames = math.ceil((target_pixel_frames - 1) / 4) + 1
    call = generate_persistent.spawn(
        prompt, run_id, seed, num_frames, sampling_steps, mode)
    print(f"Submitted detached {mode} call_id={call.object_id}")
