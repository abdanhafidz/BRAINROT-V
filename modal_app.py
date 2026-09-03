"""Run one-prompt Causal Forcing++ inference on Modal and download the MP4."""

from pathlib import Path
import subprocess

import modal


APP_NAME = "causal-forcing-inference"
REMOTE_REPO = "/root/causal_forcing"
MODEL_ROOT = "/models"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("causal-forcing-models", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10"
    )
    .entrypoint([])
    .apt_install("build-essential", "ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("git+https://github.com/openai/CLIP.git")
    .pip_install("wheel")
    .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation", gpu="A100-80GB")
    .add_local_dir(".", remote_path=REMOTE_REPO, copy=True)
)


def _download_models() -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download, snapshot_download

    wan_dir = Path(MODEL_ROOT) / "wan_models" / "Wan2.1-T2V-1.3B"
    checkpoint = (
        Path(MODEL_ROOT)
        / "checkpoints"
        / "causal-forcing++"
        / "framewise-2step.pt"
    )
    if not (wan_dir / "Wan2.1_VAE.pth").exists():
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-T2V-1.3B",
            local_dir=wan_dir,
        )
    if not checkpoint.exists():
        downloaded = Path(
            hf_hub_download(
                repo_id="zhuhz22/Causal-Forcing",
                filename="causal-forcing++/framewise-2step.pt",
                local_dir=Path(MODEL_ROOT) / "checkpoints",
            )
        )
        if downloaded != checkpoint:
            raise RuntimeError(f"Checkpoint downloaded to unexpected path: {downloaded}")
    model_volume.commit()
    return wan_dir, checkpoint


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 30,
    volumes={MODEL_ROOT: model_volume},
)
def generate(prompt: str, seed: int = 0, num_frames: int = 21) -> bytes:
    """Generate an MP4 remotely; only its bytes are returned to the caller."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if num_frames < 1:
        raise ValueError("num_frames must be positive")

    _, checkpoint = _download_models()
    repo = Path(REMOTE_REPO)
    wan_link = repo / "wan_models"
    if not wan_link.exists():
        wan_link.symlink_to(Path(MODEL_ROOT) / "wan_models", target_is_directory=True)

    output = Path("/tmp/causal-forcing-output.mp4")
    output.unlink(missing_ok=True)
    subprocess.run(
        [
            "python",
            "inference.py",
            "--config_path",
            "configs/causal_forcing_dmd_framewise_2step.yaml",
            "--checkpoint_path",
            str(checkpoint),
            "--output_folder",
            "/tmp",
            "--output_file",
            str(output),
            "--prompt",
            prompt,
            "--num_output_frames",
            str(num_frames),
            "--seed",
            str(seed),
            "--use_ema",
        ],
        cwd=repo,
        check=True,
    )
    return output.read_bytes()


@app.local_entrypoint()
def main(
    prompt: str = "A cinematic tracking shot of a red fox running through a misty pine forest at sunrise",
    output: str = "output/modal-causal-forcing.mp4",
    seed: int = 0,
    num_frames: int = 21,
):
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(generate.remote(prompt, seed, num_frames))
    print(f"Video saved to {destination}")
