"""Standalone text-to-video runner for BRAINROT-V."""

import argparse
import hashlib
from pathlib import Path
import re

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import write_video

from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb, gpu
from pipeline.brainrot_v_inference import BrainrotVInferencePipeline
from utils.misc import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="BRAINROT-V text-to-video inference")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt_file")
    parser.add_argument("--config_path", default="configs/brainrot_v_framewise.yaml")
    parser.add_argument("--checkpoint_path", default="checkpoints/framewise/ar_diffusion.pt")
    parser.add_argument("--output_folder", default="output/brainrot_v")
    parser.add_argument("--output_file")
    parser.add_argument("--num_output_frames", type=int, default=21,
                        help="Latent frames; 21 decodes to 81 video frames")
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--mode", choices=("brainrot_v", "baseline"),
                        default="brainrot_v")
    parser.add_argument("--segmented", action="store_true",
                        help="Use horizon-safe prompt scheduling and latent context carry-over")
    parser.add_argument("--segment_frames", type=int, default=21)
    parser.add_argument("--context_frames", type=int, default=8)
    return parser.parse_args()


def _read_prompts(args):
    if args.prompt_file:
        text = Path(args.prompt_file).read_text(encoding="utf-8")
        prompts = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^\[[^]]+\]\s*(.*?)(?:\s*\|\s*)?$", line)
            prompts.append((match.group(1) if match else line).strip())
        return prompts
    return [args.prompt.strip()]


def _section_counts(total_frames, section_count):
    # One initial latent followed by an even allocation of temporal strides.
    strides, remainder = divmod(total_frames - 1, section_count)
    counts = [strides + 1] + [strides] * (section_count - 1)
    for index in range(remainder):
        counts[index % section_count] += 1
    return counts


@torch.no_grad()
def _segmented_latent_rollout(pipeline, noise, prompts, segment_frames,
                              context_frames):
    if segment_frames < 1 or segment_frames > pipeline.max_train_frames:
        raise ValueError("segment_frames must be within the checkpoint horizon")

    generated = []
    noise_index = 0
    for prompt, section_frames in zip(
            prompts, _section_counts(noise.shape[1], len(prompts))):
        remaining = section_frames
        while remaining:
            count = min(segment_frames, remaining)
            segment_noise = noise[:, noise_index:noise_index + count]
            result = pipeline.inference(
                noise=segment_noise, text_prompts=[prompt],
                initial_latent=None, return_video=False,
                start_frame_index=0)
            generated.append(result[:, -count:].clone())
            noise_index += count
            remaining -= count
    if noise_index != noise.shape[1]:
        raise RuntimeError("segmented rollout did not consume all noise frames")
    return torch.cat(generated, dim=1)


def main():
    args = parse_args()
    prompts = _read_prompts(args)
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("prompt must not be empty")
    if args.num_output_frames < 1 or args.sampling_steps < 1:
        raise ValueError("frame and sampling-step counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("BRAINROT-V inference requires a CUDA GPU")

    set_seed(args.seed)
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    config = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config_path))
    config.memory_config.mode = args.mode
    if args.segmented and args.mode == "brainrot_v":
        # Sparse within-window compaction creates an abrupt attention-distribution
        # shift in this checkpoint. Long generation already bounds memory by
        # resetting and prefilling each segment, so retain the complete trained
        # 21-frame context inside every segment instead of pruning spatial tokens.
        config.memory_config.window_blocks = 20
        config.memory_config.keep_tokens_per_block = 0
        config.memory_config.bank_capacity = 0
    pipeline = BrainrotVInferencePipeline(config, device=device)
    pipeline.sampling_steps = args.sampling_steps

    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    key = "generator_ema" if args.use_ema else "generator"
    if key not in state_dict:
        raise KeyError(f"Checkpoint has no {key!r} weights")
    generator_state = state_dict[key]
    try:
        pipeline.generator.load_state_dict(generator_state)
    except RuntimeError:
        fixed = {
            name.replace("model._fsdp_wrapped_module.", "model.", 1): value
            for name, value in generator_state.items()
        }
        pipeline.generator.load_state_dict(fixed, strict=False)

    pipeline = pipeline.to(dtype=torch.bfloat16)
    if get_cuda_free_memory_gb(gpu) < 40:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)

    noise = torch.randn(
        [1, args.num_output_frames, 16, 60, 104],
        device=device, dtype=torch.bfloat16)
    if args.segmented:
        latents = _segmented_latent_rollout(
            pipeline, noise, prompts, args.segment_frames, args.context_frames)
        video = pipeline.vae.decode_to_pixel(latents)
        video = (video * 0.5 + 0.5).clamp(0, 1)
    else:
        video = pipeline.inference(noise=noise, text_prompts=prompts)
    video = 255.0 * rearrange(video, "b t c h w -> b t h w c").cpu()

    prompt_text = "\n".join(prompts)
    prompt_id = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12]
    output = (Path(args.output_file) if args.output_file else
              Path(args.output_folder) / f"brainrot-v-{prompt_id}.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(output), video[0], fps=16)
    pipeline.vae.model.clear_cache()
    print(f"Video saved to {output.resolve()}")


if __name__ == "__main__":
    main()
