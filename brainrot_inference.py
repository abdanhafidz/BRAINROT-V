"""Standalone text-to-video runner for BRAINROT-V."""

import argparse
import hashlib
import json
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


# ``[0-5 seconds]``, ``[5 - 10 s]``, ``[10-15]`` -- the leading span marker of a
# timed prompt line.  The two numbers are the section's start and end in seconds.
_SPAN_MARKER = re.compile(
    r"^\[\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?\s*\]\s*")
_ANY_MARKER = re.compile(r"^\[[^]]*\]\s*")


def _read_prompts(args):
    """Return the prompt sections and, when declared, their time spans.

    A timed prompt file states each section's own span.  Those declared spans are
    authoritative: they are returned alongside the text so the caller can map them
    onto latent-frame boundaries.  ``None`` means the file declared no spans and
    an even split is the only defined schedule.
    """
    if not args.prompt_file:
        return [args.prompt.strip()], None

    text = Path(args.prompt_file).read_text(encoding="utf-8")
    prompts, spans = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        span_match = _SPAN_MARKER.match(line)
        if span_match:
            spans.append((float(span_match.group(1)), float(span_match.group(2))))
            body = line[span_match.end():]
        else:
            spans.append(None)
            body = _ANY_MARKER.sub("", line)
        prompts.append(body.rstrip("|").strip())

    if any(span is None for span in spans):
        if any(span is not None for span in spans):
            raise ValueError(
                f"{args.prompt_file}: every prompt line must declare a time span, "
                "or none may; a partially timed file has no defined schedule")
        return prompts, None

    for index in range(1, len(spans)):
        if abs(spans[index][0] - spans[index - 1][1]) > 1e-6:
            raise ValueError(
                f"{args.prompt_file}: section {index + 1} starts at "
                f"{spans[index][0]}s but section {index} ends at "
                f"{spans[index - 1][1]}s; timed sections must be contiguous")
    if spans[0][0] != 0.0:
        raise ValueError(f"{args.prompt_file}: the first section must start at 0 seconds")
    if any(end <= start for start, end in spans):
        raise ValueError(f"{args.prompt_file}: every section must have positive duration")
    return prompts, spans


def _latent_frames_for_seconds(seconds, fps=16, temporal_stride=4):
    """Smallest latent-frame count whose decode covers ``seconds`` of video.

    ``n`` latent frames decode to ``(n - 1) * temporal_stride + 1`` pixel frames,
    so this is that relation inverted and rounded to the nearest whole frame.
    """
    pixel_frames = seconds * fps
    return int(round((pixel_frames - 1) / temporal_stride)) + 1


def _boundaries_from_spans(spans, num_frames, total_seconds):
    """Exclusive latent-frame boundaries honouring each section's declared span.

    The declared schedule is rescaled onto the rollout that was actually
    requested, so a 60-second prompt file stays correct when rendered at a
    shorter duration for a smoke test.
    """
    declared_total = spans[-1][1]
    scale = total_seconds / declared_total
    boundaries, previous = [], 0
    for index, (_, end) in enumerate(spans):
        frame = _latent_frames_for_seconds(end * scale)
        # Every section must own at least one latent frame, and the schedule must
        # end exactly on the last frame of the rollout.
        frame = max(previous + 1, min(frame, num_frames - (len(spans) - index - 1)))
        boundaries.append(frame)
        previous = frame
    boundaries[-1] = num_frames
    if boundaries != sorted(boundaries) or boundaries[0] < 1:
        raise ValueError(
            f"{len(spans)} timed sections cannot be scheduled over "
            f"{num_frames} latent frames")
    return boundaries


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
    prompts, spans = _read_prompts(args)
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

    total_seconds = ((args.num_output_frames - 1) * 4 + 1) / 16
    boundaries = None
    if spans is not None and len(prompts) > 1:
        boundaries = _boundaries_from_spans(
            spans, args.num_output_frames, total_seconds)
    print(f"Rollout: {args.num_output_frames} latent frames "
          f"-> {(args.num_output_frames - 1) * 4 + 1} pixel frames "
          f"({total_seconds:.4f}s at 16 FPS)")
    if boundaries is not None:
        previous = 0
        for (start, end), frame in zip(spans, boundaries):
            print(f"  section [{start:g}-{end:g}s] -> latent frames "
                  f"[{previous}, {frame}) "
                  f"= {((previous - 1) * 4 + 1) / 16 if previous else 0:.4f}"
                  f"-{((frame - 1) * 4 + 1) / 16:.4f}s")
            previous = frame
    elif len(prompts) > 1:
        print(f"  {len(prompts)} untimed sections -> even latent-frame split")

    noise = torch.randn(
        [1, args.num_output_frames, 16, 60, 104],
        device=device, dtype=torch.bfloat16)
    if args.segmented:
        latents = _segmented_latent_rollout(
            pipeline, noise, prompts, args.segment_frames, args.context_frames)
        video = pipeline.vae.decode_to_pixel(latents)
        video = (video * 0.5 + 0.5).clamp(0, 1)
    else:
        video = pipeline.inference(noise=noise, text_prompts=prompts,
                                   prompt_boundaries=boundaries)
    video = 255.0 * rearrange(video, "b t c h w -> b t h w c").cpu()

    prompt_text = "\n".join(prompts)
    prompt_id = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12]
    output = (Path(args.output_file) if args.output_file else
              Path(args.output_folder) / f"brainrot-v-{prompt_id}.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(output), video[0], fps=16)
    pipeline.vae.model.clear_cache()

    # The resolved schedule is part of the result, not just a log line: without it
    # a finished MP4 cannot be checked against the schedule it claims to realize.
    sections = None
    if boundaries is not None:
        sections, start = [], 0
        for index, end in enumerate(boundaries):
            sections.append({
                "index": index,
                "declared_span_seconds": list(spans[index]),
                "latent_range": [start, end],
                "seconds_range": [
                    round(((start - 1) * 4 + 1) / 16, 4) if start else 0.0,
                    round(((end - 1) * 4 + 1) / 16, 4)],
                "text": prompts[index],
            })
            start = end

    schedule = output.with_suffix(".schedule.json")
    schedule.write_text(json.dumps({
        "mode": args.mode,
        "seed": args.seed,
        "sampling_steps": args.sampling_steps,
        "latent_frames": args.num_output_frames,
        "pixel_frames": int(video.shape[1]),
        "fps": 16,
        "duration_seconds": round(total_seconds, 4),
        "prompt_file": args.prompt_file,
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "section_count": len(prompts),
        "schedule_source": ("declared" if boundaries is not None
                            else "even-split" if len(prompts) > 1 else "single"),
        "latent_boundaries": boundaries,
        "sections": sections,
    }, indent=2), encoding="utf-8")
    print(f"Video saved to {output.resolve()}")
    print(f"Schedule saved to {schedule.resolve()}")


if __name__ == "__main__":
    main()
