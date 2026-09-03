"""BRAINROT-V inference pipeline (paper Algorithm 1)."""

from typing import List, Optional

import torch
from tqdm import tqdm

from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from brainrot_v.memory import BrainrotVMemory, MemoryConfig
from brainrot_v.attn_probe import AttnProbe
from brainrot_v.rope_utils import reanchor_temporal_, assert_within_horizon


class BrainrotVInferencePipeline(CausalDiffusionInferencePipeline):
    """Training-free sink | memory bank | local-window KV cache."""

    def __init__(self, args, device, generator=None, text_encoder=None, vae=None):
        super().__init__(args, device, generator=generator,
                         text_encoder=text_encoder, vae=vae)
        cfg = MemoryConfig.from_value(getattr(args, "memory_config", None))
        cfg.validate(self.num_transformer_blocks)
        self.memory = BrainrotVMemory(
            cfg, self.frame_seq_length, self.num_frame_per_block)
        self.probe = AttnProbe(cfg.score_layers, self.num_transformer_blocks,
                               cfg.probe_query_chunk)
        self.max_train_frames = int(getattr(args, "max_train_frames", 21))

        if cfg.mode == "baseline":
            # The parent cache is sized for 21 frames. For longer rollouts use its
            # native rolling-window path rather than overflowing the fixed tensor.
            window = int(getattr(args, "baseline_window_frames", 21))
            if window < 1:
                raise ValueError("baseline_window_frames must be positive")
            self.local_attn_size = window
            self.generator.model.local_attn_size = window
            for block in self.generator.model.blocks:
                block.self_attn.local_attn_size = window
                block.self_attn.max_attention_size = window * self.frame_seq_length
        else:
            if self.local_attn_size != -1:
                raise ValueError("BRAINROT-V requires model_kwargs.local_attn_size=-1")
            # capacity() includes one temporary append block. Only the steady
            # retained cache must fit inside the checkpoint's trained horizon.
            steady_capacity = self.memory.capacity() - self.memory.block_len
            if steady_capacity > self.max_train_frames * self.frame_seq_length:
                raise ValueError("memory_config capacity exceeds the training horizon")
            self.probe.install()

    def _initialize_kv_cache(self, batch_size, dtype, device):
        if self.memory.cfg.mode == "baseline":
            return super()._initialize_kv_cache(batch_size, dtype, device)
        size = self.memory.capacity()
        self.kv_cache_pos, self.kv_cache_neg = [], []
        for bank in (self.kv_cache_pos, self.kv_cache_neg):
            for _ in range(self.num_transformer_blocks):
                bank.append({
                    "k": torch.zeros([batch_size, size, 12, 128],
                                     dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, size, 12, 128],
                                     dtype=dtype, device=device),
                    "global_end_index": torch.zeros(1, dtype=torch.long,
                                                     device=device),
                    "local_end_index": torch.zeros(1, dtype=torch.long,
                                                    device=device),
                })

    def _run(self, latents, cond, timestep, cache, xattn, record=False):
        if record:
            self.probe.begin_block()
        return self.generator(
            noisy_image_or_video=latents,
            conditional_dict=cond,
            timestep=timestep,
            kv_cache=cache,
            crossattn_cache=xattn,
            current_start=self.memory.filled_len,
            cache_start=0,
        )

    @staticmethod
    def _prompt_boundaries(num_frames: int, prompt_count: int) -> List[int]:
        """Exclusive latent-frame boundaries for an even timed prompt schedule."""
        base, remainder = divmod(num_frames, prompt_count)
        counts = [base + (index < remainder) for index in range(prompt_count)]
        boundaries, total = [], 0
        for count in counts:
            total += int(count)
            boundaries.append(total)
        return boundaries

    @staticmethod
    def _reset_cross_attention(cache) -> None:
        for layer in cache:
            layer["is_init"] = False

    def _scheduled_conditioning(self, text_prompts: List[str], num_frames: int):
        # Encode each section independently. This avoids truncating one flattened
        # 60-second prompt at the text encoder's 512-token limit.
        conditions = [self.text_encoder(text_prompts=[prompt])
                      for prompt in text_prompts]
        unconditional = self.text_encoder(
            text_prompts=[self.args.negative_prompt])
        return conditions, unconditional, self._prompt_boundaries(
            num_frames, len(text_prompts))

    @torch.no_grad()
    def _baseline_scheduled_inference(self, noise: torch.Tensor,
                                      text_prompts: List[str],
                                      return_latents: bool,
                                      return_video: bool):
        """Continuous rolling baseline with timed cross-attention changes."""
        batch_size, num_frames, channels, height, width = noise.shape
        if batch_size != 1:
            raise ValueError("timed prompt scheduling currently requires batch size 1")
        if num_frames % self.num_frame_per_block:
            raise ValueError("num_frames must be divisible by num_frame_per_block")

        conditions, uncond, boundaries = self._scheduled_conditioning(
            text_prompts, num_frames)
        output = torch.zeros_like(noise)
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(batch_size, noise.dtype, noise.device)
            self._initialize_crossattn_cache(batch_size, noise.dtype, noise.device)
        else:
            self._reset_cross_attention(self.crossattn_cache_pos)
            self._reset_cross_attention(self.crossattn_cache_neg)
            for cache in (self.kv_cache_pos, self.kv_cache_neg):
                for layer in cache:
                    layer["k"].zero_()
                    layer["v"].zero_()
                    layer["global_end_index"].zero_()
                    layer["local_end_index"].zero_()

        prompt_index = 0
        for frame_index in range(num_frames):
            while frame_index >= boundaries[prompt_index]:
                prompt_index += 1
                self._reset_cross_attention(self.crossattn_cache_pos)
            cond = conditions[prompt_index]
            latents = noise[:, frame_index:frame_index + 1]
            scheduler = self._initialize_sample_scheduler(noise)
            start = frame_index * self.frame_seq_length
            for t in tqdm(scheduler.timesteps):
                timestep = t * torch.ones(
                    [batch_size, 1], device=noise.device, dtype=torch.float32)
                flow_cond, _ = self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=cond,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=start,
                    cache_start=start)
                flow_uncond, _ = self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=uncond,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=start,
                    cache_start=start)
                flow = flow_uncond + self.args.guidance_scale * (
                    flow_cond - flow_uncond)
                latents = scheduler.step(
                    flow, t, latents, return_dict=False)[0]

            output[:, frame_index:frame_index + 1] = latents
            zero_t = torch.zeros(
                [batch_size, 1], device=noise.device, dtype=torch.float32)
            self.generator(
                noisy_image_or_video=latents, conditional_dict=cond,
                timestep=zero_t, kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=start, cache_start=start)
            self.generator(
                noisy_image_or_video=latents, conditional_dict=uncond,
                timestep=zero_t, kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=start, cache_start=start)

        if not return_video:
            return output
        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return (video, output) if return_latents else video

    @torch.no_grad()
    def _compact(self):
        old_len = self.memory.filled_len
        old_position_frames = self.memory.position_frames.clone()
        keep = self.memory.select()
        if keep is None:
            return
        old_frames = old_position_frames[keep]
        new_slots = torch.arange(keep.numel(), device=keep.device)
        new_frames = new_slots // self.frame_seq_length
        freqs = self.generator.model.freqs
        for cache in (self.kv_cache_pos, self.kv_cache_neg):
            BrainrotVMemory.apply_(cache, keep, old_len, keep.numel())
            for layer in cache:
                reanchor_temporal_(layer["k"], old_frames, new_frames, freqs)
        self.memory.position_frames = new_frames
        self.memory.logical_end = self.memory.filled_len

    @torch.no_grad()
    def inference(self, noise: torch.Tensor, text_prompts: List[str],
                  initial_latent: Optional[torch.Tensor] = None,
                  return_latents: bool = False,
                  start_frame_index: Optional[int] = 0,
                  return_video: bool = True):
        if self.memory.cfg.mode == "baseline" and len(text_prompts) == 1:
            return super().inference(noise, text_prompts, initial_latent,
                                     return_latents, start_frame_index,
                                     return_video)
        if self.memory.cfg.mode == "baseline":
            if initial_latent is not None:
                raise ValueError(
                    "timed baseline scheduling does not support initial_latent")
            return self._baseline_scheduled_inference(
                noise, text_prompts, return_latents, return_video)
        if self.independent_first_frame:
            raise NotImplementedError("BRAINROT-V currently requires independent_first_frame=false")

        batch_size, num_frames, channels, height, width = noise.shape
        num_context_frames = 0 if initial_latent is None else initial_latent.shape[1]
        if initial_latent is not None:
            if initial_latent.shape[0] != batch_size or initial_latent.shape[2:] != noise.shape[2:]:
                raise ValueError("initial_latent and noise shapes are incompatible")
        # The complete rollout may be arbitrarily long.  The checkpoint horizon
        # constrains only the *packed cache plus the block being evaluated*;
        # _compact() keeps that working set bounded and re-anchors its temporal
        # RoPE positions after every eviction.
        if num_frames % self.num_frame_per_block:
            raise ValueError("num_frames must be divisible by num_frame_per_block")
        num_blocks = num_frames // self.num_frame_per_block
        if batch_size != 1 and len(text_prompts) > 1:
            raise ValueError("timed prompt scheduling currently requires batch size 1")
        if len(text_prompts) == 1:
            conditions = [self.text_encoder(text_prompts=text_prompts)]
            uncond = self.text_encoder(text_prompts=[self.args.negative_prompt])
            boundaries = [num_frames]
        else:
            conditions, uncond, boundaries = self._scheduled_conditioning(
                text_prompts, num_frames)
        prompt_index = 0
        cond = conditions[0]
        output = torch.zeros(
            [batch_size, num_context_frames + num_frames, channels, height, width],
            device=noise.device, dtype=noise.dtype)

        self.memory.reset()
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(batch_size, noise.dtype, noise.device)
            self._initialize_crossattn_cache(batch_size, noise.dtype, noise.device)
        else:
            for cache in (self.crossattn_cache_pos, self.crossattn_cache_neg):
                for layer in cache:
                    layer["is_init"] = False
            for cache in (self.kv_cache_pos, self.kv_cache_neg):
                for layer in cache:
                    layer["k"].zero_()
                    layer["v"].zero_()
                    layer["global_end_index"].zero_()
                    layer["local_end_index"].zero_()

        block_len = self.num_frame_per_block * self.frame_seq_length
        if initial_latent is not None:
            zero_t = torch.zeros(
                [batch_size, self.num_frame_per_block], device=noise.device,
                dtype=torch.float32)
            for context_index in range(0, num_context_frames,
                                       self.num_frame_per_block):
                context = initial_latent[
                    :, context_index:context_index + self.num_frame_per_block]
                self._run(context, cond, zero_t, self.kv_cache_pos,
                          self.crossattn_cache_pos)
                self._run(context, uncond, zero_t, self.kv_cache_neg,
                          self.crossattn_cache_neg)
                output[:, context_index:context_index + self.num_frame_per_block] = context
                self.memory.note_append(block_len, noise.device,
                                        self.memory.filled_len)
                self._compact()

        output_frame_index = 0
        for _ in range(num_blocks):
            while output_frame_index >= boundaries[prompt_index]:
                prompt_index += 1
                cond = conditions[prompt_index]
                self._reset_cross_attention(self.crossattn_cache_pos)
            n_frames = self.num_frame_per_block
            assert_within_horizon(self.memory.filled_len, block_len,
                                  self.frame_seq_length, self.max_train_frames)
            latents = noise[:, output_frame_index:output_frame_index + n_frames]
            scheduler = self._initialize_sample_scheduler(noise)
            for t in tqdm(scheduler.timesteps):
                timestep = t * torch.ones(
                    [batch_size, n_frames], device=noise.device,
                    dtype=torch.float32)
                flow_cond, _ = self._run(
                    latents, cond, timestep, self.kv_cache_pos,
                    self.crossattn_cache_pos)
                flow_uncond, _ = self._run(
                    latents, uncond, timestep, self.kv_cache_neg,
                    self.crossattn_cache_neg)
                flow = flow_uncond + self.args.guidance_scale * (
                    flow_cond - flow_uncond)
                latents = scheduler.step(
                    flow, t, latents, return_dict=False)[0]

            output_start = num_context_frames + output_frame_index
            output[:, output_start:output_start + n_frames] = latents
            zero_t = torch.zeros(
                [batch_size, n_frames], device=noise.device,
                dtype=torch.float32)
            self._run(latents, cond, zero_t, self.kv_cache_pos,
                      self.crossattn_cache_pos, record=True)
            column_sum, query_count = self.probe.pop()
            self._run(latents, uncond, zero_t, self.kv_cache_neg,
                      self.crossattn_cache_neg)

            self.memory.note_append(block_len, noise.device,
                                    self.memory.filled_len)
            if column_sum is not None:
                self.memory.accumulate(column_sum, query_count)
            self._compact()
            output_frame_index += n_frames

        if not return_video:
            return output
        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return (video, output) if return_latents else video
