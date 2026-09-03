from dataclasses import dataclass, fields
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class MemoryConfig:
    mode: str = "brainrot_v"
    sink_tokens: int = 1560
    window_blocks: int = 8
    keep_tokens_per_block: int = 128
    bank_capacity: int = 2048
    score_layers: tuple[int, ...] = (0, 14, 29)
    probe_query_chunk: int = 32

    @classmethod
    def from_value(cls, value: Any) -> "MemoryConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        raw = {field.name: getattr(value, field.name) for field in fields(cls)
               if hasattr(value, field.name)}
        if isinstance(value, dict):
            raw.update({k: v for k, v in value.items()
                        if k in {field.name for field in fields(cls)}})
        if "score_layers" in raw:
            raw["score_layers"] = tuple(int(v) for v in raw["score_layers"])
        return cls(**raw)

    def validate(self, num_layers: int) -> None:
        if self.mode not in {"baseline", "brainrot_v"}:
            raise ValueError(f"Unsupported memory mode: {self.mode}")
        for name in ("sink_tokens", "window_blocks", "keep_tokens_per_block",
                     "bank_capacity", "probe_query_chunk"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.window_blocks < 1 or self.probe_query_chunk < 1:
            raise ValueError("window_blocks and probe_query_chunk must be positive")
        if any(layer < 0 or layer >= num_layers for layer in self.score_layers):
            raise ValueError(f"score_layers must be within [0, {num_layers - 1}]")


class BrainrotVMemory:
    """Shared token selection state for the conditional/unconditional KV banks."""

    def __init__(self, cfg: MemoryConfig, frame_seq_length: int,
                 frames_per_block: int):
        self.cfg = MemoryConfig.from_value(cfg)
        self.frame_seq_length = int(frame_seq_length)
        self.block_len = int(frames_per_block) * self.frame_seq_length
        self.reset()

    def capacity(self) -> int:
        # One extra block is needed while the clean-context write is being compacted.
        return (self.cfg.sink_tokens + self.cfg.bank_capacity
                + self.cfg.window_blocks * self.block_len + self.block_len)

    def reset(self) -> None:
        self.filled_len = 0
        # Under compressed-timeline RoPE, this mirrors filled_len. It remains as
        # a compatibility/debug cursor for callers and tests.
        self.logical_end = 0
        self.sink = torch.empty(0, dtype=torch.long)
        self.bank = torch.empty(0, dtype=torch.long)
        self.window: list[torch.Tensor] = []
        self.score_sum: torch.Tensor | None = None
        self.score_count: torch.Tensor | None = None
        self.position_frames: torch.Tensor | None = None

    def note_append(self, block_len: int, device: torch.device,
                    logical_start: int) -> None:
        if block_len <= 0 or self.filled_len + block_len > self.capacity():
            raise RuntimeError("BRAINROT-V cache capacity invariant violated")
        if block_len % self.frame_seq_length:
            raise RuntimeError("Appended block must contain complete latent frames")
        if logical_start != self.filled_len:
            raise RuntimeError(
                f"Packed timeline discontinuity: expected {self.filled_len}, "
                f"got {logical_start}")
        start = self.filled_len
        appended = torch.arange(start, start + block_len, device=device)
        sink_needed = max(0, self.cfg.sink_tokens - self.sink.numel())
        sink_take = min(sink_needed, block_len)
        if sink_take:
            self.sink = torch.cat((self.sink.to(device), appended[:sink_take]))
        remainder = appended[sink_take:]
        if remainder.numel():
            self.window.append(remainder)

        old_frame = logical_start // self.frame_seq_length
        frame_offsets = torch.arange(block_len, device=device) // self.frame_seq_length
        appended_positions = old_frame + frame_offsets
        if self.position_frames is None:
            self.position_frames = appended_positions
            self.score_sum = torch.zeros(block_len, device=device, dtype=torch.float32)
            self.score_count = torch.zeros(block_len, device=device, dtype=torch.float32)
        else:
            self.position_frames = torch.cat((self.position_frames, appended_positions))
            self.score_sum = torch.cat((self.score_sum,
                                        torch.zeros(block_len, device=device)))
            self.score_count = torch.cat((self.score_count,
                                          torch.zeros(block_len, device=device)))
        self.filled_len += block_len
        self.logical_end = self.filled_len

    def accumulate(self, column_sum: torch.Tensor, query_count: int) -> None:
        if self.filled_len == 0 or query_count <= 0:
            return
        n = min(self.filled_len, column_sum.numel())
        self.score_sum[:n].add_(column_sum[:n].float())
        self.score_count[:n].add_(float(query_count))

    def scores(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() == 0:
            return torch.empty(0, device=indices.device)
        return self.score_sum[indices] / self.score_count[indices].clamp_min(1.0)

    def select(self) -> torch.Tensor | None:
        if len(self.window) <= self.cfg.window_blocks:
            return None

        evicted = self.window.pop(0)
        keep_n = min(self.cfg.keep_tokens_per_block, evicted.numel())
        if keep_n:
            evicted_scores = self.scores(evicted)
            chosen = evicted[torch.topk(evicted_scores, keep_n, sorted=False).indices]
            chosen = chosen.sort().values
            self.bank = torch.cat((self.bank.to(chosen.device), chosen))

        if self.bank.numel() > self.cfg.bank_capacity:
            bank_scores = self.scores(self.bank)
            selected = torch.topk(bank_scores, self.cfg.bank_capacity,
                                  sorted=False).indices
            self.bank = self.bank[selected].sort().values

        pieces: Iterable[torch.Tensor] = (self.sink, self.bank, *self.window)
        keep = torch.cat([piece for piece in pieces if piece.numel()])
        new_slots = torch.arange(keep.numel(), device=keep.device)

        # Remap role indices to the newly packed sink | bank | window layout.
        cursor = 0
        sink_len = self.sink.numel()
        bank_len = self.bank.numel()
        self.sink = new_slots[cursor:cursor + sink_len]
        cursor += sink_len
        self.bank = new_slots[cursor:cursor + bank_len]
        cursor += bank_len
        remapped_window = []
        for block in self.window:
            remapped_window.append(new_slots[cursor:cursor + block.numel()])
            cursor += block.numel()
        self.window = remapped_window

        self.score_sum = self.score_sum[keep].clone()
        self.score_count = self.score_count[keep].clone()
        # Preserve the old coordinates long enough for the caller to rotate the
        # selected cached keys into their newly packed temporal slots.
        self.position_frames = self.position_frames[keep].clone()
        self.filled_len = keep.numel()
        self.logical_end = self.filled_len
        return keep

    @staticmethod
    def apply_(cache: list[dict], keep: torch.Tensor, old_len: int,
               logical_end: int) -> None:
        new_len = keep.numel()
        for layer in cache:
            gathered_k = layer["k"][:, keep].clone()
            gathered_v = layer["v"][:, keep].clone()
            layer["k"][:, :new_len].copy_(gathered_k)
            layer["v"][:, :new_len].copy_(gathered_v)
            if old_len > new_len:
                layer["k"][:, new_len:old_len].zero_()
                layer["v"][:, new_len:old_len].zero_()
            # The causal attention implementation deliberately has separate
            # physical (local) and logical (global) cursors. Preserve that split.
            layer["global_end_index"].fill_(logical_end)
            layer["local_end_index"].fill_(new_len)
