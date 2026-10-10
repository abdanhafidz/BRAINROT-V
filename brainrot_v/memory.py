from dataclasses import dataclass, fields
from typing import Any, Iterable

import math

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
    # Which cached tokens receive attention mass as saliency.
    #   "window": Algorithm 1 line 7 -- only tokens in the local window, from
    #             queries produced while they are there; banked scores freeze
    #             at compression (\S3.4, "dropped once a block is compressed").
    #   "all":    every cached token, including sink and bank (pre-18 Sep runs).
    saliency_scope: str = "window"
    # Algorithm 1 line 3: C_n <- S u TopK(G; Q_n) u W. The number of banked
    # entries the next block may attend to. 0 means dense recall (k = M_max),
    # which the paper permits and which every run before 8 Oct used.
    recall_k: int = 0

    # --- inference-time variants drawn from LLM KV-cache literature ----------
    # Spatial grid a latent frame's tokens form, after the (1, 2, 2) patchify.
    grid_height: int = 30
    grid_width: int = 52
    # SnapKV-style clustering: max-pool the saliency map over this odd kernel
    # before Top-m, so a token is kept when its *neighbourhood* matters. 0 is
    # off and reproduces Eq. 8 exactly.
    summary_pool: int = 0
    # Which score Eq. 8 ranks by. "cumulative" is Eq. 7, the mean attention a
    # token received while in the window (H2O). "last" ranks by the most recent
    # block's attention only (TOVA).
    score_mode: str = "cumulative"
    # Scissorhands-style forgetting: the running sum and count are scaled by
    # this each block, making Eq. 7 an exponentially weighted mean. 1.0 is off.
    score_decay: float = 1.0
    # Ada-KV-style budget: spend more of the token budget on frames whose
    # attention is diffuse, fewer on frames where it is concentrated. Applied
    # per frame, so the shared-index property of section 3.4 is preserved.
    adaptive_keep: bool = False

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
        if self.saliency_scope not in {"window", "all"}:
            raise ValueError(f"Unsupported saliency_scope: {self.saliency_scope}")
        if self.recall_k < 0:
            raise ValueError("recall_k must be non-negative")
        if self.score_mode not in {"cumulative", "last"}:
            raise ValueError(f"Unsupported score_mode: {self.score_mode}")
        if self.summary_pool < 0 or (self.summary_pool and not self.summary_pool % 2):
            raise ValueError("summary_pool must be 0 or an odd kernel size")
        if not 0.0 < self.score_decay <= 1.0:
            raise ValueError("score_decay must be in (0, 1]")
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
        if self.cfg.sink_tokens % self.frame_seq_length:
            # A partial sink frame would carry two temporal RoPE indices.
            raise ValueError("sink_tokens must be a whole number of latent frames")
        # Summaries the bank may hold at once; the caller narrows this to what the
        # checkpoint's temporal horizon allows via group_budget().
        self.group_limit = max(
            1, math.ceil(self.cfg.bank_capacity / max(1, self.cfg.keep_tokens_per_block))
        ) if self.cfg.keep_tokens_per_block else 0
        self.reset()

    @property
    def bank(self) -> torch.Tensor:
        """Every banked token, oldest summary first."""
        if not self.bank_groups:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(self.bank_groups)

    def group_budget(self, max_train_frames: int) -> int:
        """How many banked summaries fit beside the sink, window and current block.

        Eq. 12 bounds the cache in tokens, which is what the method needs in
        general. This checkpoint additionally bounds temporal *frames*, and one
        summary costs one frame, so the summary count needs its own budget.
        """
        f = self.frame_seq_length
        return (int(max_train_frames) - self.cfg.sink_tokens // f
                - self.cfg.window_blocks * (self.block_len // f)
                - self.block_len // f)

    def logical_frames(self) -> int:
        """Temporal RoPE frames the steady packed cache plus one block occupies."""
        f = self.frame_seq_length
        return (self.cfg.sink_tokens // f
                + self.group_limit
                + self.cfg.window_blocks * (self.block_len // f)
                + self.block_len // f)

    def capacity(self) -> int:
        # One extra block is needed while the clean-context write is being compacted.
        return (self.cfg.sink_tokens + self.cfg.bank_capacity
                + self.cfg.window_blocks * self.block_len + self.block_len)

    def reset(self) -> None:
        self.filled_len = 0
        # Temporal RoPE cursor in tokens, always a whole number of latent frames.
        # It equals filled_len until the first compaction; after that the packed
        # cache is laid out frame-aligned, so it can exceed the physical length.
        self.logical_end = 0
        # Old temporal frames of the slots kept by the last select(), for the
        # caller's key re-anchoring.
        self.reanchor_from: torch.Tensor | None = None
        # Attention mass the most recent block's queries placed on each slot.
        # Overwritten every block -- this is content-based recall (Eq. 10), not
        # the accumulated saliency that drives eviction (Eq. 7).
        self.recall_score: torch.Tensor | None = None
        self.sink = torch.empty(0, dtype=torch.long)
        # Paper \S3.7: "each banked summary receives a compressed increasing range".
        # A summary is one evicted block (Eq. 8), so the bank is kept as an ordered
        # list of per-summary groups, oldest first, not one flat tensor.
        self.bank_groups: list[torch.Tensor] = []
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
        if logical_start != self.logical_end:
            raise RuntimeError(
                f"Packed timeline discontinuity: expected {self.logical_end}, "
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
        self.logical_end += block_len

    def accumulate(self, column_sum: torch.Tensor, query_count: int) -> None:
        """Add one block's attention mass to the running saliency (Eq. 7).

        Called after ``note_append``: ``column_sum`` holds the attention the
        newest block's queries placed on every cached token.
        """
        if self.filled_len == 0 or query_count <= 0:
            return
        n = min(self.filled_len, column_sum.numel())
        if self.cfg.saliency_scope == "all":
            self.score_sum[:n].add_(column_sum[:n].float())
            self.score_count[:n].add_(float(query_count))
            return

        # Algorithm 1 line 7 updates saliency for tokens in W only, and Eq. 7
        # counts queries produced *while* a block is in the window. The newest
        # block produced these queries, so it is excluded; sink and bank tokens
        # are excluded, which freezes a banked summary's score at compression.
        newest = self.filled_len - self.block_len
        targets = [block for block in self.window
                   if block.numel() and int(block[0]) < newest]
        if not targets:
            return
        slots = torch.cat(targets)
        slots = slots[slots < n]
        if self.cfg.score_decay < 1.0:
            # Scissorhands: importance persists, but not forever. Scaling the
            # sum and the count together keeps Eq. 7 a mean -- an exponentially
            # weighted one, so recent evidence outweighs old evidence.
            self.score_sum[slots] *= self.cfg.score_decay
            self.score_count[slots] *= self.cfg.score_decay
        self.score_sum.index_add_(0, slots, column_sum[slots].float())
        self.score_count.index_add_(
            0, slots, torch.full((slots.numel(),), float(query_count),
                                 device=slots.device))

    def note_recall(self, column_sum: torch.Tensor) -> None:
        """Record how strongly the newest block's queries matched each slot."""
        n = min(self.filled_len, column_sum.numel())
        score = torch.zeros(self.filled_len, device=column_sum.device,
                            dtype=torch.float32)
        score[:n] = column_sum[:n].float()
        self.recall_score = score

    def read_index(self, block_len: int, device) -> torch.Tensor | None:
        """Slots the next block may attend over: sink, recalled bank, window, self.

        Returns None for dense recall, letting the model use its own contiguous
        read path unchanged.
        """
        k = self.cfg.recall_k
        bank_slots = self.bank
        if k <= 0 or bank_slots.numel() <= k:
            return None
        if self.recall_score is None:
            recalled = bank_slots[:k]
        else:
            scores = self.recall_score.to(bank_slots.device)[bank_slots]
            recalled = bank_slots[torch.topk(scores, k, sorted=False).indices]
        pieces = [self.sink.to(device), recalled.sort().values.to(device)]
        pieces += [block.to(device) for block in self.window]
        pieces.append(torch.arange(self.filled_len, self.filled_len + block_len,
                                   device=device))
        # Keys keep the rotary phase of their own temporal index, so the read set
        # only has to stay ordered, not contiguous.
        return torch.cat([p for p in pieces if p.numel()])

    def summary_scores(self, indices: torch.Tensor) -> torch.Tensor:
        """The score Eq. 8 ranks by, under the configured scoring rule."""
        if self.cfg.score_mode == "last" and self.recall_score is not None:
            return self.recall_score.to(indices.device)[indices]
        return self.scores(indices)

    def pooled(self, scores: torch.Tensor) -> torch.Tensor:
        """SnapKV clustering: rank by the neighbourhood's score, not the token's.

        A latent frame's tokens are a grid_height x grid_width image, so a
        max-pool over that grid makes Top-m keep coherent patches instead of
        isolated tokens scattered across the frame.
        """
        kernel = self.cfg.summary_pool
        if kernel <= 1:
            return scores
        import torch.nn.functional as F
        h, w = self.cfg.grid_height, self.cfg.grid_width
        frames = scores.numel() // (h * w)
        if frames * h * w != scores.numel():
            raise RuntimeError("summary_pool requires scores aligned to the latent grid")
        grid = scores.view(1, frames, h, w)
        smooth = F.max_pool2d(grid, kernel_size=kernel, stride=1, padding=kernel // 2)
        return smooth.reshape(-1)

    def keep_budget(self, scores: torch.Tensor) -> int:
        """How many tokens this summary gets (Ada-KV-style, per frame)."""
        m = self.cfg.keep_tokens_per_block
        if not self.cfg.adaptive_keep or scores.numel() == 0:
            return m
        # Normalised entropy of the frame's attention: 1 means perfectly
        # diffuse, 0 means all mass on one token. A concentrated frame is
        # summarised well by few tokens; a diffuse one needs more.
        p = scores.clamp_min(0)
        total = p.sum()
        if total <= 0:
            return m
        p = p / total
        entropy = -(p * (p + 1e-12).log()).sum()
        normalised = float(entropy / torch.log(torch.tensor(float(p.numel()))))
        scaled = int(round(m * (0.5 + normalised)))
        return max(m // 2, min(scaled, 2 * m))

    def scores(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() == 0:
            return torch.empty(0, device=indices.device)
        return self.score_sum[indices] / self.score_count[indices].clamp_min(1.0)

    def select(self) -> torch.Tensor | None:
        if len(self.window) <= self.cfg.window_blocks:
            return None

        evicted = self.window.pop(0)
        evicted_scores = self.summary_scores(evicted)
        keep_n = min(self.keep_budget(evicted_scores), evicted.numel())
        if keep_n:
            ranking = self.pooled(evicted_scores)
            chosen = evicted[torch.topk(ranking, keep_n, sorted=False).indices]
            self.bank_groups.append(chosen.sort().values)

        # Eq. 9: drop the lowest-saliency banked entries until the budget holds.
        total = sum(group.numel() for group in self.bank_groups)
        if total > self.cfg.bank_capacity:
            flat = torch.cat(self.bank_groups)
            owner = torch.cat([
                torch.full((group.numel(),), index, dtype=torch.long,
                           device=flat.device)
                for index, group in enumerate(self.bank_groups)])
            survivors = torch.topk(self.scores(flat), self.cfg.bank_capacity,
                                   sorted=False).indices
            self.bank_groups = [
                flat[survivors][owner[survivors] == index].sort().values
                for index in range(len(self.bank_groups))]
            self.bank_groups = [g for g in self.bank_groups if g.numel()]

        # One summary costs one temporal frame, so the number of summaries is
        # capped as well; the lowest-saliency summary goes first, as in Eq. 9.
        while self.group_limit and len(self.bank_groups) > self.group_limit:
            means = [self.scores(group).mean().item() for group in self.bank_groups]
            self.bank_groups.pop(means.index(min(means)))

        pieces: Iterable[torch.Tensor] = (self.sink, *self.bank_groups, *self.window)
        keep = torch.cat([piece for piece in pieces if piece.numel()])
        new_slots = torch.arange(keep.numel(), device=keep.device)

        # Remap role indices to the newly packed sink | bank | window layout.
        cursor = 0
        sink_len = self.sink.numel()
        self.sink = new_slots[cursor:cursor + sink_len]
        cursor += sink_len
        remapped_bank = []
        for group in self.bank_groups:
            remapped_bank.append(new_slots[cursor:cursor + group.numel()])
            cursor += group.numel()
        self.bank_groups = remapped_bank
        remapped_window = []
        for block in self.window:
            remapped_window.append(new_slots[cursor:cursor + block.numel()])
            cursor += block.numel()
        self.window = remapped_window

        self.score_sum = self.score_sum[keep].clone()
        self.score_count = self.score_count[keep].clone()
        # Preserve the old coordinates long enough for the caller to rotate the
        # selected cached keys into their newly packed temporal slots.
        self.reanchor_from = self.position_frames[keep].clone()
        self.position_frames, next_frame = self._frame_aligned_positions(
            keep.numel(), keep.device)
        self.filled_len = keep.numel()
        self.logical_end = next_frame * self.frame_seq_length
        return keep

    def _frame_aligned_positions(self, total: int,
                                 device) -> tuple[torch.Tensor, int]:
        """Temporal RoPE frame of every packed slot, one index per latent frame.

        Follows \S3.7: the sink occupies the first frames, each banked summary
        receives its own compressed index immediately after, and the window
        follows, preserving sink < memory < window. One index per summary keeps
        every banked token inside a frame drawn from a single source block, so no
        two of them share a (time, height, width) code -- a pattern that never
        occurs in training. Physical slots stay densely packed; only the temporal
        code changes.
        """
        f = self.frame_seq_length
        positions = torch.empty(total, dtype=torch.long, device=device)
        sink_len = self.sink.numel()
        positions[:sink_len] = torch.arange(sink_len, device=device) // f
        frame = sink_len // f
        for group in self.bank_groups:
            positions[group] = frame
            frame += 1
        for block in self.window:
            positions[block] = frame + torch.arange(block.numel(), device=device) // f
            frame += block.numel() // f
        return positions, frame

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
