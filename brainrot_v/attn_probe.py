import math

import torch


class AttnProbe:
    """Light exact-score side path for selected clean conditional layers."""

    def __init__(self, score_layers, num_layers: int, query_chunk: int = 32):
        self.score_layers = set(int(v) for v in score_layers)
        self.num_layers = int(num_layers)
        self.query_chunk = int(query_chunk)
        self.active = False
        self.layer_index = 0
        self._column_sum = None
        self._query_count = 0
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        import wan.modules.causal_model as causal_model

        original = causal_model.attention
        probe = self

        def wrapped(q, k, v, *args, **kwargs):
            output = original(q, k, v, *args, **kwargs)
            if probe.active:
                layer = probe.layer_index
                probe.layer_index += 1
                if layer in probe.score_layers:
                    probe._record(q, k, kwargs.get("softmax_scale"))
            return output

        causal_model.attention = wrapped
        self._installed = True

    def begin_block(self) -> None:
        self.active = True
        self.layer_index = 0
        self._column_sum = None
        self._query_count = 0

    @torch.no_grad()
    def _record(self, q: torch.Tensor, k: torch.Tensor, scale) -> None:
        scale = float(scale) if scale is not None else 1.0 / math.sqrt(q.shape[-1])
        column_sum = torch.zeros(k.shape[1], device=k.device, dtype=torch.float32)
        keys = k.float()
        for start in range(0, q.shape[1], self.query_chunk):
            query = q[:, start:start + self.query_chunk].float()
            logits = torch.einsum("bqhd,bkhd->bhqk", query, keys) * scale
            column_sum.add_(logits.softmax(dim=-1).sum(dim=(0, 1, 2)))
            self._query_count += query.shape[0] * query.shape[1] * query.shape[2]
        self._column_sum = (column_sum if self._column_sum is None
                            else self._column_sum + column_sum)

    def pop(self):
        self.active = False
        if self.layer_index != self.num_layers:
            raise RuntimeError(
                f"Expected {self.num_layers} self-attention calls, got {self.layer_index}"
            )
        result = self._column_sum
        count = self._query_count
        self._column_sum = None
        self._query_count = 0
        return result, count
