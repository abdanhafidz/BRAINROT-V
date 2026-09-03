import torch


@torch.no_grad()
def reanchor_temporal_(keys: torch.Tensor, old_frames: torch.Tensor,
                       new_frames: torch.Tensor, freqs: torch.Tensor) -> None:
    """Move cached keys to compressed temporal RoPE positions in place."""
    n = old_frames.numel()
    if n == 0:
        return
    complex_dims = keys.shape[-1] // 2
    temporal_dims = complex_dims - 2 * (complex_dims // 3)
    table = freqs[:, :temporal_dims]
    delta = table[new_frames] * table[old_frames].conj()
    temporal = keys[:, :n, :, :2 * temporal_dims]
    shape = temporal.shape
    z = torch.view_as_complex(temporal.float().reshape(*shape[:-1], temporal_dims, 2))
    z.mul_(delta.view(1, n, 1, temporal_dims))
    temporal.copy_(torch.view_as_real(z).reshape(shape).to(keys.dtype))


def assert_within_horizon(start_token: int, block_len: int,
                          frame_seq_length: int, max_train_frames: int) -> None:
    horizon = int(frame_seq_length) * int(max_train_frames)
    if start_token < 0 or start_token + block_len > horizon:
        raise RuntimeError(
            f"Compressed RoPE range [{start_token}, {start_token + block_len}) "
            f"exceeds training horizon {horizon}"
        )
