"""Objective degradation metrics for long-horizon rollouts.

The saturation proxy used earlier confounded artifacts with genuine prompt
adherence: a scene that legitimately turns blue scored as "collapsed". These
metrics instead target the three failure modes directly, and each one is
calibrated against the clip's own opening seconds rather than an absolute
threshold, so a run is judged against the quality it started at.

  flicker   second difference of the frame's global luma and chroma means. A
            moving subject or a smooth lighting ramp barely moves a global mean
            and is linear in time, so both cancel; frame-to-frame jitter does
            not. (Mean per-pixel |dY| was tried first and simply measured
            motion: it fired at 6 s on every run, clean or not.)
  speckle   Laplacian energy of the chroma planes. Isolated coloured pixels --
            the blocky rainbow noise on the ground -- have high chroma
            curvature, while a genuinely colourful but smooth scene does not.
  streak    ratio of horizontal to vertical gradient energy in luma. Vertical
            banding creates strong gradients across x and little variation
            along y, so the ratio climbs well above its opening value.

``clean horizon`` is the first second at which any metric stays above its
tolerance for a sustained window, i.e. the point the output stops resembling
how it began.

Usage:
    python video_metrics.py out.json a.mp4 [b.mp4 ...]
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

FPS = 16
BASELINE_SECONDS = 5          # the opening stretch every clip is judged against
SUSTAIN_SECONDS = 2           # a breach must persist this long to count
TOLERANCE = {"flicker": 3.0, "speckle": 1.5, "streak": 1.5}   # multiples of baseline


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"]), int(s["nb_frames"])


def frames(path, width, height):
    """Yield (Y, U, V) planes per frame, full resolution, as float32."""
    ysize, csize = width * height, (width // 2) * (height // 2)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "yuv420p", "-"], stdout=subprocess.PIPE, bufsize=10 ** 8)
    try:
        while True:
            buf = proc.stdout.read(ysize + 2 * csize)
            if len(buf) < ysize + 2 * csize:
                return
            raw = np.frombuffer(buf, dtype=np.uint8)
            y = raw[:ysize].reshape(height, width).astype(np.float32)
            u = raw[ysize:ysize + csize].reshape(height // 2, width // 2).astype(np.float32)
            v = raw[ysize + csize:].reshape(height // 2, width // 2).astype(np.float32)
            yield y, u, v
    finally:
        proc.stdout.close()
        proc.wait()


def laplacian_energy(plane):
    lap = (4.0 * plane[1:-1, 1:-1] - plane[:-2, 1:-1] - plane[2:, 1:-1]
           - plane[1:-1, :-2] - plane[1:-1, 2:])
    return float(np.abs(lap).mean())


def gradient_ratio(y):
    dx = float(np.abs(np.diff(y, axis=1)).mean())
    dy = float(np.abs(np.diff(y, axis=0)).mean())
    return dx / max(dy, 1e-6)


def measure(path):
    width, height, _ = probe(path)
    per_frame = {"speckle": [], "streak": []}
    means = []
    for y, u, v in frames(path, width, height):
        per_frame["speckle"].append(
            0.5 * (laplacian_energy(u) + laplacian_energy(v)))
        per_frame["streak"].append(gradient_ratio(y))
        means.append((float(y.mean()), float(u.mean()), float(v.mean())))
    out = {k: np.asarray(v, dtype=np.float64) for k, v in per_frame.items()}
    # |2*m_t - m_(t-1) - m_(t+1)| summed over Y, U, V: zero for any sequence of
    # global means that moves linearly, large for frame-to-frame jitter.
    m = np.asarray(means, dtype=np.float64)
    jitter = np.abs(2.0 * m[1:-1] - m[:-2] - m[2:]).sum(axis=1)
    out["flicker"] = np.concatenate(([0.0], jitter, [0.0]))
    return out


def per_second(series):
    n = len(series) // FPS
    return np.asarray([series[i * FPS:(i + 1) * FPS].mean() for i in range(n)])


def analyse(path):
    raw = measure(path)
    seconds = {k: per_second(v) for k, v in raw.items()}
    # The first frame has no predecessor, so flicker's baseline skips second 0.
    report, breaches = {}, {}
    for name, series in seconds.items():
        base_slice = series[1:BASELINE_SECONDS] if name == "flicker" else series[:BASELINE_SECONDS]
        # A near-zero opening would make any later value a huge multiple.
        floor = {"flicker": 0.05, "speckle": 0.5, "streak": 0.5}[name]
        base = max(float(np.median(base_slice)), floor)
        limit = base * TOLERANCE[name]
        over = series > limit
        first = None
        for i in range(len(over) - SUSTAIN_SECONDS + 1):
            if over[i:i + SUSTAIN_SECONDS].all():
                first = i
                break
        report[name] = {
            "baseline": round(base, 4),
            "limit": round(limit, 4),
            "first_breach_second": first,
            "per_second": [round(float(x), 4) for x in series],
        }
        breaches[name] = first
    valid = [v for v in breaches.values() if v is not None]
    return {
        "file": Path(path).name,
        "clean_horizon_seconds": min(valid) if valid else None,
        "first_breach_by_metric": breaches,
        "metrics": report,
    }


def main():
    destination, videos = Path(sys.argv[1]), sys.argv[2:]
    results = [analyse(v) for v in videos]
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    width = max(len(r["file"]) for r in results)
    print(f"{'file':<{width}}  clean_to  flicker  speckle  streak")
    for r in results:
        b = r["first_breach_by_metric"]
        horizon = r["clean_horizon_seconds"]
        print(f"{r['file']:<{width}}  {str(horizon) + ' s' if horizon is not None else 'full':>8}  "
              f"{str(b['flicker']):>7}  {str(b['speckle']):>7}  {str(b['streak']):>6}")
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
