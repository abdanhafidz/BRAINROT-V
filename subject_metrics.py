"""Count the subject per frame, to detect duplication.

The three metrics in ``video_metrics.py`` measure damage: jitter, chroma noise,
banding. They are blind to the artifact that dominates the current output, where
the frame is clean and stable but contains three foxes when the prompt asked for
one. Duplication is a failure of *meaning*, so it needs its own measure.

The subject here is a red fox: strongly orange fur against moss, bark and sky.
Thresholding hue and counting connected components of sufficient area gives a
per-frame subject count without any model weights.

    python subject_metrics.py out.json a.mp4 [b.mp4 ...]

Limitations are real and worth stating: the count is colour-based, so an orange
background (the volcanic and neon-market sections of the long prompt) inflates
it, and a fox seen against orange will merge with it. It is reliable over the
forest sections, which is where duplication first appears, and it is a diagnostic
for one specific prompt, not a general metric.
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

FPS = 16
# Orange fur in OpenCV HSV (hue 0-179). Deliberately narrow: better to miss a
# dim fox than to count bark as one.
HUE_LO, HUE_HI = 5, 22
SAT_MIN, VAL_MIN = 110, 70
MIN_AREA_FRACTION = 0.004     # a fox occupies far more than 0.4% of the frame


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"])


def bgr_frames(path, width, height):
    size = width * height * 3
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, bufsize=10 ** 8)
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                return
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def subject_count(frame, min_area):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (HUE_LO, SAT_MIN, VAL_MIN), (HUE_HI, 255, 255))
    # Close small gaps so one fox is not split by its own scarf or shadow.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = [stats[i, cv2.CC_STAT_AREA] for i in range(1, count)
             if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return len(blobs), float(sum(blobs))


def analyse(path):
    width, height = probe(path)
    min_area = MIN_AREA_FRACTION * width * height
    counts, areas = [], []
    for frame in bgr_frames(path, width, height):
        n, a = subject_count(frame, min_area)
        counts.append(n)
        areas.append(a / (width * height))
    counts, areas = np.asarray(counts), np.asarray(areas)
    seconds = len(counts) // FPS
    per_second = [float(np.median(counts[i * FPS:(i + 1) * FPS])) for i in range(seconds)]
    area_per_second = [float(np.mean(areas[i * FPS:(i + 1) * FPS])) for i in range(seconds)]

    # First second where the median count stays above one for two seconds.
    first_dup = None
    for i in range(len(per_second) - 1):
        if per_second[i] > 1 and per_second[i + 1] > 1:
            first_dup = i
            break
    present = [i for i, c in enumerate(per_second) if c >= 1]
    return {
        "file": Path(path).name,
        "first_duplication_second": first_dup,
        "seconds_with_subject": len(present),
        "seconds_duplicated": int(sum(1 for c in per_second if c > 1)),
        "subject_count_per_second": per_second,
        "subject_area_fraction_per_second": [round(a, 4) for a in area_per_second],
    }


def main():
    destination, videos = Path(sys.argv[1]), sys.argv[2:]
    results = [analyse(v) for v in videos]
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    width = max(len(r["file"]) for r in results)
    print(f"{'file':<{width}}  dup_from  dup_secs  subj_secs  counts @5s steps")
    for r in results:
        c = r["subject_count_per_second"]
        sample = " ".join(f"{c[i]:.0f}" for i in range(0, min(len(c), 60), 5))
        print(f"{r['file']:<{width}}  "
              f"{str(r['first_duplication_second']) + 's':>8}  "
              f"{r['seconds_duplicated']:>8}  {r['seconds_with_subject']:>9}  {sample}")
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
