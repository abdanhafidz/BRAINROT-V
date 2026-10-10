"""Count the subject per second of a finished run, on Modal.

Duplicated subjects are the dominant remaining artifact, and the damage metrics in
``video_metrics.py`` cannot see them: a frame holding three foxes is perfectly
smooth and unstreaked. A colour-threshold count was tried first and failed --
the sunrise forest puts 8-15% of its pixels in the same hue band as the fur, so
it reported five or six foxes in frames that plainly hold one.

This uses a COCO detector instead and counts animal detections. A red fox is
read as dog/cat/bear by such a model, which is fine: the question is how many
animals are in frame, not which species.

    modal run modal_subject_count.py::app.count --run-id <volume run id>
"""

from pathlib import Path
import json
import subprocess
import sys

import modal

# In the container this file sits at /root while the repo is copied to
# /root/causal_forcing, so modal_app is not importable without this.
sys.path.insert(0, "/root/causal_forcing")
from modal_app import image

app = modal.App("brainrot-v-subject-count")
OUTPUT_ROOT = "/outputs"
output_volume = modal.Volume.from_name("brainrot-v-outputs", create_if_missing=True)

# COCO animal classes; a fox lands among these.
ANIMAL_IDS = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25}
SCORE_MIN = 0.5
FPS_SAMPLE = 1          # one frame per second is enough to count subjects
NMS_IOU = 0.45          # merge boxes that describe the same animal


# Sixty sampled frames do not need a GPU, and CPU avoids queueing for one.
@app.function(image=image, cpu=8.0, timeout=60 * 60,
              volumes={OUTPUT_ROOT: output_volume})
def count(run_id: str, filename: str = "brainrot_v.mp4") -> dict:
    import numpy as np
    import torch
    from torchvision.models import detection

    video = Path(OUTPUT_ROOT) / run_id / filename
    if not video.exists():
        raise FileNotFoundError(str(video))

    frames_dir = Path("/tmp/frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video),
                    "-vf", f"fps={FPS_SAMPLE}", str(frames_dir / "%04d.png")],
                   check=True)
    paths = sorted(frames_dir.glob("*.png"))

    weights = detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = detection.fasterrcnn_resnet50_fpn(weights=weights).eval()
    transform = weights.transforms()

    per_second, detail = [], []
    with torch.no_grad():
        for path in paths:
            import PIL.Image
            img = PIL.Image.open(path).convert("RGB")
            tensor = transform(img)
            out = model([tensor])[0]
            mask = torch.tensor(
                [int(l) in ANIMAL_IDS and float(sc) >= SCORE_MIN
                 for l, sc in zip(out["labels"], out["scores"])], dtype=torch.bool)
            boxes, scores = out["boxes"][mask], out["scores"][mask]
            # One animal often yields several boxes under different classes
            # (dog and cat on the same fox). Count instances, not detections:
            # suppress overlaps class-agnostically before counting.
            if boxes.numel():
                from torchvision.ops import nms
                keep_idx = nms(boxes, scores, iou_threshold=NMS_IOU)
                boxes, scores = boxes[keep_idx], scores[keep_idx]
            per_second.append(int(boxes.shape[0]))
            detail.append({"second": len(per_second) - 1,
                           "animals": int(boxes.shape[0]),
                           "scores": [round(float(x), 3) for x in scores[:6]]})

    counts = np.asarray(per_second)
    first_dup = None
    for i in range(len(counts) - 1):
        if counts[i] > 1 and counts[i + 1] > 1:
            first_dup = i
            break
    report = {
        "run_id": run_id,
        "seconds": int(counts.size),
        "subject_count_per_second": [int(c) for c in counts],
        "first_duplication_second": first_dup,
        "seconds_duplicated": int((counts > 1).sum()),
        "seconds_with_subject": int((counts >= 1).sum()),
        "seconds_empty": int((counts == 0).sum()),
        "detail": detail,
    }
    destination = Path(OUTPUT_ROOT) / run_id / "subject_count.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_volume.commit()
    return {k: v for k, v in report.items() if k != "detail"}


@app.local_entrypoint()
def main(run_id: str, filename: str = "brainrot_v.mp4"):
    result = count.remote(run_id, filename)
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def batch(run_ids: str, filename: str = "brainrot_v.mp4"):
    """Count several runs at once; run_ids is a comma-separated list."""
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    calls = [(r, count.spawn(r, filename)) for r in ids]
    for run_id, call in calls:
        try:
            result = call.get()
            print(f"{run_id}: dup_from={result['first_duplication_second']} "
                  f"dup={result['seconds_duplicated']} "
                  f"single={result['seconds_with_subject'] - result['seconds_duplicated']} "
                  f"empty={result['seconds_empty']} of {result['seconds']}s")
        except Exception as exc:
            print(f"{run_id}: FAILED {exc}")
