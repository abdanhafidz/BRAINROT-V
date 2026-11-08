"""Drive layout, checkpointing and status reporting shared by the Colab notebooks.

Colab Pro has no background execution, so a session ends whenever the browser
does. Everything here assumes the runtime can die at any instant: state lives on
Drive, writes are atomic, and every stage records what it finished so the next
session resumes instead of restarting.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

RUN_SETS = ("bridge", "pilot", "final")

# Numbered so Drive's own file listing shows the pipeline in order.
_TREE = {
    "setup": "00_setup",
    "code": "00_setup/code",
    "wheels": "00_setup/wheels",
    "models": "00_setup/models",
    "runs": "01_runs",
    "reference": "02_reference",
    "scores": "03_scores",
    "results": "04_results",
    "figures": "04_results/figures",
    "tables": "04_results/tables",
    "cache": "_cache",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path: Path, text: str) -> None:
    """Write via a temp file in the same directory, then replace.

    A plain write that is interrupted leaves a truncated file; this leaves
    either the old content or the new one, never half of either.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def write_json(path: Path, payload) -> None:
    write_atomic(path, json.dumps(payload, indent=2))


def read_json(path: Path, default=None):
    """Returns `default` for a missing file and for one left truncated."""
    if not Path(path).exists():
        return default
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return default


class Workspace:
    """Every path the notebooks touch, in one place."""

    def __init__(self, root: str | Path = "/content/drive/MyDrive/brainrot-v"):
        self.root = Path(root)
        for key, relative in _TREE.items():
            setattr(self, key, self.root / relative)

    def ensure(self) -> "Workspace":
        for relative in _TREE.values():
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        for run_set in RUN_SETS:
            for kind in ("video", "manifest", "log", "schedule"):
                (self.runs / run_set / kind).mkdir(parents=True, exist_ok=True)
        probe = self.root / ".writable"
        write_atomic(probe, utcnow())
        probe.unlink(missing_ok=True)
        self.write_readme()
        return self

    # -- per-run paths -------------------------------------------------------
    def video(self, run_set: str, run_id: str) -> Path:
        return self.runs / run_set / "video" / f"{run_id}.mp4"

    def manifest(self, run_set: str, run_id: str) -> Path:
        return self.runs / run_set / "manifest" / f"{run_id}.json"

    def log(self, run_set: str, run_id: str) -> Path:
        return self.runs / run_set / "log" / f"{run_id}.log"

    def schedule(self, run_set: str, run_id: str) -> Path:
        return self.runs / run_set / "schedule" / f"{run_id}.json"

    def score(self, name: str) -> Path:
        return self.scores / f"{name}.json"

    def figure(self, name: str) -> Path:
        return self.figures / f"{name}.png"

    def table(self, name: str) -> Path:
        return self.tables / name

    # -- state ---------------------------------------------------------------
    def run_done(self, run_set: str, run_id: str) -> bool:
        """Finished means: manifest says so, the video exists, and it still
        hashes to what the manifest recorded. A manifest alone is not enough -
        a stale one was twice mistaken for a live result on Modal."""
        record = read_json(self.manifest(run_set, run_id), {})
        if record.get("status") != "succeeded":
            return False
        video = self.video(run_set, run_id)
        return (video.exists()
                and record.get("output_sha256") == sha256_file(video))

    def inventory(self, run_set: str) -> list[dict]:
        rows = []
        for path in sorted((self.runs / run_set / "manifest").glob("*.json")):
            record = read_json(path, {}) or {}
            run_id = path.stem
            rows.append({
                "run_id": run_id,
                "status": record.get("status", "unknown"),
                "verified": self.run_done(run_set, run_id),
                "seed": record.get("seed"),
                "mode": record.get("mode"),
                "wall_seconds": record.get("wall_seconds"),
                "gpu": record.get("gpu"),
                "memory_path": record.get("memory_path"),
                "flash_attn": record.get("flash_attn"),
                "finished_at": record.get("finished_at"),
            })
        return rows

    # -- human-readable state ------------------------------------------------
    def write_readme(self) -> Path:
        path = self.root / "README.md"
        if path.exists():
            return path
        write_atomic(path, """# Brainrot-V workspace

| Folder | Holds |
| --- | --- |
| `00_setup/` | `code/causal_forcing.zip`, cached `wheels/`, 23 GB of `models/` |
| `01_runs/<set>/` | `video/`, `manifest/`, `log/`, `schedule/` per generated run |
| `02_reference/` | Clips generated elsewhere (e.g. Modal) to pair against |
| `03_scores/` | Cached metric output: harness, VBench, DINO |
| `04_results/` | `figures/` and `tables/` - the deliverables |
| `_cache/` | Model weights downloaded by VBench |

`STATUS.md` is regenerated by the notebooks and shows what has finished.

Nothing here is written non-atomically, so a runtime that dies mid-write leaves
the previous version intact rather than a truncated file.
""")
        return path

    def write_status(self) -> Path:
        """Regenerate STATUS.md so Drive itself shows the current state."""
        lines = ["# Status", "", f"Updated {utcnow()}", ""]
        for run_set in RUN_SETS:
            rows = self.inventory(run_set)
            if not rows:
                continue
            done = sum(r["verified"] for r in rows)
            lines += [f"## `{run_set}` — {done} of {len(rows)} verified", "",
                      "| Run | Verified | Status | Wall (s) | GPU | Memory path | Attention |",
                      "| --- | :---: | --- | ---: | --- | --- | --- |"]
            for row in rows:
                wall = f"{row['wall_seconds']:.0f}" if row["wall_seconds"] else "—"
                lines.append(
                    f"| `{row['run_id']}` | {'yes' if row['verified'] else '**no**'} "
                    f"| {row['status']} | {wall} | {row['gpu'] or '—'} "
                    f"| {row['memory_path'] or '—'} | {row['flash_attn'] or 'sdpa'} |")
            lines.append("")

        scores = sorted(self.scores.glob("*.json"))
        if scores:
            lines += ["## Cached scores", "", "| File | Entries |", "| --- | ---: |"]
            for path in scores:
                payload = read_json(path, {}) or {}
                lines.append(f"| `{path.name}` | {len(payload)} |")
            lines.append("")

        results = sorted(self.figures.glob("*.png")) + sorted(self.tables.glob("*"))
        if results:
            lines += ["## Results", ""] + [f"- `{p.relative_to(self.root)}`"
                                           for p in results]
        path = self.root / "STATUS.md"
        write_atomic(path, "\n".join(lines) + "\n")
        return path


class Checkpoint:
    """A resumable record of finished work units, kept on Drive.

    Each unit is stored under a key, so a stage that dies halfway resumes from
    the units it already completed instead of repeating them:

        ck = Checkpoint(ws.score("vbench-pilot"))
        for key, work in units:
            if ck.done(key):
                continue
            ck.put(key, work())        # saved immediately, atomically
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = read_json(self.path, {}) or {}

    def done(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def put(self, key: str, value) -> None:
        self.data[key] = value
        write_json(self.path, self.data)      # flush now; the session may end

    def update(self, mapping: dict) -> None:
        self.data.update(mapping)
        write_json(self.path, self.data)

    def drop(self, key: str) -> None:
        if key in self.data:
            del self.data[key]
            write_json(self.path, self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, key) -> bool:
        return key in self.data
