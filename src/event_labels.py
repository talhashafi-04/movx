"""Read and write the AH / BR event labels in data/labels.csv.

labels.csv has one row per clip:
    clip, player, split, ah_1based, br_1based, span, trim_start_frame, frame_count
AH and BR are 1-based frame numbers inside the trimmed clip. A row with empty
AH / BR is a clip that was looked at and skipped.

The older events.csv layout (clip, ah_frame, br_frame, status) is still read,
so an old file passed with --events / --csv keeps working.
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = ROOT / "data" / "labels.csv"
FIELDS = ["clip", "player", "split", "ah_1based", "br_1based", "span",
          "trim_start_frame", "frame_count"]


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def _from_events_row(r):
    """Turn an old events.csv row into a labels.csv row."""
    ok = r.get("status", "ok") == "ok" and r["ah_frame"] and r["br_frame"]
    ah, br = (int(r["ah_frame"]), int(r["br_frame"])) if ok else ("", "")
    return {"clip": r["clip"], "player": r["clip"].split("/")[0],
            "ah_1based": ah, "br_1based": br, "span": br - ah if ok else ""}


def read_labels(path):
    """Return ({clip: row}, fieldnames). Missing file gives ({}, FIELDS)."""
    path = Path(path)
    if not path.exists():
        return {}, list(FIELDS)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or list(FIELDS)
    if "ah_frame" in fields:
        return {r["clip"]: _from_events_row(r) for r in rows}, list(FIELDS)
    return {r["clip"]: r for r in rows}, fields


def write_labels(path, rows, fields=FIELDS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for k in sorted(rows, key=natural_key):
            w.writerow(rows[k])


def get_events(rows, clip):
    """Return (ah, br) for a clip key like 'p1/20261008_120553_trim.mp4', else None."""
    r = rows.get(clip)
    if not r or not r.get("ah_1based") or not r.get("br_1based"):
        return None
    return int(r["ah_1based"]), int(r["br_1based"])


def set_events(rows, clip, ah, br, frame_count=None):
    """Store AH / BR (or None, None for a skipped clip), keeping the row's other columns."""
    r = rows.setdefault(clip, {"clip": clip, "player": clip.split("/")[0]})
    r["ah_1based"] = ah or ""
    r["br_1based"] = br or ""
    r["span"] = br - ah if ah and br else ""
    if frame_count and not r.get("frame_count"):
        r["frame_count"] = frame_count
