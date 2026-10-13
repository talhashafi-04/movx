#!/usr/bin/env python3
"""Build a YOLO26 17-keypoint pose dataset from the movx CVAT labels.

The CVAT work corrects only the bowling-arm shoulder, elbow, and wrist. The
application, however, consumes the standard 17-point COCO pose output. This
builder therefore keeps teacher-generated full-person boxes and the other 14
keypoints, then overwrites the bowling-arm points with the human corrections.

Examples (run from the repository root):

    python src/build_yolo_pose_dataset.py --audit-only
    python src/build_yolo_pose_dataset.py --output outputs/yolo_pose_event --overwrite
    python src/build_yolo_pose_dataset.py \
        --output outputs/yolo_pose_nano --context-stride 5 --overwrite

The optional context frames are added only for training participants and are
fully pseudo-labelled. They are useful for YOLO26n, which runs over complete
clips in the first pass. Validation and test always contain human-corrected
event-window frames only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
ARM_INDICES = {"right": (6, 8, 10), "left": (5, 7, 9)}
ARM_NAMES = ("shoulder", "elbow", "wrist")
FRAME_RE = re.compile(r"_f(\d+)\.(?:png|jpe?g)$", re.IGNORECASE)


@dataclass(frozen=True)
class ClipRecord:
    clip_rel: str
    player: str
    split: str
    ah: int
    br: int
    span: int

    @property
    def clip_stem(self) -> str:
        return Path(self.clip_rel).stem

    @property
    def stamp(self) -> str:
        return self.clip_stem.removesuffix("_trim")

    @property
    def task_name(self) -> str:
        return f"{self.player}_{self.stamp}"


@dataclass(frozen=True)
class HumanFrame:
    frame_1based: int
    points: np.ndarray
    image_width: int
    image_height: int
    annotation_source: str


def parse_args() -> argparse.Namespace:
    repo_default = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Build the movx hybrid 17-keypoint YOLO pose dataset."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_default)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/yolo_pose_event"),
        help="output directory, relative paths are resolved from --repo-root",
    )
    parser.add_argument(
        "--teacher",
        choices=("average", "yolo", "vitpose"),
        default="average",
        help="source for the 14 non-human-corrected keypoints",
    )
    parser.add_argument(
        "--pseudo-confidence",
        type=float,
        default=0.25,
        help="minimum teacher confidence for a pseudo-keypoint to be visible",
    )
    parser.add_argument(
        "--arm",
        choices=tuple(ARM_INDICES),
        default="right",
        help="anatomical bowling arm represented by the CVAT labels",
    )
    parser.add_argument(
        "--context-stride",
        type=int,
        default=0,
        help="add every Nth non-event frame to train only; 0 disables context",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--players",
        nargs="+",
        help="optional participant subset, e.g. --players p1 p2",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="validate source coverage and exit without extracting frames",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output directory",
    )
    return parser.parse_args()


def read_records(labels_path: Path, players: set[str] | None) -> list[ClipRecord]:
    records: list[ClipRecord] = []
    with labels_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if players and row["player"] not in players:
                continue
            records.append(
                ClipRecord(
                    clip_rel=row["clip"],
                    player=row["player"],
                    split=row["split"],
                    ah=int(row["ah_1based"]),
                    br=int(row["br_1based"]),
                    span=int(row["span"]),
                )
            )
    if not records:
        raise SystemExit("no clips matched the requested participants")
    return records


def load_coco(path: Path) -> dict[str, Any]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".json")]
            if len(names) != 1:
                raise ValueError(f"{path}: expected one JSON member, found {names}")
            return json.loads(archive.read(names[0]))
    return json.loads(path.read_text())


def annotation_source(task_dir: Path) -> Path:
    corrected = sorted(task_dir.glob("*_corrected.zip"))
    if len(corrected) > 1:
        raise ValueError(f"{task_dir}: multiple corrected exports: {corrected}")
    if corrected:
        return corrected[0]
    generic = task_dir / "annotations" / "person_keypoints_default.json"
    if not generic.is_file():
        raise FileNotFoundError(f"{task_dir}: no corrected ZIP or annotation JSON")
    return generic


def load_human_frames(record: ClipRecord, cvat_root: Path) -> list[HumanFrame]:
    source = annotation_source(cvat_root / record.task_name)
    coco = load_coco(source)
    categories = coco.get("categories", [])
    if len(categories) != 1 or categories[0].get("keypoints") != list(ARM_NAMES):
        raise ValueError(f"{source}: expected shoulder/elbow/wrist keypoints")

    images = {int(image["id"]): image for image in coco.get("images", [])}
    by_image: dict[int, dict[str, Any]] = {}
    for annotation in coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id in by_image:
            raise ValueError(f"{source}: multiple annotations for image {image_id}")
        by_image[image_id] = annotation

    if set(images) != set(by_image):
        missing = sorted(set(images) - set(by_image))
        extra = sorted(set(by_image) - set(images))
        raise ValueError(f"{source}: image/annotation mismatch, missing={missing}, extra={extra}")

    expected_count = record.span + 5  # AH-2 through BR+2, inclusive
    if len(images) != expected_count:
        raise ValueError(
            f"{source}: expected {expected_count} frames from labels.csv, found {len(images)}"
        )

    frames: list[HumanFrame] = []
    for image_id, image in images.items():
        match = FRAME_RE.search(image["file_name"])
        if not match:
            raise ValueError(f"{source}: cannot read frame number from {image['file_name']}")
        raw = by_image[image_id].get("keypoints", [])
        if len(raw) != 9:
            raise ValueError(f"{source}: image {image_id} does not have three keypoints")
        keypoints = np.asarray(raw, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(keypoints[:, :2]).all() or np.any(keypoints[:, 2] <= 0):
            raise ValueError(f"{source}: image {image_id} has an invisible/invalid human point")
        frames.append(
            HumanFrame(
                frame_1based=int(match.group(1)),
                points=keypoints[:, :2],
                image_width=int(image["width"]),
                image_height=int(image["height"]),
                annotation_source=str(source.relative_to(cvat_root.parent.parent)),
            )
        )
    frames = sorted(frames, key=lambda item: item.frame_1based)
    expected_numbers = set(range(record.ah - 2, record.br + 3))
    actual_numbers = {item.frame_1based for item in frames}
    if actual_numbers != expected_numbers:
        raise ValueError(
            f"{source}: frame range does not match AH-2 through BR+2; "
            f"missing={sorted(expected_numbers - actual_numbers)}, "
            f"extra={sorted(actual_numbers - expected_numbers)}"
        )
    return frames


def validate_sources(records: Iterable[ClipRecord], repo: Path) -> tuple[list[tuple[ClipRecord, list[HumanFrame]]], Counter]:
    loaded: list[tuple[ClipRecord, list[HumanFrame]]] = []
    counts: Counter = Counter()
    for record in records:
        clip = repo / "trimmed" / record.clip_rel
        yolo = repo / "data" / "yolo26x" / f"{record.player}_{record.clip_stem}.npz"
        vit = repo / "data" / "vitpose_plus_base" / f"{record.player}_{record.clip_stem}.npz"
        for path in (clip, yolo, vit):
            if not path.is_file():
                raise FileNotFoundError(path)
        human = load_human_frames(record, repo / "outputs" / "cvat")
        loaded.append((record, human))
        counts[record.split] += len(human)
        source_kind = "corrected_zip" if human[0].annotation_source.endswith(".zip") else "json"
        counts[source_kind] += 1
    return loaded, counts


def fuse_teacher(
    yolo: np.ndarray,
    vit_xy: np.ndarray,
    vit_conf: np.ndarray,
    source: str,
) -> tuple[np.ndarray, np.ndarray]:
    yolo_xy = yolo[:, :2].astype(np.float64)
    yolo_conf = yolo[:, 2].astype(np.float64)
    vit_xy = vit_xy.astype(np.float64)
    vit_conf = vit_conf.astype(np.float64)

    if source == "yolo":
        return yolo_xy.copy(), yolo_conf.copy()
    if source == "vitpose":
        return vit_xy.copy(), vit_conf.copy()

    xy = np.full((17, 2), np.nan, dtype=np.float64)
    confidence = np.zeros(17, dtype=np.float64)
    for index in range(17):
        candidates: list[np.ndarray] = []
        scores: list[float] = []
        if np.isfinite(yolo_xy[index]).all() and math.isfinite(yolo_conf[index]):
            candidates.append(yolo_xy[index])
            scores.append(max(0.0, float(yolo_conf[index])))
        if np.isfinite(vit_xy[index]).all() and math.isfinite(vit_conf[index]):
            candidates.append(vit_xy[index])
            scores.append(max(0.0, float(vit_conf[index])))
        if candidates:
            weights = np.asarray(scores, dtype=np.float64)
            if weights.sum() <= 0:
                weights = np.ones_like(weights)
            xy[index] = np.average(np.stack(candidates), axis=0, weights=weights)
            confidence[index] = float(np.max(scores))
    return xy, confidence


def person_box(
    raw_box: np.ndarray,
    points: np.ndarray,
    visible: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    box = np.asarray(raw_box, dtype=np.float64).copy()
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        shown = points[visible > 0]
        if len(shown) < 2:
            raise ValueError("teacher has neither a valid person box nor enough visible keypoints")
        x0, y0 = shown.min(axis=0)
        x1, y1 = shown.max(axis=0)
        padding = 0.1 * max(x1 - x0, y1 - y0, 1.0)
        box = np.array([x0 - padding, y0 - padding, x1 + padding, y1 + padding])

    shown = points[visible > 0]
    if len(shown):
        margin = 0.01 * max(width, height)
        box[0] = min(box[0], float(shown[:, 0].min()) - margin)
        box[1] = min(box[1], float(shown[:, 1].min()) - margin)
        box[2] = max(box[2], float(shown[:, 0].max()) + margin)
        box[3] = max(box[3], float(shown[:, 1].max()) + margin)

    box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        raise ValueError(f"invalid person box after clipping: {box.tolist()}")
    return box


def yolo_line(
    points: np.ndarray,
    visible: np.ndarray,
    box: np.ndarray,
    width: int,
    height: int,
) -> str:
    x0, y0, x1, y1 = box
    values: list[float | int] = [
        0,
        ((x0 + x1) / 2) / width,
        ((y0 + y1) / 2) / height,
        (x1 - x0) / width,
        (y1 - y0) / height,
    ]
    for point, state in zip(points, visible):
        if state:
            x = float(np.clip(point[0] / width, 0.0, 1.0))
            y = float(np.clip(point[1] / height, 0.0, 1.0))
            values.extend((x, y, int(state)))
        else:
            values.extend((0.0, 0.0, 0))
    if len(values) != 56:
        raise AssertionError(f"expected 56 YOLO fields, got {len(values)}")
    return " ".join(str(value) if isinstance(value, int) else f"{value:.8f}" for value in values)


def safe_prepare_output(output: Path, repo: Path, overwrite: bool) -> None:
    resolved = output.resolve()
    forbidden = {Path("/").resolve(), repo.resolve(), repo.resolve().parent}
    if resolved in forbidden:
        raise ValueError(f"refusing to use broad output directory: {resolved}")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def save_frame(cap: cv2.VideoCapture, frame_index: int, path: Path, quality: int) -> tuple[int, int]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_index + 1} for {path.name}")
    height, width = frame.shape[:2]
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise RuntimeError(f"cannot write {path}")
    return width, height


def build(args: argparse.Namespace) -> None:
    repo = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else repo / args.output
    labels_path = repo / "data" / "labels.csv"
    players = set(args.players) if args.players else None
    records = read_records(labels_path, players)
    loaded, audit_counts = validate_sources(records, repo)

    print(
        f"audit passed: {len(loaded)} clips, "
        f"{sum(audit_counts[s] for s in ('train', 'val', 'test'))} human-labelled frames "
        f"({audit_counts['corrected_zip']} corrected ZIP tasks, {audit_counts['json']} JSON tasks)"
    )
    print(
        f"split: train={audit_counts['train']}, val={audit_counts['val']}, "
        f"test={audit_counts['test']}"
    )
    if args.audit_only:
        return

    if not 0 <= args.pseudo_confidence <= 1:
        raise ValueError("--pseudo-confidence must be between 0 and 1")
    if args.context_stride < 0:
        raise ValueError("--context-stride cannot be negative")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    safe_prepare_output(output, repo, args.overwrite)
    manifest_path = output / "manifest.csv"
    manifest_rows: list[dict[str, Any]] = []
    split_counts: Counter = Counter()
    origin_counts: Counter = Counter()
    arm_indices = ARM_INDICES[args.arm]

    for clip_number, (record, human_frames) in enumerate(loaded, 1):
        clip_path = repo / "trimmed" / record.clip_rel
        yolo_path = repo / "data" / "yolo26x" / f"{record.player}_{record.clip_stem}.npz"
        vit_path = repo / "data" / "vitpose_plus_base" / f"{record.player}_{record.clip_stem}.npz"
        yolo_data = np.load(yolo_path)
        vit_data = np.load(vit_path)
        yolo_kpts = yolo_data["kpts"][:, 0]
        yolo_boxes = yolo_data["boxes"][:, 0]
        vit_kpts = vit_data["kpts"]
        vit_conf = vit_data["conf"]
        available = min(len(yolo_kpts), len(yolo_boxes), len(vit_kpts), len(vit_conf))

        human_by_index = {item.frame_1based - 1: item for item in human_frames}
        selected: list[tuple[int, HumanFrame | None, str]] = [
            (index, item, "human_hybrid") for index, item in human_by_index.items()
        ]
        if args.context_stride and record.split == "train":
            for index in range(0, available, args.context_stride):
                if index not in human_by_index:
                    selected.append((index, None, "pseudo_context"))
        selected.sort(key=lambda item: item[0])

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {clip_path}")
        try:
            for frame_index, human, origin in selected:
                if frame_index < 0 or frame_index >= available:
                    raise IndexError(
                        f"{record.task_name}: frame {frame_index + 1} outside teacher arrays ({available})"
                    )
                stem = f"{record.task_name}_f{frame_index + 1:04d}"
                image_path = output / "images" / record.split / f"{stem}.jpg"
                label_path = output / "labels" / record.split / f"{stem}.txt"
                width, height = save_frame(cap, frame_index, image_path, args.jpeg_quality)
                if human and (human.image_width != width or human.image_height != height):
                    raise ValueError(
                        f"{record.task_name}: frame dimensions {width}x{height} do not match "
                        f"annotation {human.image_width}x{human.image_height}"
                    )

                points, confidence = fuse_teacher(
                    yolo_kpts[frame_index],
                    vit_kpts[frame_index],
                    vit_conf[frame_index],
                    args.teacher,
                )
                visible = np.where(
                    np.isfinite(points).all(axis=1) & (confidence >= args.pseudo_confidence),
                    2,
                    0,
                ).astype(np.int64)
                if human is not None:
                    points[list(arm_indices)] = human.points
                    visible[list(arm_indices)] = 2
                box = person_box(yolo_boxes[frame_index], points, visible, width, height)
                label_path.write_text(yolo_line(points, visible, box, width, height) + "\n")

                manifest_rows.append(
                    {
                        "image": str(image_path.relative_to(output)),
                        "label": str(label_path.relative_to(output)),
                        "split": record.split,
                        "player": record.player,
                        "clip": record.clip_rel,
                        "frame_1based": frame_index + 1,
                        "origin": origin,
                        "annotation_source": human.annotation_source if human else "teacher",
                    }
                )
                split_counts[record.split] += 1
                origin_counts[origin] += 1
        finally:
            cap.release()
        print(
            f"[{clip_number}/{len(loaded)}] {record.task_name}: "
            f"{len(human_frames)} human + {len(selected) - len(human_frames)} context"
        )

    fieldnames = [
        "image",
        "label",
        "split",
        "player",
        "clip",
        "frame_1based",
        "origin",
        "annotation_source",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    yaml_text = "\n".join(
        [
            f"path: {json.dumps(str(output.resolve()))}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "kpt_shape: [17, 3]",
            f"flip_idx: {FLIP_IDX}",
            "names:",
            "  0: person",
            "",
        ]
    )
    (output / "dataset.yaml").write_text(yaml_text)
    summary = {
        "clips": len(loaded),
        "images": len(manifest_rows),
        "split_images": dict(sorted(split_counts.items())),
        "origins": dict(sorted(origin_counts.items())),
        "source_tasks": {
            "corrected_zip": audit_counts["corrected_zip"],
            "json": audit_counts["json"],
        },
        "teacher": args.teacher,
        "pseudo_confidence": args.pseudo_confidence,
        "arm": args.arm,
        "arm_indices": list(arm_indices),
        "context_stride": args.context_stride,
        "kpt_shape": [17, 3],
        "keypoint_names": COCO_KEYPOINTS,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"built {len(manifest_rows)} images in {output}")
    print(f"dataset YAML: {output / 'dataset.yaml'}")


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
