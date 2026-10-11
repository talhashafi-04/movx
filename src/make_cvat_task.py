"""Prepare clips for CVAT: frame images plus pre-labels for the 3 arm points.

Run from the FYP folder, one clip at a time:
    python src/make_cvat_task.py 120913 --start 119 --end 147
    python src/make_cvat_task.py 120913 --start 119 --end 147 --step 2 --arm right

Or a whole player at once, using the AH/BR events in data/labels.csv
(start/end default to AH-2 and BR+2, same margin used for the first clip):
    python src/make_cvat_task.py --player p1

Makes outputs/cvat/<player>_<stamp>/
    images/                               PNG frames at the original resolution
    annotations/person_keypoints_default.json   pre-labels in COCO Keypoints format
    annotations.zip                       the same json zipped, for CVAT upload

The zip entry must be named person_keypoints_default.json, not
instances_default.json - CVAT's "COCO Keypoints 1.0" importer picks the
format by that exact filename and otherwise rejects the upload.

Pre-label = average of YOLO26x and ViTPose++ for shoulder, elbow, wrist
(if only one model has the point, that one is used). Pass --source vitpose
or --source yolo to use just one model instead of averaging.
Frame numbers start at 1, same as the event labeler.
"""
import argparse
import csv
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# COCO indices: right = shoulder 6, elbow 8, wrist 10. left = 5, 7, 9
ARMS = {"right": (6, 8, 10), "left": (5, 7, 9)}
NAMES = ["shoulder", "elbow", "wrist"]


def find_clip(name, trimmed):
    for path in (Path(name), trimmed / name):
        if path.is_file():
            return path
    matches = sorted(trimmed.glob(f"p*/*{name}*.mp4"))
    if len(matches) != 1:
        raise SystemExit(f"'{name}' matches {len(matches)} clips, be more specific")
    return matches[0]


def player_clips(player, labels_csv):
    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["player"] == player:
                yield row["clip"], int(row["ah_1based"]), int(row["br_1based"])


def make_task(clip, start, end, step, arm, vit_dir, yolo_dir, out_dir, source="avg"):
    player = clip.parent.name
    stamp = clip.stem.replace("_trim", "")
    npz_name = f"{player}_{clip.stem}.npz"
    yolo = np.load(Path(yolo_dir) / npz_name)["kpts"][:, 0, :, :2]  # (T, 17, 2)
    vit = np.load(Path(vit_dir) / npz_name)["kpts"]                  # (T, 17, 2)
    idx = list(ARMS[arm])

    task = Path(out_dir) / f"{player}_{stamp}"
    (task / "images").mkdir(parents=True, exist_ok=True)
    (task / "annotations").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(clip))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    images, annotations = [], []
    frames = range(start, end + 1, step)
    for n, frame_no in enumerate(frames, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
        ok, img = cap.read()
        if not ok:
            raise SystemExit(f"cannot read frame {frame_no}")
        name = f"{player}_{stamp}_f{frame_no:04d}.png"
        cv2.imwrite(str(task / "images" / name), img)  # PNG, lossless
        images.append({"id": n, "file_name": name, "width": width, "height": height})

        if source == "vitpose":
            pts = vit[frame_no - 1, idx]
        elif source == "yolo":
            pts = yolo[frame_no - 1, idx]
        else:
            pts = np.nanmean(np.stack([yolo[frame_no - 1, idx], vit[frame_no - 1, idx]]), axis=0)
        if np.isnan(pts).any():
            continue  # no estimate here, label this frame by hand
        kp = []
        for x, y in pts:
            kp += [round(float(x), 2), round(float(y), 2), 2]
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        annotations.append({
            "id": n, "image_id": n, "category_id": 1, "iscrowd": 0,
            "keypoints": kp, "num_keypoints": 3,
            "bbox": [round(float(x0), 2), round(float(y0), 2),
                     round(float(x1 - x0), 2), round(float(y1 - y0), 2)],
            "area": round(float((x1 - x0) * (y1 - y0)), 2),
        })
    cap.release()

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "arm", "supercategory": "arm",
                        "keypoints": NAMES, "skeleton": [[1, 2], [2, 3]]}],
    }
    json_path = task / "annotations" / "person_keypoints_default.json"
    json_path.write_text(json.dumps(coco))
    with zipfile.ZipFile(task / "annotations.zip", "w") as z:
        z.write(json_path, "annotations/person_keypoints_default.json")

    print(f"{len(images)} frames, {len(annotations)} pre-labeled, saved in {task}")
    print(f"arm used: {arm} (COCO points {idx}). Check that it is the bowling arm.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", help="clip name/stamp, omit when using --player")
    ap.add_argument("--player", help="process every clip for this player using data/labels.csv")
    ap.add_argument("--labels", default=str(ROOT / "data" / "labels.csv"))
    ap.add_argument("--margin", type=int, default=2, help="frames before AH / after BR, --player mode only")
    ap.add_argument("--start", type=int, help="first frame (1-based)")
    ap.add_argument("--end", type=int, help="last frame (1-based)")
    ap.add_argument("--step", type=int, default=1, help="take every Nth frame")
    ap.add_argument("--arm", choices=ARMS, default="right", help="bowling arm of the bowler")
    ap.add_argument("--source", choices=["avg", "vitpose", "yolo"], default="avg",
                     help="pre-label source: average of both models, or just one")
    ap.add_argument("--trimmed", default=str(ROOT / "trimmed"))
    ap.add_argument("--vit", default=str(ROOT / "data" / "vitpose_plus_base"))
    ap.add_argument("--yolo", default=str(ROOT / "data" / "yolo26x"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "cvat"))
    args = ap.parse_args()

    if args.player:
        clips = list(player_clips(args.player, args.labels))
        if not clips:
            raise SystemExit(f"no rows for player '{args.player}' in {args.labels}")
        for clip_rel, ah, br in clips:
            clip = find_clip(clip_rel, Path(args.trimmed))
            make_task(clip, ah - args.margin, br + args.margin, args.step, args.arm,
                       args.vit, args.yolo, args.out, args.source)
        return

    if args.clip is None or args.start is None or args.end is None:
        raise SystemExit("give a clip with --start/--end, or use --player")
    clip = find_clip(args.clip, Path(args.trimmed))
    make_task(clip, args.start, args.end, args.step, args.arm, args.vit, args.yolo, args.out, args.source)


if __name__ == "__main__":
    main()