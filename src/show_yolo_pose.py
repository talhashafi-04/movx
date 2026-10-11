"""Draw saved pose estimates on a clip and write an mp4 you can play slowly.

Run from the FYP folder:
    python src/show_pose.py p1/20261008_120913_trim.mp4
    python src/show_pose.py 120913 --start 100 --end 170
    python src/show_pose.py 120913 --vit data/vitpose_plus_base

Green is YOLO26x. Orange is ViTPose++ (only if --vit is given).
Red dots are points the model was not sure about.
Frame numbers start at 1, same as the event labeler.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from event_labels import DEFAULT_LABELS, get_events, read_labels

ROOT = Path(__file__).resolve().parent.parent

# COCO 17 keypoints, pairs that get a line between them
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
ARM_POINTS = (7, 8, 9, 10)  # elbows and wrists get bigger dots

GREEN = (0, 220, 0)
ORANGE = (0, 140, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
MIN_CONF = 0.3


def find_clip(name, trimmed):
    """Accept a full path, a path inside trimmed/, or a part of the file name."""
    for path in (Path(name), trimmed / name):
        if path.is_file():
            return path
    matches = sorted(trimmed.glob(f"p*/*{name}*.mp4"))
    if len(matches) != 1:
        raise SystemExit(f"'{name}' matches {len(matches)} clips, be more specific")
    return matches[0]


def draw_pose(frame, xy, conf, color, scale, thickness):
    pts = xy * scale
    for a, b in EDGES:
        ok = conf[a] >= MIN_CONF and conf[b] >= MIN_CONF
        if ok and not np.isnan(pts[[a, b]]).any():
            cv2.line(frame, tuple(int(v) for v in pts[a]),
                     tuple(int(v) for v in pts[b]), color, thickness)
    for i in range(17):
        if np.isnan(pts[i]).any():
            continue
        radius = 8 if i in ARM_POINTS else 4
        dot = color if conf[i] >= MIN_CONF else RED
        cv2.circle(frame, (int(pts[i][0]), int(pts[i][1])), radius, dot, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", help="path, p1/..._trim.mp4, or part of the file name")
    ap.add_argument("--start", type=int, default=1, help="first frame to draw")
    ap.add_argument("--end", type=int, default=None, help="last frame to draw")
    ap.add_argument("--fps", type=float, default=30, help="playback speed of the output")
    ap.add_argument("--width", type=int, default=1280, help="output width in pixels")
    ap.add_argument("--vit", default=None, help="folder with ViTPose .npz files")
    ap.add_argument("--trimmed", default=str(ROOT / "trimmed"))
    ap.add_argument("--yolo", default=str(ROOT / "data" / "yolo26x"))
    ap.add_argument("--events", default=str(DEFAULT_LABELS), help="labels.csv with AH / BR")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    clip = find_clip(args.clip, Path(args.trimmed))
    player = clip.parent.name
    npz_name = f"{player}_{clip.stem}.npz"

    yolo_file = Path(args.yolo) / npz_name
    if not yolo_file.exists():
        raise SystemExit(f"no YOLO file: {yolo_file}")
    yolo = np.load(yolo_file)["kpts"][:, 0]  # (frames, 17, 3), slot 0 = biggest person

    vit_xy = vit_conf = None
    if args.vit:
        vit_file = Path(args.vit) / npz_name
        if not vit_file.exists():
            raise SystemExit(f"no ViTPose file: {vit_file}")
        data = np.load(vit_file)
        vit_xy, vit_conf = data["kpts"], data["conf"]

    events = get_events(read_labels(args.events)[0], f"{player}/{clip.name}")

    cap = cv2.VideoCapture(str(clip))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = args.width / width
    out_size = (args.width, int(round(height * scale)))
    last = min(args.end or total, total)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pose_{player}_{clip.stem.replace('_trim', '')}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, out_size)

    frame_no = 0
    written = 0
    while frame_no < last:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if frame_no < args.start:
            continue

        frame = cv2.resize(frame, out_size)
        t = frame_no - 1
        if t < len(yolo):
            draw_pose(frame, yolo[t, :, :2], yolo[t, :, 2], GREEN, scale, 3)
        if vit_xy is not None and t < len(vit_xy):
            draw_pose(frame, vit_xy[t], vit_conf[t], ORANGE, scale, 2)

        cv2.putText(frame, f"frame {frame_no}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2)
        legend = "green = YOLO26x" + ("   orange = ViTPose++" if vit_xy is not None else "")
        cv2.putText(frame, legend, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        if events:
            ah, br = events
            tag = "AH" if frame_no == ah else "BR" if frame_no == br else None
            if tag:
                cv2.putText(frame, tag, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.8, YELLOW, 4)
            elif ah < frame_no < br:
                cv2.putText(frame, "AH to BR", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, YELLOW, 2)

        writer.write(frame)
        written += 1

    cap.release()
    writer.release()
    print(f"wrote {written} frames to {out_path}")
    if events:
        print(f"AH {events[0]}, BR {events[1]} (from {Path(args.events).name})")


if __name__ == "__main__":
    main()
