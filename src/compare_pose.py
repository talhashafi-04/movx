"""Make a full-resolution picture comparing YOLO26x and ViTPose++ on a few frames.

Run from the FYP folder:
    python src/compare_pose.py 145239
    python src/compare_pose.py 145239 --frames 133 137 141 145
    python src/compare_pose.py 145239 --frames 121 130 140 145 --pad 60

Top row is YOLO26x (green), bottom row is ViTPose++ (orange).
Nothing is shrunk or enlarged. Every tile is a crop of the original clip pixels,
saved as a lossless PNG. Frame numbers start at 1, same as the event labeler.
Only the arm points (shoulders, elbows, wrists) are drawn.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

ARM_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6)]
ARM_POINTS = (5, 6, 7, 8, 9, 10)
GREEN = (0, 220, 0)
ORANGE = (0, 140, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
MIN_CONF = 0.3


def find_clip(name, trimmed):
    for path in (Path(name), trimmed / name):
        if path.is_file():
            return path
    matches = sorted(trimmed.glob(f"p*/*{name}*.mp4"))
    if len(matches) != 1:
        raise SystemExit(f"'{name}' matches {len(matches)} clips, be more specific")
    return matches[0]


def draw_arm(tile, xy, conf, color, x0, y0):
    pts = xy - np.array([x0, y0])
    for a, b in ARM_EDGES:
        if conf[a] >= MIN_CONF and conf[b] >= MIN_CONF and not np.isnan(pts[[a, b]]).any():
            cv2.line(tile, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]),
                     color, 2, cv2.LINE_AA)
    for i in ARM_POINTS:
        if np.isnan(pts[i]).any():
            continue
        dot = color if conf[i] >= MIN_CONF else RED
        cv2.circle(tile, (int(pts[i][0]), int(pts[i][1])), 5, dot, -1, cv2.LINE_AA)


def label(tile, text):
    cv2.putText(tile, text, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(tile, text, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", help="path, p1/..._trim.mp4, or part of the file name")
    ap.add_argument("--frames", type=int, nargs="+", default=[133, 137, 141, 145])
    ap.add_argument("--pad", type=int, default=80, help="extra pixels around the arm")
    ap.add_argument("--trimmed", default=str(ROOT / "trimmed"))
    ap.add_argument("--vit", default=str(ROOT / "data" / "vitpose_plus_base"))
    ap.add_argument("--yolo", default=str(ROOT / "data" / "yolo26x"))
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    clip = find_clip(args.clip, Path(args.trimmed))
    player = clip.parent.name
    npz_name = f"{player}_{clip.stem}.npz"
    yolo = np.load(Path(args.yolo) / npz_name)["kpts"][:, 0]  # (T, 17, 3)
    vit = np.load(Path(args.vit) / npz_name)
    vit_xy, vit_conf = vit["kpts"], vit["conf"]

    cap = cv2.VideoCapture(str(clip))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # one crop box for every tile, around the arm points of both models in all chosen frames
    idx = [f - 1 for f in args.frames]
    pts = np.concatenate([yolo[idx][:, list(ARM_POINTS), :2].reshape(-1, 2),
                          vit_xy[idx][:, list(ARM_POINTS)].reshape(-1, 2)])
    pts = pts[~np.isnan(pts).any(axis=1)]
    x0 = max(int(pts[:, 0].min()) - args.pad, 0)
    y0 = max(int(pts[:, 1].min()) - args.pad, 0)
    x1 = min(int(pts[:, 0].max()) + args.pad, width)
    y1 = min(int(pts[:, 1].max()) + args.pad, height)

    top, bottom = [], []
    for n in args.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"cannot read frame {n}")
        a = frame[y0:y1, x0:x1].copy()
        b = a.copy()
        draw_arm(a, yolo[n - 1, :, :2], yolo[n - 1, :, 2], GREEN, x0, y0)
        draw_arm(b, vit_xy[n - 1], vit_conf[n - 1], ORANGE, x0, y0)
        label(a, f"YOLO26x  f{n}")
        label(b, f"ViTPose++  f{n}")
        top.append(a)
        bottom.append(b)
    cap.release()

    img = np.vstack([np.hstack(top), np.hstack(bottom)])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"compare_{player}_{clip.stem.replace('_trim', '')}.png"
    cv2.imwrite(str(out), img)  # PNG is lossless
    print(f"wrote {out}  {img.shape[1]}x{img.shape[0]} px, tile {x1 - x0}x{y1 - y0} "
          f"cropped from the {width}x{height} clip, no resizing")


if __name__ == "__main__":
    main()
