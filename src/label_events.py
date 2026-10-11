#!/usr/bin/env python3
"""
Frame-by-frame AH / BR event labeler.

Place this file in FYP/src/. It finds FYP/trimmed/ by itself (relative to the
script, so it works from any directory) and writes FYP/data/labels.csv.

    python3 src/label_events.py                # label everything not yet done
    python3 src/label_events.py --only p6 p10  # only these players
    python3 src/label_events.py --redo         # revisit already-labeled clips

Frame numbers are 1-based, matching ffmpeg's %04d.jpg extraction.
Output: data/labels.csv  (clip, player, split, ah_1based, br_1based, span,
trim_start_frame, frame_count). Existing columns such as split are kept; a
skipped clip gets empty AH / BR.

Keys
    d / a      next / previous frame
    e / q      +10 / -10 frames
    c / z      +1 / -1 frame, slow-step alias (same as d / a)
    h          mark AH (arm horizontal) on current frame
    b          mark BR (ball release) on current frame
    r          clear both marks for this clip
    g          jump to marked AH, then marked BR (press again to toggle)
    ENTER / n  save and go to next clip (needs AH < BR)
    p          go back to previous clip
    x          mark clip as bad/skip and go to next
    ESC        save progress and quit
"""
import argparse, sys
from pathlib import Path
import cv2

from event_labels import DEFAULT_LABELS, get_events, natural_key, read_labels, set_events, write_labels

KEYS_HELP = "a/d:+-1  q/e:+-10  h:AH  b:BR  g:jump  r:clear  ENTER:save+next  p:prev  x:skip  ESC:quit"


HERE = Path(__file__).resolve().parent          # FYP/src
DEFAULT_ROOT = HERE.parent / "trimmed"          # FYP/trimmed
DEFAULT_CSV = DEFAULT_LABELS                    # FYP/data/labels.csv


def load_frames(path, width):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        if width and w > width:
            f = cv2.resize(f, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        # keep frames JPEG-compressed in RAM (~100-150 KB each instead of ~2.8 MB raw)
        ok2, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 92])
        frames.append(buf)
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--only", nargs="*", help="player folders, e.g. p6 p10")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--width", type=int, default=1280, help="display width")
    a = ap.parse_args()

    root, csv_path = Path(a.root), Path(a.csv)
    players = a.only or [d.name for d in root.iterdir() if d.is_dir() and d.name.startswith("p")]
    clips = sorted((f"{p}/{f.name}" for p in players for f in (root / p).glob("*.mp4")), key=natural_key)
    if not clips:
        sys.exit(f"No clips found under {root}. Pass --root /path/to/trimmed.")

    rows, fields = read_labels(csv_path)
    todo = [c for c in clips if a.redo or c not in rows]
    if not todo:
        sys.exit(f"All {len(clips)} clips already labeled in {csv_path}. Use --redo to revisit.")

    win = "label_events"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    i = 0
    while 0 <= i < len(todo):
        clip = todo[i]
        frames = load_frames(root / clip, a.width)
        n = len(frames)
        if n == 0:
            print(f"[warn] could not read {clip}, skipping")
            i += 1
            continue
        prev = rows.get(clip)
        ah, br = get_events(rows, clip) or (None, None)
        cur = (ah or int(n * 0.40)) - 1   # 0-based index; start near the delivery
        cur = max(0, min(n - 1, cur))
        jump_to_br = False
        msg = ""

        while True:
            img = cv2.imdecode(frames[cur], cv2.IMREAD_COLOR)
            fh, fw = img.shape[:2]
            cv2.rectangle(img, (0, 0), (fw, 64), (0, 0, 0), -1)
            tag = ""
            if ah == cur + 1: tag += "  << AH"
            if br == cur + 1: tag += "  << BR"
            cv2.putText(img, f"{clip}   [{len(rows) + 0}/{len(clips)} done, {i + 1}/{len(todo)} in queue]",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            diff = f"  AH->BR = {br - ah}" if ah and br else ""
            cv2.putText(img, f"frame {cur + 1}/{n}   AH={ah}  BR={br}{diff}{tag}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(img, (0, fh - 28), (fw, fh), (0, 0, 0), -1)
            cv2.putText(img, msg or KEYS_HELP, (10, fh - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1, cv2.LINE_AA)
            if ah == cur + 1 or br == cur + 1:
                cv2.rectangle(img, (0, 0), (fw - 1, fh - 1), (0, 255, 0), 4)
            cv2.imshow(win, img)

            k = cv2.waitKey(0) & 0xFF
            msg = ""
            if k in (ord("d"), ord("c")): cur = min(n - 1, cur + 1)
            elif k in (ord("a"), ord("z")): cur = max(0, cur - 1)
            elif k == ord("e"): cur = min(n - 1, cur + 10)
            elif k == ord("q"): cur = max(0, cur - 10)
            elif k == ord("h"): ah = cur + 1; msg = f"AH set to {ah}"
            elif k == ord("b"): br = cur + 1; msg = f"BR set to {br}"
            elif k == ord("r"): ah = br = None; msg = "marks cleared"
            elif k == ord("g"):
                tgt = (br if jump_to_br else ah)
                if tgt: cur = tgt - 1
                jump_to_br = not jump_to_br
            elif k in (13, 10, ord("n")):
                if not (ah and br):
                    msg = "need both AH and BR before saving"
                elif br <= ah:
                    msg = f"BR ({br}) must be after AH ({ah})"
                else:
                    set_events(rows, clip, ah, br, frame_count=n)
                    write_labels(csv_path, rows, fields)
                    print(f"{clip}: AH={ah} BR={br} ({br - ah} frames)")
                    i += 1
                    break
            elif k == ord("p"):
                i = max(0, i - 1)
                break
            elif k == ord("x"):
                set_events(rows, clip, None, None, frame_count=n)
                write_labels(csv_path, rows, fields)
                print(f"{clip}: skipped")
                i += 1
                break
            elif k == 27:
                write_labels(csv_path, rows, fields)
                cv2.destroyAllWindows()
                print(f"Saved {csv_path}. Quit.")
                return

    cv2.destroyAllWindows()
    write_labels(csv_path, rows, fields)
    ok = [e for e in (get_events(rows, c) for c in rows) if e]
    if ok:
        d = sorted(br - ah for ah, br in ok)
        print(f"Done. {len(ok)} clips labeled. AH->BR frames: min {d[0]}, median {d[len(d)//2]}, max {d[-1]}")


if __name__ == "__main__":
    main()
