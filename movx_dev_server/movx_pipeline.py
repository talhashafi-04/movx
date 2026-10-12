"""The movx V1 two-pass pipeline for the dev server. Mirrors the phone app (Analyzer.kt):

    pass 1: YOLO26n pose on every frame -> largest person -> TCN -> rough AH/BR
    pass 2: chosen pose model on AH - 60 .. BR + 60 -> TCN -> final AH/BR

Pose models are the same .tflite files the app ships, and the TCN reads the app's tcn.bin,
so the server and the phone give the same answers. Only numpy, OpenCV and LiteRT are needed.
"""
import json
import struct
import time
from pathlib import Path

import cv2
import numpy as np

CONF, IOU = 0.25, 0.7
MARGIN = 60
MIN_SCORE = 0.2
MIN_ARM_CONF = 0.4
SAVE_MAX = 1280
PASS1 = "yolo26n-pose.tflite"
L_SHOULDER, R_SHOULDER, R_ELBOW, R_WRIST, L_HIP, R_HIP = 5, 6, 8, 10, 11, 12
DILATIONS = (1, 2, 4, 8, 16)


class Rejected(Exception):
    """A clip we can't measure, with a reason and what to do about it."""

    def __init__(self, reason, fix):
        super().__init__(reason)
        self.reason, self.fix = reason, fix


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------- pose

class PoseModel:
    def __init__(self, path, threads=4):
        from ai_edge_litert.interpreter import Interpreter
        self.it = Interpreter(str(path), num_threads=threads)
        self.it.allocate_tensors()
        d = self.it.get_input_details()[0]
        self.size, self.inp = int(d["shape"][2]), d["index"]
        self.out = self.it.get_output_details()[0]["index"]

    def detect(self, rgb):
        """All people in an RGB frame: list of (area, kpts (17, 3)) in frame pixels."""
        S = self.size
        H0, W0 = rgb.shape[:2]
        s = min(S / W0, S / H0)
        w, h = round(W0 * s), round(H0 * s)
        x0, y0 = (S - w) // 2, (S - h) // 2
        box = np.full((S, S, 3), 114, np.uint8)
        box[y0:y0 + h, x0:x0 + w] = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        self.it.set_tensor(self.inp, (box.astype(np.float32) / 255).transpose(2, 0, 1)[None])
        self.it.invoke()
        return decode(self.it.get_tensor(self.out)[0], S, s, x0, y0)


def decode(o, S, s, x0, y0):
    """o: (56, N) raw output -> NMS'd people as (area, kpts)."""
    keep = np.flatnonzero(o[4] > CONF)
    if len(keep) == 0:
        return []
    v = o[:, keep]
    cx, cy, w, h = (v[0] * S - x0) / s, (v[1] * S - y0) / s, v[2] * S / s, v[3] * S / s
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    order = np.argsort(-v[4])
    out = []
    while len(order):
        i = order[0]
        out.append(i)
        a, b = xyxy[i], xyxy[order[1:]]
        iw = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
        ih = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
        inter = iw * ih
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
        order = order[1:][inter / union <= IOU]
    return [((xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1]),
             np.stack([(v[5::3, i] * S - x0) / s, (v[6::3, i] * S - y0) / s, v[7::3, i]], 1)) for i in out]


def bowler(people):
    k = np.full((17, 3), np.nan, np.float32)
    if people:
        k[:] = max(people, key=lambda p: p[0])[1]
    return k


# ---------------------------------------------------------------- TCN (numpy, reads the app's tcn.bin)

class Tcn:
    def __init__(self, path):
        b = Path(path).read_bytes()
        magic, n_models, self.n_in, ch = struct.unpack_from("<4i", b)
        assert magic == 0x4D4F5654, "not a tcn.bin file"
        a = np.frombuffer(b, "<f4", offset=16)
        pos = 0

        def read(*shape):
            nonlocal pos
            n = int(np.prod(shape))
            r = a[pos:pos + n].reshape(shape).astype(np.float64)
            pos += n
            return r
        self.models = []
        for _ in range(n_models):
            m = {"inp": (read(ch, self.n_in), read(ch)), "blocks": []}
            for _ in DILATIONS:
                m["blocks"].append((read(ch, ch, 3), read(ch), read(ch, ch), read(ch)))
            m["out"] = (read(2, ch), read(2))
            self.models.append(m)

    @staticmethod
    def _run(m, x):
        """x: (T, n_in) -> probabilities (2, T)."""
        T = len(x)
        h = m["inp"][0] @ x.T + m["inp"][1][:, None]
        for (w1, b1, w2, b2), d in zip(m["blocks"], DILATIONS):
            p = np.pad(h, ((0, 0), (d, d)))
            y = w1[:, :, 0] @ p[:, :T] + w1[:, :, 1] @ p[:, d:d + T] + w1[:, :, 2] @ p[:, 2 * d:2 * d + T] + b1[:, None]
            h = h + w2 @ np.maximum(y, 0) + b2[:, None]
        return 1 / (1 + np.exp(-(m["out"][0] @ h + m["out"][1][:, None])))

    def predict(self, x):
        """0-based (ah, br) plus scores and the ensemble's spread, as Tcn.kt."""
        per = [self._run(m, x) for m in self.models]
        p = np.mean(per, 0)
        ah, br = decode_events(p)
        single = [decode_events(q) for q in per]
        ahs, brs = [s[0] for s in single], [s[1] for s in single]
        return dict(ah=ah, br=br, ahScore=float(p[0, ah]), brScore=float(p[1, br]),
                    ahSpread=max(ahs) - min(ahs), brSpread=max(brs) - min(brs))


def peak(p, lo=0, hi=None, radius=3):
    hi = len(p) if hi is None else hi
    i = lo + int(p[lo:hi].argmax())
    a, b = max(lo, i - radius), min(hi, i + radius + 1)
    w = p[a:b]
    return float((np.arange(a, b) * w).sum() / w.sum())


def decode_events(p):
    ah = int(round(peak(p[0])))
    return ah, int(round(peak(p[1], lo=min(ah + 1, p.shape[1] - 1))))


def features(k):
    """(T, 17, 3) bowler keypoints, NaN where nobody -> (T, 97) features, as event_tcn.clip_features."""
    xy, conf = k[..., :2].astype(np.float64), np.nan_to_num(k[..., 2]).astype(np.float64)
    hip = xy[:, [L_HIP, R_HIP]].mean(1)
    shoulder = xy[:, [L_SHOULDER, R_SHOULDER]].mean(1)
    torso = np.nanmedian(np.linalg.norm(shoulder - hip, axis=1))
    centre = hip.copy()
    good = ~np.isnan(centre).any(1)
    if not good.any():
        raise ValueError("no frame with both hips")
    for d in range(2):
        centre[:, d] = np.interp(np.arange(len(centre)), np.flatnonzero(good), centre[good, d])
    xy = np.nan_to_num((xy - centre[:, None]) / torso)
    vel = np.diff(xy, axis=0, prepend=xy[:1])
    ang = []
    for a, b in ((R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (R_SHOULDER, R_WRIST)):
        v = xy[:, b] - xy[:, a]
        t = np.arctan2(v[:, 1], v[:, 0])
        ang += [np.sin(t), np.cos(t)]
    ang = np.stack(ang, 1)
    ang = np.concatenate([ang, np.diff(ang, axis=0, prepend=ang[:1])], 1)
    T = len(xy)
    return np.concatenate([xy.reshape(T, -1), vel.reshape(T, -1), conf, ang], 1).astype(np.float32)


# ---------------------------------------------------------------- the two passes

class Analyzer:
    def __init__(self, model_dir, threads=4):
        self.dir = Path(model_dir)
        self.threads = threads
        self.tcn = Tcn(self.dir / "tcn.bin")
        self._models = {}

    def model(self, name):
        if name not in self._models:
            self._models[name] = PoseModel(self.dir / name, self.threads)
        return self._models[name]

    def _events(self, k):
        if np.isnan(k[:, L_HIP, 0]).all():
            return None
        try:
            return self.tcn.predict(features(k))
        except ValueError:
            return None

    def analyze(self, video, out_dir, pass2_model, source="video", progress=lambda s, f: None,
                cancelled=lambda: False):
        out_dir = Path(out_dir)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise Rejected("This video can't be opened.", "Try exporting the clip again as MP4.")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # pass 1: every frame, streamed
        t0 = time.time()
        m1 = self.model(PASS1)
        rough = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if cancelled():
                raise Cancelled()
            rough.append(bowler(m1.detect(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))))
            if total and len(rough) % 8 == 0:
                progress("Finding the bowler", min(1.0, len(rough) / total))
        cap.release()
        total = len(rough)
        if total < 60:
            raise Rejected(f"The clip is too short ({total} frames).", "Record from the start of the run-up until after release.")
        rough = np.array(rough)
        t1 = time.time()
        progress("Finding the delivery", 0.0)
        ev1 = self._events(rough)
        if ev1 is None:
            raise Rejected("No bowler was found in the clip.", "Film side-on with the whole bowler in frame.")
        if ev1["ahScore"] < MIN_SCORE or ev1["brScore"] < MIN_SCORE:
            raise Rejected("No delivery was found in the clip.",
                           "Make sure the whole action, from arm horizontal to release, is in the recording.")
        ws, we = max(0, ev1["ah"] - MARGIN), min(total - 1, ev1["br"] + MARGIN)
        n = we - ws + 1

        # pass 2: the window only
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        m2 = self.model(pass2_model)
        cap = cv2.VideoCapture(str(video))
        fine, size, i = [], (0, 0), 0
        while i <= we:
            ok, f = cap.read()
            if not ok:
                break
            if i >= ws:
                if cancelled():
                    raise Cancelled()
                size = (f.shape[1], f.shape[0])
                fine.append(bowler(m2.detect(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))))
                s = min(1.0, SAVE_MAX / max(size))
                small = cv2.resize(f, (round(size[0] * s), round(size[1] * s)), interpolation=cv2.INTER_AREA) if s < 1 else f
                cv2.imwrite(str(frames_dir / f"{i - ws:05d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
                progress("Measuring the delivery", (i - ws + 1) / n)
            i += 1
        cap.release()
        fine = np.array(fine)
        n = len(fine)
        t2 = time.time()
        progress("Finding AH and BR", 0.0)
        ev2 = self._events(fine) if n else None
        if ev2 is None:
            raise Rejected("The bowler was lost during the delivery.", "Keep the whole bowler in frame.")
        arm = fine[:, [R_SHOULDER, R_ELBOW, R_WRIST], 2]
        arm_conf = float(np.nanmean(arm)) if not np.isnan(arm).all() else 0.0
        if arm_conf < MIN_ARM_CONF:
            raise Rejected("The bowling arm isn't clearly visible.", "Film from the bowling-arm side, side-on, in good light.")
        t3 = time.time()

        kp = [None if np.isnan(v) else round(float(v), 1) for v in fine.reshape(-1)]
        res = {
            "version": 1, "id": out_dir.name, "createdAt": int(time.time() * 1000), "source": source,
            "frameCount": total, "durationMs": int(total / fps * 1000) if fps else 0,
            "captureFps": None, "width": size[0], "height": size[1],
            "windowStart": ws + 1, "windowEnd": ws + n,
            "ah": ws + ev2["ah"] + 1, "br": ws + ev2["br"] + 1,
            "ahScore": ev2["ahScore"], "brScore": ev2["brScore"],
            "ahSpread": ev2["ahSpread"], "brSpread": ev2["brSpread"],
            "roughAh": ev1["ah"] + 1, "roughBr": ev1["br"] + 1, "armVisibility": arm_conf,
            "pass1Model": PASS1, "pass2Model": pass2_model,
            "pass1Accelerator": "server CPU", "pass2Accelerator": "server CPU",
            "pass1Ms": int((t1 - t0) * 1000), "pass2Ms": int((t2 - t1) * 1000), "tcnMs": int((t3 - t2) * 1000),
            "keypoints": kp,
        }
        (out_dir / "result.json").write_text(json.dumps(res))
        return res
