"""Train and evaluate a TCN that finds the AH and BR frames from pose sequences.

Run from the FYP folder:
    python src/event_tcn.py train            # 5-model ensemble on the train split, best epoch picked on val
    python src/event_tcn.py eval             # frame errors on val and test, per player
    python src/event_tcn.py eval --split test --csv outputs/tcn_test.csv
    python src/event_tcn.py train --pose yolo vit   # also feed the ViTPose++ keypoints

Input per frame: the bowler's 17 keypoints (centred on the hips, scaled by torso
length), their frame-to-frame velocity, the keypoint confidences, and the
directions of the bowling upper arm, forearm and shoulder-to-wrist line.
The predicted frame is the probability-weighted centre of the peak, averaged
over the ensemble.
Target per frame: two soft curves, a Gaussian around AH and one around BR,
because each clip has only one AH and one BR frame.
Frame numbers start at 1, same as labels.csv.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from event_labels import DEFAULT_LABELS, get_events, natural_key, read_labels

ROOT = Path(__file__).resolve().parent.parent
POSE_DIR = ROOT / "data" / "yolo26x"
VIT_DIR = ROOT / "data" / "vitpose_plus_base"
MODEL_DIR = ROOT / "models" / "event_tcn"
L_SHOULDER, R_SHOULDER, R_ELBOW, R_WRIST, L_HIP, R_HIP = 5, 6, 8, 10, 11, 12
SIGMA = 2.0          # width of the soft target, in frames
CROP = 128           # training window length, in frames


# ---------------------------------------------------------------- data

def _pose(model, clip):
    """Return (xy (T, 17, 2), conf (T, 17)) for the bowler."""
    name = clip.replace("/", "_")[:-4] + ".npz"
    if model == "yolo":
        k = np.load(POSE_DIR / name)["kpts"][:, 0]                 # largest person
        return k[..., :2].copy(), np.nan_to_num(k[..., 2])
    d = np.load(VIT_DIR / name)
    return d["kpts"].copy(), np.nan_to_num(d["conf"])


def _normalise(xy):
    hip = xy[:, [L_HIP, R_HIP]].mean(1)
    shoulder = xy[:, [L_SHOULDER, R_SHOULDER]].mean(1)
    torso = np.nanmedian(np.linalg.norm(shoulder - hip, axis=1))
    centre = hip.copy()
    good = ~np.isnan(centre).any(1)
    for d in range(2):                         # fill frames with no hips from neighbours
        centre[:, d] = np.interp(np.arange(len(centre)), np.flatnonzero(good), centre[good, d])
    return np.nan_to_num((xy - centre[:, None]) / torso)


def _arm_angles(xy):
    """sin/cos of the upper-arm, forearm and shoulder-to-wrist directions of the bowling arm."""
    out = []
    for a, b in ((R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (R_SHOULDER, R_WRIST)):
        v = xy[:, b] - xy[:, a]
        ang = np.arctan2(v[:, 1], v[:, 0])
        out += [np.sin(ang), np.cos(ang)]
    a = np.stack(out, 1)
    return np.concatenate([a, np.diff(a, axis=0, prepend=a[:1])], 1)


def clip_features(clip, models=("yolo",)):
    """Return (T, n_features) float32 features for one clip.

    The first 68 columns are always the YOLO positions and velocities (used by augmentation).
    """
    feats, extra = [], []
    for m in models:
        xy, conf = _pose(m, clip)
        xy = _normalise(xy)
        vel = np.diff(xy, axis=0, prepend=xy[:1])
        feats += [xy.reshape(len(xy), -1), vel.reshape(len(xy), -1)]
        extra += [conf, _arm_angles(xy)]
    return np.concatenate(feats + extra, 1).astype(np.float32)


def soft_targets(T, ah, br):
    t = np.arange(T)
    return np.stack([np.exp(-0.5 * ((t - (e - 1)) / SIGMA) ** 2) for e in (ah, br)], 1).astype(np.float32)


def load_split(labels, splits, models=("yolo",)):
    items = []
    for clip in sorted(labels, key=natural_key):
        ev = get_events(labels, clip)
        if ev and labels[clip].get("split") in splits:
            x = clip_features(clip, models)
            items.append({"clip": clip, "player": labels[clip]["player"], "x": x,
                          "y": soft_targets(len(x), *ev), "ah": ev[0], "br": ev[1]})
    return items


def random_batch(items, batch_size, rng):
    """Random CROP-frame windows, so the events land at different positions each time."""
    xs, ys = [], []
    for it in rng.choice(items, batch_size):
        x, y = it["x"], it["y"]
        s = rng.integers(0, len(x) - CROP + 1)
        x = x[s:s + CROP].copy()
        x[:, :68] *= rng.uniform(0.9, 1.1)                 # small scale jitter on positions/velocities
        x[:, :34] += rng.normal(0, 0.01, x[:, :34].shape)  # keypoint noise
        xs.append(x)
        ys.append(y[s:s + CROP])
    to_t = lambda a: torch.from_numpy(np.stack(a)).transpose(1, 2)  # (B, C, T)
    return to_t(xs), to_t(ys)


# ---------------------------------------------------------------- model

class Block(nn.Module):
    def __init__(self, ch, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, 1)
        self.norm = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.conv2(F.relu(self.norm(self.conv1(x)))))


class EventTCN(nn.Module):
    """Non-causal dilated TCN: one logit per frame for AH and for BR."""

    def __init__(self, n_in, ch=64, dilations=(1, 2, 4, 8, 16), dropout=0.2):
        super().__init__()
        self.inp = nn.Conv1d(n_in, ch, 1)
        self.blocks = nn.Sequential(*[Block(ch, d, dropout) for d in dilations])
        self.out = nn.Conv1d(ch, 2, 1)

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))


# ---------------------------------------------------------------- predict / evaluate

@torch.no_grad()
def probabilities(models, item):
    """Average per-frame AH / BR probabilities over an ensemble: shape (2, T)."""
    x = torch.from_numpy(item["x"]).T[None]
    ps = []
    for m in models:
        m.eval()
        ps.append(torch.sigmoid(m(x))[0].numpy())
    return np.mean(ps, 0)


def peak(p, lo=0, hi=None, radius=3):
    """Frame index of the peak of p[lo:hi], refined to the probability-weighted centre around it."""
    hi = len(p) if hi is None else hi
    i = lo + int(p[lo:hi].argmax())
    a, b = max(lo, i - radius), min(hi, i + radius + 1)
    w = p[a:b]
    return float((np.arange(a, b) * w).sum() / w.sum())


def predict(models, item):
    """Return predicted (ah, br), 1-based. BR is searched only after AH."""
    p = probabilities(models, item)
    ah = peak(p[0])
    br = peak(p[1], lo=min(int(round(ah)) + 1, p.shape[1] - 1))
    return int(round(ah)) + 1, int(round(br)) + 1


def evaluate(models, items):
    rows = []
    for it in items:
        ah, br = predict(models, it)
        rows.append({"clip": it["clip"], "player": it["player"], "ah": it["ah"], "br": it["br"],
                     "pred_ah": ah, "pred_br": br, "err_ah": ah - it["ah"], "err_br": br - it["br"]})
    return rows


def summary(rows):
    e_ah = np.abs([r["err_ah"] for r in rows])
    e_br = np.abs([r["err_br"] for r in rows])
    s = f"n={len(rows):3d}  AH mae {e_ah.mean():5.2f}  BR mae {e_br.mean():5.2f}  "
    s += "  ".join(f"<={k}: AH {np.mean(e_ah <= k):4.0%} BR {np.mean(e_br <= k):4.0%}" for k in (1, 2, 3))
    return s


# ---------------------------------------------------------------- main

def train_one(tr, va, seed, epochs, log=print):
    """Train one model; keep the epoch with the lowest val error. Returns the model."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = EventTCN(tr[0]["x"].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    pos_weight = torch.tensor([20.0, 20.0])[None, :, None]   # events are rare frames
    best, best_state = None, None
    for epoch in range(1, epochs + 1):
        model.train()
        for _ in range(20):
            x, y = random_batch(tr, 32, rng)
            loss = F.binary_cross_entropy_with_logits(model(x), y, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 5 == 0:
            rows = evaluate([model], va)
            score = np.mean([abs(r["err_ah"]) + abs(r["err_br"]) for r in rows])
            if best is None or score < best:
                best, best_state = score, {k: v.clone() for k, v in model.state_dict().items()}
            log(f"  seed {seed} epoch {epoch:3d} loss {loss.item():.4f}  val {summary(rows)}")
    model.load_state_dict(best_state)
    return model


def train(args, labels):
    models = tuple(args.pose)
    tr, va = load_split(labels, {"train"}, models), load_split(labels, {"val"}, models)
    print(f"train {len(tr)} clips, val {len(va)} clips, pose {'+'.join(models)}, "
          f"{tr[0]['x'].shape[1]} features")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for old in MODEL_DIR.glob("seed*.pt"):
        old.unlink()
    for seed in range(args.seeds):
        model = train_one(tr, va, seed, args.epochs, log=print if args.verbose else lambda s: None)
        torch.save({"state": model.state_dict(), "pose": models, "n_in": tr[0]["x"].shape[1]},
                   MODEL_DIR / f"seed{seed}.pt")
        print(f"seed {seed}: val {summary(evaluate([model], va))}", flush=True)
    print(f"{args.seeds} models -> {MODEL_DIR.relative_to(ROOT)}/")


def load_models():
    models, pose = [], None
    for f in sorted(MODEL_DIR.glob("seed*.pt")):
        ck = torch.load(f)
        m = EventTCN(ck["n_in"])
        m.load_state_dict(ck["state"])
        models.append(m)
        pose = tuple(ck["pose"])
    if not models:
        raise SystemExit(f"no models in {MODEL_DIR}, run train first")
    return models, pose


def run_eval(args, labels):
    models, pose = load_models()
    print(f"ensemble of {len(models)} models, pose {'+'.join(pose)}")
    splits = [args.split] if args.split else ["val", "test"]
    all_rows = []
    for split in splits:
        rows = evaluate(models, load_split(labels, {split}, pose))
        print(f"{split:5s} {summary(rows)}")
        for p in sorted({r["player"] for r in rows}, key=natural_key):
            print(f"  {p:4s} {summary([r for r in rows if r['player'] == p])}")
        all_rows += [{"split": split, **r} for r in rows]
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
        print(f"-> {args.csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--pose", nargs="+", choices=["yolo", "vit"], default=["yolo"],
                    help="which pose outputs to use as input")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=5, help="ensemble size")
    ap.add_argument("--verbose", action="store_true", help="print val scores during training")
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--csv", help="also save per-clip predictions here")
    args = ap.parse_args()
    labels, _ = read_labels(args.labels)
    train(args, labels) if args.mode == "train" else run_eval(args, labels)


if __name__ == "__main__":
    main()
