#!/usr/bin/env python3
# Minimal distance-weighted k-NN for vowel classification
# GT json -> immediate prediction

import json
import math
import argparse
from collections import defaultdict, Counter

EPS = 1e-6

def load_gt_points(gt_json, include_close=False):
    """
    Support common schemas:
      A) frame has: mouth_id (0..5), f1_hz, f2_hz (top-level)   [your *.formant.raw.json]
      B) frame has: mouth_id (0..5), meta.f1_hz, meta.f2_hz
      C) legacy:    vowel_id (0..5), f1_hz, f2_hz

    Container keys supported:
      - dict["frames"] (recommended)
      - dict["timeline"]
      - root list
    """
    pts = []
    with open(gt_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    dbg = None

    # pick frames list robustly
    if isinstance(data, dict):
        frames = data.get("frames")
        if frames is None:
            frames = data.get("timeline")
        if frames is None:
            frames = []
        # some dumps keep formants only in debug_frames[].meta
        dbg = data.get("debug_frames")
        if not (isinstance(dbg, list) and len(dbg) == len(frames)):
            dbg = None
    elif isinstance(data, list):
        frames = data
    else:
        frames = []

    for i, fr in enumerate(frames):
        if not isinstance(fr, dict):
            continue
        dfr = dbg[i] if dbg is not None else None

        # label: prefer mouth_id, fallback to vowel_id
        vid = fr.get("mouth_id")
        if vid is None:
            vid = fr.get("vowel_id")
        if vid is None and isinstance(dfr, dict):
            vid = dfr.get("mouth_id", dfr.get("vowel_id"))
        if vid is None:
            continue
        vid = int(vid)

        if (not include_close) and vid == 0:
            continue

        # features:
        # 1) fr.f1_hz/f2_hz (may exist but be None)
        f1 = fr.get("f1_hz")
        f2 = fr.get("f2_hz")
        # 2) fr.meta.f1_hz/f2_hz
        if f1 is None or f2 is None:
            meta = fr.get("meta") if isinstance(fr.get("meta"), dict) else {}
            f1 = meta.get("f1_hz") if f1 is None else f1
            f2 = meta.get("f2_hz") if f2 is None else f2
        # 3) debug_frames[i].meta.f1_hz/f2_hz
        if (f1 is None or f2 is None) and isinstance(dfr, dict):
            dmeta = dfr.get("meta") if isinstance(dfr.get("meta"), dict) else {}
            f1 = dmeta.get("f1_hz") if f1 is None else f1
            f2 = dmeta.get("f2_hz") if f2 is None else f2

        if f1 is None or f2 is None:
            continue
        pts.append((float(f1), float(f2), int(vid)))
    return pts

def fit_scaler(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mu_x = sum(xs) / len(xs)
    mu_y = sum(ys) / len(ys)
    sx = math.sqrt(sum((x - mu_x) ** 2 for x in xs) / len(xs)) or 1.0
    sy = math.sqrt(sum((y - mu_y) ** 2 for y in ys) / len(ys)) or 1.0
    return (mu_x, mu_y, sx, sy)

def scale(p, scaler):
    mu_x, mu_y, sx, sy = scaler
    return ((p[0] - mu_x) / sx, (p[1] - mu_y) / sy)

def knn_predict(point, train_scaled, k):
    # train_scaled: [(x,y,label), ...] scaled
    x, y = point
    dists = []
    for tx, ty, lab in train_scaled:
        d2 = (x - tx) ** 2 + (y - ty) ** 2
        dists.append((d2, lab))
    dists.sort(key=lambda z: z[0])
    topk = dists[:k]

    weights = defaultdict(float)
    for d2, lab in topk:
        w = 1.0 / (d2 + EPS)
        weights[lab] += w

    # argmax weight
    best_lab = max(weights.items(), key=lambda kv: kv[1])[0]
    return best_lab, weights, topk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT mouth_timeline json (with f1_hz,f2_hz)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--include_close", action="store_true")
    ap.add_argument("--self_test", action="store_true",
                    help="Predict each GT frame using the same GT set (sanity check)")
    args = ap.parse_args()

    pts = load_gt_points(args.gt, include_close=args.include_close)
    if len(pts) <= 0:
        raise SystemExit("[ERR] No usable GT points. Check JSON schema / f1,f2 location.")

    # for leave-one-out we need at least 2 points; also keep k within usable range
    if args.self_test:
        k_eff = min(args.k, len(pts))
    else:
        if len(pts) < 2:
            raise SystemExit("[ERR] Need at least 2 usable points for leave-one-out.")
        k_eff = min(args.k, len(pts) - 1)

    if k_eff < args.k:
        print(f"[WARN] k reduced: requested k={args.k} -> using k={k_eff} (n_pts={len(pts)})", flush=True)

    scaler = fit_scaler(pts)
    scaled = [(*scale(p, scaler), p[2]) for p in pts]

    ok = 0
    n = 0
    conf = defaultdict(Counter)

    for i, p in enumerate(pts):
        x, y = scale(p, scaler)
        if args.self_test:
            pred, _, _ = knn_predict((x, y), scaled, k_eff)
        else:
            train = scaled[:i] + scaled[i+1:]
            pred, _, _ = knn_predict((x, y), train, k_eff)

        gt = p[2]
        conf[gt][pred] += 1
        ok += int(pred == gt)
        n += 1

    acc = ok / max(1, n)
    print("===== k-NN minimal =====")
    print(f"- k={k_eff}  n={n}  acc={acc:.4f}")
    print("[confusion] rows=GT cols=PRED")
    labs = sorted(set(l for _,_,l in pts))
    header = "     " + " ".join(f"{l:>4}" for l in labs)
    print(header)
    for g in labs:
        row = f"{g:>4} "
        for p in labs:
            row += f"{conf[g][p]:>4} "
        print(row)

if __name__ == "__main__":
    main()