#!/usr/bin/env python3
# tools/knn_from_formant_raw_to_mouth_timeline.py
import argparse
import glob
import json
import math
from typing import Any, Dict, List, Tuple

VALID_VOWEL_IDS = {1, 2, 3, 4, 5}

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and (x == x)

def _load_knn_db(gt_paths: List[str]) -> List[Tuple[float, float, int]]:
    # returns list of (f1_hz, f2_hz, vowel_id)
    db: List[Tuple[float, float, int]] = []
    for p in gt_paths:
        obj = _load_json(p)
        for fr in obj.get("frames", []):
            vid = fr.get("vowel_id")
            f1 = fr.get("f1_hz")
            f2 = fr.get("f2_hz")
            if vid in VALID_VOWEL_IDS and _is_num(f1) and _is_num(f2):
                db.append((float(f1), float(f2), int(vid)))
    if not db:
        raise SystemExit("[ERROR] empty kNN db (no frames loaded)")
    return db

def _compute_z(db: List[Tuple[float, float, int]]) -> Tuple[float, float, float, float]:
    # mu1, mu2, s1, s2
    n = len(db)
    mu1 = sum(x for x, _, _ in db) / n
    mu2 = sum(y for _, y, _ in db) / n
    v1 = sum((x - mu1) ** 2 for x, _, _ in db) / n
    v2 = sum((y - mu2) ** 2 for _, y, _ in db) / n
    s1 = math.sqrt(v1) if v1 > 0 else 1.0
    s2 = math.sqrt(v2) if v2 > 0 else 1.0
    return mu1, mu2, s1, s2

def _z_point(f1: float, f2: float, z: Tuple[float, float, float, float]) -> Tuple[float, float]:
    mu1, mu2, s1, s2 = z
    return ( (f1 - mu1) / s1, (f2 - mu2) / s2 )

def _predict_knn(
    q: Tuple[float, float],
    db_z: List[Tuple[Tuple[float, float], int]],
    k: int,
    eps: float = 1e-9,
) -> Tuple[int, float, float]:
    # returns (pred_id, top_weight, top2_weight)
    q1, q2 = q
    dlist: List[Tuple[float, int]] = []
    for (x1, x2), vid in db_z:
        d2 = (q1 - x1) ** 2 + (q2 - x2) ** 2
        dlist.append((d2, vid))
    dlist.sort(key=lambda t: t[0])
    kk = min(k, len(dlist))
    votes: Dict[int, float] = {}
    for i in range(kk):
        d2, vid = dlist[i]
        w = 1.0 / (d2 + eps)
        votes[vid] = votes.get(vid, 0.0) + w
    # top / top2
    items = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    pred = items[0][0]
    top = items[0][1]
    top2 = items[1][1] if len(items) >= 2 else 0.0
    return pred, top, top2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="input: *.mouth_timeline.formant.raw.json")
    ap.add_argument("--gt_glob", default="data/knn_db/*.f1f2.json", help="kNN db glob")
    ap.add_argument("--out", required=True, help="output mouth_timeline.json")
    ap.add_argument("--audio", default=None, help="optional wav path to write into output json")
    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--fallback_id_active", type=int, default=2, help="when vad_active=1 but f1/f2 missing, use this mouth_id (default: 2=i)")
    ap.add_argument("--min_conf_ratio", type=float, default=1.0, help="if top/top2 < ratio, treat as low-conf (still non-zero if vad_active=1)")
    args = ap.parse_args()

    gt_paths = sorted(glob.glob(args.gt_glob))
    if not gt_paths:
        raise SystemExit(f"[ERROR] no gt files matched: {args.gt_glob}")

    db = _load_knn_db(gt_paths)
    z = _compute_z(db)
    db_z = [(_z_point(f1, f2, z), vid) for (f1, f2, vid) in db]

    raw = _load_json(args.raw)
    frames_in = raw.get("frames", [])
    out_frames: List[Dict[str, Any]] = []

    missing_active = 0
    lowconf_active = 0

    for fr in frames_in:
        t_ms = fr.get("t_ms")
        vad = fr.get("vad_active")
        if t_ms is None:
            continue
        t_ms = int(t_ms)
        vad = int(vad) if vad is not None else 0

        if vad == 0:
            out_frames.append({"t_ms": t_ms, "mouth_id": 0})
            continue

        # vad==1: mouth_id must be 1..5
        f1 = fr.get("f1_hz")
        f2 = fr.get("f2_hz")
        if not (_is_num(f1) and _is_num(f2)):
            missing_active += 1
            mid = int(args.fallback_id_active)
            if mid == 0:
                mid = 2
            out_frames.append({"t_ms": t_ms, "mouth_id": mid})
            continue

        qz = _z_point(float(f1), float(f2), z)
        pred, top, top2 = _predict_knn(qz, db_z, args.k)
        ratio = (top / top2) if top2 > 0 else 999.0

        if ratio < args.min_conf_ratio:
            lowconf_active += 1
            # 低信頼でも vad==1 なので 0 に落とさない。i に寄せる（無難）
            pred = int(args.fallback_id_active) if int(args.fallback_id_active) != 0 else 2

        out_frames.append({"t_ms": t_ms, "mouth_id": int(pred)})

    out_obj: Dict[str, Any] = {
        "audio": args.audio if args.audio is not None else "",
        "step_ms": int(args.step_ms),
        "frames": out_frames,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print("[OK] wrote:", args.out)
    print("  raw:", args.raw)
    print("  gt_files:", len(gt_paths), "db_frames:", len(db))
    print("  step_ms:", args.step_ms, "k:", args.k)
    print("  missing_active_frames:", missing_active)
    print("  lowconf_active_frames:", lowconf_active)

if __name__ == "__main__":
    main()
