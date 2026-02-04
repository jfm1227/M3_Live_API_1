#!/usr/bin/env python3
# tools/knn_eval_pack.py
import argparse
import glob
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

VALID_VOWEL_IDS = {1, 2, 3, 4, 5}

def _load_gt(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    frames = obj.get("frames", [])
    out: List[Dict[str, Any]] = []
    for fr in frames:
        vid = fr.get("vowel_id")
        f1 = fr.get("f1_hz")
        f2 = fr.get("f2_hz")
        if vid in VALID_VOWEL_IDS and isinstance(f1, (int, float)) and isinstance(f2, (int, float)):
            out.append({"vowel_id": int(vid), "f1_hz": float(f1), "f2_hz": float(f2)})
    return out

@dataclass
class ZStats:
    mu1: float
    mu2: float
    s1: float
    s2: float

def _compute_zstats(frames: List[Dict[str, Any]]) -> ZStats:
    # GT由来の固定 μ/σ（全データから計算）
    n = len(frames)
    if n == 0:
        raise SystemExit("[ERROR] no frames")
    sum1 = sum(fr["f1_hz"] for fr in frames)
    sum2 = sum(fr["f2_hz"] for fr in frames)
    mu1 = sum1 / n
    mu2 = sum2 / n
    var1 = sum((fr["f1_hz"] - mu1) ** 2 for fr in frames) / n
    var2 = sum((fr["f2_hz"] - mu2) ** 2 for fr in frames) / n
    s1 = (var1 ** 0.5) if var1 > 0 else 1.0
    s2 = (var2 ** 0.5) if var2 > 0 else 1.0
    return ZStats(mu1=mu1, mu2=mu2, s1=s1, s2=s2)

def _z(fr: Dict[str, Any], zs: ZStats) -> Tuple[float, float]:
    return ((fr["f1_hz"] - zs.mu1) / zs.s1, (fr["f2_hz"] - zs.mu2) / zs.s2)

def _predict_knn(
    query: Tuple[float, float],
    train: List[Tuple[Tuple[float, float], int]],
    k: int,
    eps: float = 1e-9,
) -> int:
    # 距離² + 逆数重み
    q1, q2 = query
    dlist: List[Tuple[float, int]] = []
    for (x1, x2), vid in train:
        d2 = (q1 - x1) ** 2 + (q2 - x2) ** 2
        dlist.append((d2, vid))
    dlist.sort(key=lambda t: t[0])
    kk = min(k, len(dlist))
    votes: Dict[int, float] = {}
    for i in range(kk):
        d2, vid = dlist[i]
        w = 1.0 / (d2 + eps)
        votes[vid] = votes.get(vid, 0.0) + w
    # 最大重みのクラス
    return max(votes.items(), key=lambda kv: kv[1])[0]

def _print_confusion(conf: List[List[int]]) -> None:
    # rows=GT cols=PRED, 1..5
    print("[confusion] rows=GT cols=PRED")
    hdr = "        " + " ".join(f"{i:>4d}" for i in range(1, 6))
    print(hdr)
    for gt in range(1, 6):
        row = conf[gt]
        s = f"{gt:>4d} " + " ".join(f"{row[p]:>4d}" for p in range(1, 6))
        print(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gts", nargs="+", default=None, help="GT json paths (out/knn_data/train/*.f1f2.json)")
    ap.add_argument("--glob", default=None, help="glob pattern for GT json paths")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", choices=["global_loo", "file_loo"], default="global_loo")
    args = ap.parse_args()

    paths: List[str] = []
    if args.gts:
        paths.extend(args.gts)
    if args.glob:
        paths.extend(sorted(glob.glob(args.glob)))
    paths = sorted(list(dict.fromkeys(paths)))  # unique + keep order

    if not paths:
        raise SystemExit("[ERROR] specify --gts or --glob")

    all_frames: List[Dict[str, Any]] = []
    per_file: List[Tuple[str, List[Dict[str, Any]]]] = []
    for p in paths:
        frs = _load_gt(p)
        if len(frs) == 0:
            raise SystemExit(f"[ERROR] empty frames: {p}")
        per_file.append((p, frs))
        all_frames.extend(frs)

    zs = _compute_zstats(all_frames)

    print("===== k-NN pack eval =====")
    print(f"- files={len(per_file)}  total_frames={len(all_frames)}  k={args.k}  mode={args.mode}")

    conf = [[0] * 6 for _ in range(6)]
    correct = 0
    total = 0

    if args.mode == "global_loo":
        # 全フレームを1集合としてleave-one-out
        z_all = [(_z(fr, zs), fr["vowel_id"]) for fr in all_frames]
        for i, fr in enumerate(all_frames):
            gt = fr["vowel_id"]
            q = _z(fr, zs)
            train = z_all[:i] + z_all[i+1:]
            pred = _predict_knn(q, train, args.k)
            conf[gt][pred] += 1
            total += 1
            if pred == gt:
                correct += 1

    else:
        # file-one-out: 1ファイルを丸ごとテスト、残りを学習に使う
        for test_path, test_frames in per_file:
            train_frames = []
            for p, frs in per_file:
                if p != test_path:
                    train_frames.extend(frs)
            train_z = [(_z(fr, zs), fr["vowel_id"]) for fr in train_frames]
            for fr in test_frames:
                gt = fr["vowel_id"]
                q = _z(fr, zs)
                pred = _predict_knn(q, train_z, args.k)
                conf[gt][pred] += 1
                total += 1
                if pred == gt:
                    correct += 1

    acc = (correct / total) if total > 0 else 0.0
    print(f"- acc={acc:.4f}  correct={correct}  total={total}")
    _print_confusion(conf)

if __name__ == "__main__":
    main()
