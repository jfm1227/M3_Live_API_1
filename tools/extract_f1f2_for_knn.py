#!/usr/bin/env python3
# tools/extract_f1f2_for_knn.py
import argparse
import json
import os
from typing import Any, Dict, List

# mouth_id -> vowel_id (a i u e o)
# 合意：close(0) は除外（必要なら --keep_close で残せる）
VALID_VOWEL_IDS = {1, 2, 3, 4, 5}

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and (x == x)  # NaN除外

def extract_frames(src: Dict[str, Any], keep_close: bool) -> List[Dict[str, Any]]:
    frames = src.get("frames", [])
    out: List[Dict[str, Any]] = []
    for fr in frames:
        vowel_id = fr.get("mouth_id", None)  # この raw.json では mouth_id が教師ラベル
        f1 = fr.get("f1_hz", None)
        f2 = fr.get("f2_hz", None)

        if vowel_id is None:
            continue

        if (not keep_close) and (vowel_id == 0):
            continue

        if (vowel_id != 0) and (vowel_id not in VALID_VOWEL_IDS):
            # 想定外IDは無視（事故防止）
            continue

        if not (_is_number(f1) and _is_number(f2)):
            # null混在を確実に排除
            continue

        out.append({"vowel_id": int(vowel_id), "f1_hz": float(f1), "f2_hz": float(f2)})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--keep_close", action="store_true", help="vowel_id==0 も出力に残す")
    ap.add_argument("--min_frames", type=int, default=1)
    args = ap.parse_args()

    src = _load_json(args.in_path)
    frames = extract_frames(src, keep_close=args.keep_close)

    if len(frames) < args.min_frames:
        raise SystemExit(f"[ERROR] too few numeric frames: {len(frames)} < min_frames={args.min_frames}")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump({"frames": frames}, f, ensure_ascii=False)

    # 簡易サマリ（事実のみ）
    print(f"[OK] wrote: {args.out_path}")
    print(f"  numeric_frames: {len(frames)}")
    # vowel 分布
    counts: Dict[int, int] = {}
    for fr in frames:
        vid = fr["vowel_id"]
        counts[vid] = counts.get(vid, 0) + 1
    print(f"  vowel_counts: {dict(sorted(counts.items()))}")

if __name__ == "__main__":
    main()
