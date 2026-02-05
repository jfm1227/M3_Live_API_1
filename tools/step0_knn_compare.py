#!/usr/bin/env python3
# tools/step0_knn_compare.py
import argparse
import json
from typing import Any, Dict, List, Tuple

from scripts.mouth_insert_intermediate_i import insert_intermediate_i  # i挿入（結合後一括）

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _dump_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _frames_to_map(frames: List[Dict[str, Any]]) -> Dict[int, int]:
    m: Dict[int, int] = {}
    for fr in frames:
        t = fr.get("t_ms")
        mid = fr.get("mouth_id")
        if t is None or mid is None:
            continue
        m[int(t)] = int(mid)  # same t_ms: last wins
    return m

def _switch_per_sec(frames: List[Dict[str, Any]], step_ms: int) -> float:
    if not frames:
        return 0.0
    # assume frames are already in time order
    prev = int(frames[0].get("mouth_id", 0))
    switches = 0
    for fr in frames[1:]:
        cur = int(fr.get("mouth_id", 0))
        if cur != prev:
            switches += 1
            prev = cur
    duration_s = (step_ms * max(0, len(frames) - 1)) / 1000.0
    return switches / max(duration_s, 1e-9)

def _compare(golden: Dict[int, int], pred: Dict[int, int]) -> Tuple[int, int, int]:
    # returns (n_total, n_mouth_mismatch, n_vad_mismatch)
    keys = sorted(set(golden.keys()) | set(pred.keys()))
    mouth_mis = 0
    vad_mis = 0
    for t in keys:
        g = int(golden.get(t, 0))
        p = int(pred.get(t, 0))
        if g != p:
            mouth_mis += 1
        gv = 1 if g != 0 else 0
        pv = 1 if p != 0 else 0
        if gv != pv:
            vad_mis += 1
    return len(keys), mouth_mis, vad_mis

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="kNN output json (before i-insert)")
    ap.add_argument("--golden", required=True, help="golden mouth_timeline json (SSOT)")
    ap.add_argument("--out_json", required=True, help="output json after i-insert")
    ap.add_argument("--min_run_ms", type=int, default=320)
    ap.add_argument("--interval_ms", type=int, default=320)
    ap.add_argument("--max_i_per_run", type=int, default=3)
    args = ap.parse_args()

    src = _load_json(args.in_json)
    step_ms = int(src.get("step_ms", 40))
    frames = src.get("frames", [])
    if not isinstance(frames, list):
        raise SystemExit("[ERROR] invalid frames in in_json")

    # 1) i挿入（結合後一括）
    inserted_frames = insert_intermediate_i(
        frames,
        step_ms=step_ms,
        min_run_ms=args.min_run_ms,
        interval_ms=args.interval_ms,
        max_i_per_run=args.max_i_per_run,
    )

    out_obj: Dict[str, Any] = dict(src)
    out_obj["frames"] = inserted_frames
    _dump_json(args.out_json, out_obj)

    # 2) 比較
    gold = _load_json(args.golden)
    gold_step = int(gold.get("step_ms", step_ms))
    if gold_step != step_ms:
        print(f"[WARN] step_ms mismatch: pred={step_ms} golden={gold_step}")

    gold_map = _frames_to_map(gold.get("frames", []))
    pred_map0 = _frames_to_map(frames)
    pred_map1 = _frames_to_map(inserted_frames)

    n0, mouth_mis0, vad_mis0 = _compare(gold_map, pred_map0)
    n1, mouth_mis1, vad_mis1 = _compare(gold_map, pred_map1)

    sw0 = _switch_per_sec(frames, step_ms)
    sw1 = _switch_per_sec(inserted_frames, step_ms)

    print("===== Step0 kNN regression check =====")
    print(f"- in      : {args.in_json}")
    print(f"- golden  : {args.golden}")
    print(f"- out(i)  : {args.out_json}")
    print(f"- step_ms : {step_ms}")
    print("")
    print("[compare] BEFORE i-insert")
    print(f"  total_t_ms={n0}  mouth_mismatch={mouth_mis0}  vad_mismatch={vad_mis0}  switch_per_sec={sw0:.3f}")
    print("[compare] AFTER  i-insert")
    print(f"  total_t_ms={n1}  mouth_mismatch={mouth_mis1}  vad_mismatch={vad_mis1}  switch_per_sec={sw1:.3f}")

    # 合否の最低条件（ここは必要なら後で調整）
    if vad_mis1 != 0:
        raise SystemExit("[FAIL] VAD mismatch exists after i-insert (must be 0 in Step0)")
    if mouth_mis1 != 0:
        raise SystemExit("[FAIL] mouth_id mismatch exists after i-insert (must be 0 in Step0)")
    print("[PASS] Step0 matches golden (after i-insert).")

if __name__ == "__main__":
    main()
