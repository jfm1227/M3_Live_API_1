#!/usr/bin/env python3
# scripts/mouth_hold_postprocess.py
#
# 目的:
#   knn後の mouth_timeline（例: *.mouth_timeline.knn.i.json）に対して、
#   chunk境界ごとに「前chunkの最後の口形を 1点 hold」として反映する（口の跳ね防止）。
#
# 仕様（このスクリプトの挙動）:
#   - chunk_start_ms は t_min_ms から stride_ms=chunk_ms-overlap_ms 刻みで生成
#   - 各 chunk_start_ms（ただし最初のchunkを除く）について:
#       boundary t_ms が存在し、かつ (boundary-step_ms) が存在する場合
#         -> boundary の mouth_id を (boundary-step_ms) の mouth_id へ上書きする（hold 1点）
#       boundary t_ms が存在しないが直前フレームがある場合
#         -> boundary に 1フレーム挿入（t_ms=boundary, mouth_id=prev）
#   - フレーム数は「上書きのみ」なら増えない。挿入が発生した場合のみ増える。
#   - “golden一致”はここでは目的ではない（STEP-Cの安定化処理）。
#
# 入力例:
#   out/chunk_regress/1_0_72/joined.mouth_timeline.knn.i.json
#
# 出力例:
#   out/chunk_regress/1_0_72/joined.mouth_timeline.knn.i.hold.json
#
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _sorted_frames(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(frames, key=lambda fr: int(fr.get("t_ms", 0)))


def _index_by_t(frames: List[Dict[str, Any]]) -> Dict[int, int]:
    # t_ms -> index (最後に出たものを採用)
    m: Dict[int, int] = {}
    for i, fr in enumerate(frames):
        if "t_ms" not in fr:
            continue
        m[int(fr["t_ms"])] = i
    return m


def _chunk_starts(t_min: int, t_max: int, stride_ms: int) -> List[int]:
    starts: List[int] = []
    t = int(t_min)
    while t <= int(t_max):
        starts.append(int(t))
        t += int(stride_ms)
    return starts


def _clone_frame_with_new_t(fr: Dict[str, Any], t_ms: int) -> Dict[str, Any]:
    nf = dict(fr)
    nf["t_ms"] = int(t_ms)
    return nf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--chunk_ms", type=int, default=400)
    ap.add_argument("--overlap_ms", type=int, default=200)
    ap.add_argument("--step_ms", type=int, default=40)

    # advanced toggles
    ap.add_argument("--always_overwrite", action="store_true",
                    help="overwrite even if boundary mouth_id already equals prev")
    args = ap.parse_args()

    in_path = Path(args.in_json)
    out_path = Path(args.out_json)

    chunk_ms = int(args.chunk_ms)
    overlap_ms = int(args.overlap_ms)
    step_ms = int(args.step_ms)

    if overlap_ms < 0 or overlap_ms >= chunk_ms:
        raise SystemExit("[ERROR] overlap_ms must satisfy 0 <= overlap_ms < chunk_ms")
    stride_ms = chunk_ms - overlap_ms
    if step_ms <= 0:
        raise SystemExit("[ERROR] step_ms must be > 0")

    obj = _load_json(in_path)
    frames = obj.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise SystemExit("[ERROR] frames empty/invalid")

    frames = _sorted_frames(frames)

    t_list = [int(fr["t_ms"]) for fr in frames if "t_ms" in fr]
    if not t_list:
        raise SystemExit("[ERROR] no t_ms in frames")
    t_min, t_max = min(t_list), max(t_list)

    starts = _chunk_starts(t_min, t_max, stride_ms)

    # boundary hold changes
    changes_overwrite: List[Dict[str, Any]] = []
    changes_insert: List[Dict[str, Any]] = []

    # We'll mutate frames in-place, and re-sort at end if we insert.
    # For stable indexing after inserts, we recompute index map per boundary (cheap enough).
    for idx, cs in enumerate(starts):
        if idx == 0:
            continue  # first chunk: no previous chunk to hold from

        boundary_t = int(cs)
        prev_t = boundary_t - step_ms

        frames_map = _index_by_t(frames)
        if prev_t not in frames_map:
            # no previous frame => cannot define hold; skip
            continue

        prev_fr = frames[frames_map[prev_t]]
        prev_mouth = int(prev_fr.get("mouth_id", 0))

        if boundary_t in frames_map:
            # overwrite boundary mouth_id
            bi = frames_map[boundary_t]
            cur_mouth = int(frames[bi].get("mouth_id", 0))
            if args.always_overwrite or (cur_mouth != prev_mouth):
                frames[bi]["mouth_id"] = prev_mouth
                changes_overwrite.append({
                    "t_ms": boundary_t,
                    "prev_t_ms": prev_t,
                    "old_mouth_id": cur_mouth,
                    "new_mouth_id": prev_mouth,
                })
        else:
            # insert new boundary frame cloned from prev
            ins = _clone_frame_with_new_t(prev_fr, boundary_t)
            ins["mouth_id"] = prev_mouth
            frames.append(ins)
            changes_insert.append({
                "t_ms": boundary_t,
                "prev_t_ms": prev_t,
                "new_mouth_id": prev_mouth,
            })

    frames = _sorted_frames(frames)

    meta = obj.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}

    meta["hold_postprocess"] = {
        "schema": "m3.mouth_hold_postprocess.v1",
        "chunk_ms": chunk_ms,
        "overlap_ms": overlap_ms,
        "stride_ms": stride_ms,
        "step_ms": step_ms,
        "t_min_ms": t_min,
        "t_max_ms": t_max,
        "chunk_starts_n": len(starts),
        "overwrite_n": len(changes_overwrite),
        "insert_n": len(changes_insert),
        "overwrites": changes_overwrite[:200],  # debug cap
        "inserts": changes_insert[:200],        # debug cap
    }

    out_obj = dict(obj)
    out_obj["frames"] = frames
    out_obj["meta"] = meta

    _write_json(out_path, out_obj)

    print("[mouth_hold_postprocess][OK]")
    print(" in :", in_path.as_posix())
    print(" out:", out_path.as_posix())
    print(" frames_in :", len(t_list))
    print(" frames_out:", len(frames))
    print(" overwrite_n:", len(changes_overwrite))
    print(" insert_n   :", len(changes_insert))


if __name__ == "__main__":
    main()
