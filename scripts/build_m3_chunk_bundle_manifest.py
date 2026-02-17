#!/usr/bin/env python3
# scripts/build_m3_chunk_bundle_manifest.py
#
# 目的（STEP-Fのための“最小統合”）:
#   raw_chunks/manifest.json の chunk_start_ms をSSOTとして、
#   - mouth（session結合済み：knn.i.hold）を chunk単位に切り出し
#   - expression_chunks.v1.json を chunk単位に（必要なら overlap 前半を落として）切り出し
#   - mouth/expression が同じ chunk_start_ms / step_ms で揃っていることを機械的に検査できる
#   - 最終的に「chunk bundle manifest」を1本生成する
#
# 生成物:
#   <out_dir>/m3_chunk_bundle.manifest.json
#   <out_dir>/chunks/chunk_0000.mouth.json
#   <out_dir>/chunks/chunk_0000.expression.json
#   ...
#
# mouthの切り出し仕様:
#   - chunk window は [cs, cs+chunk_ms)
#   - chunk i>0 では「重複送信を避ける」ため overlap 前半を落とす:
#       keep: t_rel==0 (初期hold) と t_rel>=overlap_ms
#     ※ t_rel==0 は mouth_hold_postprocess 済みを前提に “境界で跳ねない” 初期状態固定用
#
# expressionの切り出し仕様:
#   - expression_chunks.v1.json の該当chunk.events（t_msはchunk相対）を
#     chunk i>0 では mouth と同じ条件で filter:
#       keep: t_ms==0 と t_ms>=overlap_ms
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


def _sorted_by_t(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(frames, key=lambda fr: int(fr.get("t_ms", 0)))


def _index_frames_by_abs_t(frames: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    m: Dict[int, Dict[str, Any]] = {}
    for fr in frames:
        if "t_ms" not in fr:
            continue
        m[int(fr["t_ms"])] = fr
    return m


def _extract_abs_mouth_frames(
    mouth_frames_abs: List[Dict[str, Any]],
    cs: int,
    chunk_ms: int,
    step_ms: int,
) -> List[Dict[str, Any]]:
    # [cs, cs+chunk_ms) を step_ms 格子で取り出す（存在するものだけ）
    by_t = _index_frames_by_abs_t(mouth_frames_abs)
    out: List[Dict[str, Any]] = []
    for t in range(int(cs), int(cs) + int(chunk_ms), int(step_ms)):
        fr = by_t.get(int(t))
        if fr is not None:
            out.append(fr)
    return out


def _mouth_abs_to_rel(
    frames_abs: List[Dict[str, Any]],
    base_cs: int,
) -> List[Dict[str, Any]]:
    # t_ms を chunk 相対にし、必要最小キーのみ残す（mouth_id / vad_active があれば残す）
    out: List[Dict[str, Any]] = []
    for fr in frames_abs:
        t_abs = int(fr["t_ms"])
        fr2: Dict[str, Any] = {
            "t_ms": int(t_abs - int(base_cs)),
            "mouth_id": int(fr.get("mouth_id", 0)),
        }
        if "vad_active" in fr:
            fr2["vad_active"] = int(fr.get("vad_active", 0))
        out.append(fr2)
    out = _sorted_by_t(out)
    return out


def _filter_drop_overlap_keep_hold(
    frames_rel: List[Dict[str, Any]],
    overlap_ms: int,
) -> List[Dict[str, Any]]:
    # keep: t==0 OR t>=overlap_ms
    out = [fr for fr in frames_rel if int(fr.get("t_ms", -1)) == 0 or int(fr.get("t_ms", -1)) >= int(overlap_ms)]
    out = _sorted_by_t(out)
    return out


def _find_expression_chunk(exp_obj: Dict[str, Any], cs: int) -> Dict[str, Any]:
    chunks = exp_obj.get("chunks", [])
    for ch in chunks:
        if int(ch.get("chunk_start_ms", -1)) == int(cs):
            return ch
    raise FileNotFoundError(f"[build_m3_chunk_bundle_manifest][ERROR] expression chunk not found for chunk_start_ms={cs}")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--raw_manifest", required=True, help="SSOT: out/.../raw_chunks/manifest.json")
    ap.add_argument("--mouth_session", required=True, help="session mouth (e.g. joined.mouth_timeline.knn.i.hold.json)")
    ap.add_argument("--expression_chunks", required=True, help="expression_chunks.v1.json")
    ap.add_argument("--out_dir", required=True, help="output dir for bundle manifest + chunks/*")

    ap.add_argument("--chunk_ms", type=int, default=400)
    ap.add_argument("--overlap_ms", type=int, default=200)
    ap.add_argument("--step_ms", type=int, default=40)

    ap.add_argument("--drop_overlap", action="store_true", help="for chunk i>0, drop overlap part (keep t=0 + t>=overlap_ms)")
    args = ap.parse_args()

    raw_manifest_path = Path(args.raw_manifest)
    mouth_session_path = Path(args.mouth_session)
    exp_chunks_path = Path(args.expression_chunks)
    out_dir = Path(args.out_dir)

    chunk_ms = int(args.chunk_ms)
    overlap_ms = int(args.overlap_ms)
    step_ms = int(args.step_ms)

    if overlap_ms < 0 or overlap_ms >= chunk_ms:
        raise SystemExit("[ERROR] overlap_ms must satisfy 0 <= overlap_ms < chunk_ms")

    man = _load_json(raw_manifest_path)
    man_chunks = man.get("chunks", [])
    if not isinstance(man_chunks, list) or not man_chunks:
        raise SystemExit("[ERROR] raw_manifest.chunks empty")

    mouth = _load_json(mouth_session_path)
    mouth_frames_abs = mouth.get("frames", [])
    if not isinstance(mouth_frames_abs, list) or not mouth_frames_abs:
        raise SystemExit("[ERROR] mouth_session.frames empty")

    exp = _load_json(exp_chunks_path)
    exp_step = int(exp.get("step_ms", step_ms))

    mouth_step = int(mouth.get("step_ms", step_ms))
    if mouth_step != step_ms:
        raise SystemExit(f"[ERROR] mouth_session step_ms={mouth_step} != --step_ms {step_ms}")
    if exp_step != step_ms:
        raise SystemExit(f"[ERROR] expression step_ms={exp_step} != --step_ms {step_ms}")

    chunks_out_dir = out_dir / "chunks"
    chunks_out_dir.mkdir(parents=True, exist_ok=True)

    bundle_chunks: List[Dict[str, Any]] = []
    total_missing_mouth_frames = 0

    for idx, ch in enumerate(man_chunks):
        i = int(ch["i"])
        cs = int(ch["chunk_start_ms"])

        # mouth: abs slice -> rel
        mouth_abs_slice = _extract_abs_mouth_frames(mouth_frames_abs, cs, chunk_ms, step_ms)
        mouth_rel = _mouth_abs_to_rel(mouth_abs_slice, base_cs=cs)

        # sanity: expected count before filtering (if fully dense)
        expected_dense = chunk_ms // step_ms
        # how many are missing from ideal grid
        missing = expected_dense - len(mouth_rel)
        if missing < 0:
            missing = 0
        total_missing_mouth_frames += missing

        # drop overlap for chunk i>0 (keep t=0 as initial hold)
        if args.drop_overlap and i > 0:
            mouth_rel = _filter_drop_overlap_keep_hold(mouth_rel, overlap_ms)

        mouth_chunk_obj = {
            "schema": "m3.mouth_chunk.v1",
            "chunk_index": i,
            "chunk_start_ms": cs,
            "step_ms": step_ms,
            "chunk_ms": chunk_ms,
            "overlap_ms": overlap_ms,
            "frames": mouth_rel,
        }

        mouth_path = chunks_out_dir / f"chunk_{i:04d}.mouth.json"
        _write_json(mouth_path, mouth_chunk_obj)

        # expression: find chunk, optionally drop overlap
        exp_ch = _find_expression_chunk(exp, cs)
        exp_events = exp_ch.get("events", [])
        if not isinstance(exp_events, list):
            raise SystemExit(f"[ERROR] invalid expression events for cs={cs}")

        if args.drop_overlap and i > 0:
            exp_events = [ev for ev in exp_events if int(ev.get("t_ms", -1)) == 0 or int(ev.get("t_ms", -1)) >= int(overlap_ms)]

        exp_chunk_obj = {
            "schema": "m3.expression_chunk.v1",
            "chunk_index": i,
            "chunk_start_ms": cs,
            "step_ms": step_ms,
            "chunk_ms": chunk_ms,
            "overlap_ms": overlap_ms,
            "events": exp_events,
        }

        exp_path = chunks_out_dir / f"chunk_{i:04d}.expression.json"
        _write_json(exp_path, exp_chunk_obj)

        # sanity checks (minimal)
        has_mouth_hold0 = any(int(fr.get("t_ms", -1)) == 0 for fr in mouth_rel)
        has_exp_hold0 = any(int(ev.get("t_ms", -1)) == 0 and str(ev.get("source", "")) == "hold" for ev in exp_events)

        bundle_chunks.append(
            {
                "i": i,
                "chunk_start_ms": cs,
                "mouth_chunk": str(mouth_path).replace("\\", "/"),
                "expression_chunk": str(exp_path).replace("\\", "/"),
                "sanity": {
                    "mouth_has_t0": bool(has_mouth_hold0),
                    "expression_has_hold_t0": bool(has_exp_hold0),
                    "mouth_frames_n": len(mouth_rel),
                    "expression_events_n": len(exp_events),
                    "missing_mouth_frames_from_dense_grid": int(missing),
                },
            }
        )

    bundle = {
        "schema": "m3.chunk_bundle_manifest.v1",
        "cfg": {
            "step_ms": step_ms,
            "chunk_ms": chunk_ms,
            "overlap_ms": overlap_ms,
            "drop_overlap": bool(args.drop_overlap),
            "stride_ms": chunk_ms - overlap_ms,
        },
        "inputs": {
            "raw_manifest": str(raw_manifest_path).replace("\\", "/"),
            "mouth_session": str(mouth_session_path).replace("\\", "/"),
            "expression_chunks": str(exp_chunks_path).replace("\\", "/"),
        },
        "chunks_n": len(bundle_chunks),
        "chunks": bundle_chunks,
        "summary": {
            "total_missing_mouth_frames_from_dense_grid": int(total_missing_mouth_frames),
        },
    }

    out_manifest = out_dir / "m3_chunk_bundle.manifest.json"
    _write_json(out_manifest, bundle)

    print("[build_m3_chunk_bundle_manifest][OK]")
    print(" out_manifest:", out_manifest.as_posix())
    print(" chunks_dir  :", chunks_out_dir.as_posix())
    print(" chunks_n    :", len(bundle_chunks))
    print(" drop_overlap:", bool(args.drop_overlap))
    print(" total_missing_mouth_frames_from_dense_grid:", total_missing_mouth_frames)


if __name__ == "__main__":
    main()
