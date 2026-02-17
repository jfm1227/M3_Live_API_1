#!/usr/bin/env python3
# scripts/run_chunked_knn_regression.py
#
# STEP-A〜B を 1コマンド回帰で固定する:
#   raw(full) -> raw_chunker(manifest+chunk raw)
#            -> per-chunk: tools/knn_from_formant_raw_to_mouth_timeline.py
#            -> join (drop first overlap part for non-first chunk) => session_knn
#            -> scripts/mouth_insert_intermediate_i.py => session_knn_i
#            -> diff vs golden (t_ms/mouth_id)
#
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _run(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def _index_mouth(frames: List[Dict[str, Any]]) -> Dict[int, int]:
    by_t: Dict[int, int] = {}
    for fr in frames:
        if "t_ms" not in fr:
            continue
        t = int(fr["t_ms"])
        mid = int(fr.get("mouth_id", 0))
        by_t[t] = mid
    return by_t


def _diff_mouth(got: Dict[int, int], exp: Dict[int, int]) -> Tuple[int, int, int, List[Tuple[int, int, int]]]:
    # returns (missing, extra, mismatch, samples)
    got_ts = set(got.keys())
    exp_ts = set(exp.keys())
    missing_ts = sorted(exp_ts - got_ts)
    extra_ts = sorted(got_ts - exp_ts)

    mismatches: List[Tuple[int, int, int]] = []
    for t in sorted(got_ts & exp_ts):
        if got[t] != exp[t]:
            mismatches.append((t, got[t], exp[t]))

    samples = mismatches[:20]
    return len(missing_ts), len(extra_ts), len(mismatches), samples


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--raw", required=True, help="input raw (mouth_timeline.formant.raw.json)")
    ap.add_argument("--gt_glob", default="data/knn_db/*.f1f2.json")
    ap.add_argument("--golden", required=True, help="golden SSOT (mouth_timeline.knn.i.json)")
    ap.add_argument("--out_dir", required=True, help="output dir for regression artifacts")

    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--chunk_ms", type=int, default=400)
    ap.add_argument("--overlap_ms", type=int, default=200)

    # kNN params (tools/knn_from_formant_raw_to_mouth_timeline.py)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--fallback_id_active", type=int, default=2)
    ap.add_argument("--min_conf_ratio", type=float, default=1.15)

    # i-insert params (scripts/mouth_insert_intermediate_i.py)
    ap.add_argument("--min_run_ms", type=int, default=160)
    ap.add_argument("--interval_ms", type=int, default=160)
    ap.add_argument("--max_i_per_run", type=int, default=3)

    # audio field to write into kNN output json (optional; keep stable)
    ap.add_argument("--audio", default="live")

    args = ap.parse_args()

    step_ms = int(args.step_ms)
    chunk_ms = int(args.chunk_ms)
    overlap_ms = int(args.overlap_ms)
    if overlap_ms < 0 or overlap_ms >= chunk_ms:
        raise SystemExit("[ERROR] overlap_ms must satisfy 0 <= overlap_ms < chunk_ms")
    stride_ms = chunk_ms - overlap_ms

    raw_path = Path(args.raw)
    golden_path = Path(args.golden)
    out_dir = Path(args.out_dir)

    # 0) load raw to know t_min/t_max + for sanity
    raw = _load_json(raw_path)
    raw_frames = raw.get("frames", [])
    if not isinstance(raw_frames, list) or not raw_frames:
        raise SystemExit("[ERROR] raw.frames empty")
    t_list = sorted(int(fr["t_ms"]) for fr in raw_frames if "t_ms" in fr)
    if not t_list:
        raise SystemExit("[ERROR] raw has no t_ms")
    t_min, t_max = t_list[0], t_list[-1]

    # 1) raw_chunker: write manifest + chunk raw files
    chunk_root = out_dir / "raw_chunks"
    _run([
        sys.executable, "scripts/raw_chunker.py",
        "--raw", str(raw_path),
        "--out_dir", str(chunk_root),
        "--step_ms", str(step_ms),
        "--chunk_ms", str(chunk_ms),
        "--overlap_ms", str(overlap_ms),
    ])
    manifest = _load_json(chunk_root / "manifest.json")
    chunks = manifest.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        raise SystemExit("[ERROR] manifest.chunks empty")

    # 2) per-chunk: raw -> knn (NO i-insert here)
    knn_chunks_dir = out_dir / "knn_chunks"
    knn_chunks_dir.mkdir(parents=True, exist_ok=True)

    for ch in chunks:
        i = int(ch["i"])
        cs = int(ch["chunk_start_ms"])
        raw_chunk_path = Path(ch["path"])
        out_knn = knn_chunks_dir / f"chunk_{i:04d}.mouth_timeline.knn.json"

        _run([
            sys.executable, "tools/knn_from_formant_raw_to_mouth_timeline.py",
            "--raw", str(raw_chunk_path),
            "--gt_glob", str(args.gt_glob),
            "--out", str(out_knn),
            "--audio", str(args.audio),
            "--step_ms", str(step_ms),
            "--k", str(int(args.k)),
            "--fallback_id_active", str(int(args.fallback_id_active)),
            "--min_conf_ratio", str(float(args.min_conf_ratio)),
        ])

        # keep tiny index json for debug (optional)
        _write_json(knn_chunks_dir / f"chunk_{i:04d}.meta.json", {"i": i, "chunk_start_ms": cs, "raw": str(raw_chunk_path)})

    # 3) join knn chunks with overlap-drop to ensure dup=0
    joined_knn_path = out_dir / "joined.mouth_timeline.knn.json"

    joined_frames: List[Dict[str, Any]] = []
    for ch in chunks:
        i = int(ch["i"])
        cs = int(ch["chunk_start_ms"])
        ce_keep = cs + overlap_ms  # non-first: drop t < cs+overlap
        knn_path = knn_chunks_dir / f"chunk_{i:04d}.mouth_timeline.knn.json"
        knn_obj = _load_json(knn_path)
        frs = knn_obj.get("frames", [])
        if not isinstance(frs, list):
            raise SystemExit(f"[ERROR] invalid knn chunk frames: {knn_path}")

        if i == 0:
            use = frs
        else:
            use = [fr for fr in frs if int(fr.get("t_ms", -1)) >= ce_keep]

        joined_frames.extend(use)

    joined_frames.sort(key=lambda fr: int(fr.get("t_ms", 0)))
    joined_obj = {
        "audio": str(args.audio),
        "step_ms": step_ms,
        "frames": joined_frames,
        "meta": {
            "schema": "m3.joined_knn.v1",
            "raw": str(raw_path).replace("\\", "/"),
            "t_min_ms": int(t_min),
            "t_max_ms": int(t_max),
            "chunk_ms": int(chunk_ms),
            "overlap_ms": int(overlap_ms),
            "stride_ms": int(stride_ms),
            "gt_glob": str(args.gt_glob),
            "k": int(args.k),
            "fallback_id_active": int(args.fallback_id_active),
            "min_conf_ratio": float(args.min_conf_ratio),
        },
    }
    _write_json(joined_knn_path, joined_obj)
    print("\n[join][OK] wrote:", joined_knn_path.as_posix(), "frames:", len(joined_frames))

    # 4) i-insert on the joined timeline
    joined_knn_i_path = out_dir / "joined.mouth_timeline.knn.i.json"
    _run([
        sys.executable, "scripts/mouth_insert_intermediate_i.py",
        str(joined_knn_path),
        str(joined_knn_i_path),
        "--min_run_ms", str(int(args.min_run_ms)),
        "--interval_ms", str(int(args.interval_ms)),
        "--max_i_per_run", str(int(args.max_i_per_run)),
    ])

    # 5) diff vs golden
    got = _load_json(joined_knn_i_path)
    exp = _load_json(golden_path)

    got_step = int(got.get("step_ms", step_ms))
    exp_step = int(exp.get("step_ms", step_ms))
    if got_step != exp_step:
        raise SystemExit(f"[DIFF][ERROR] step_ms mismatch: got={got_step} exp={exp_step}")

    got_map = _index_mouth(got.get("frames", []))
    exp_map = _index_mouth(exp.get("frames", []))

    missing, extra, mismatch, samples = _diff_mouth(got_map, exp_map)

    print("\n===== chunked kNN regression diff =====")
    print("raw      :", str(raw_path))
    print("golden   :", str(golden_path))
    print("joined_i :", str(joined_knn_i_path))
    print(f"step_ms  : {step_ms}")
    print(f"chunk_ms : {chunk_ms} overlap_ms: {overlap_ms} stride_ms: {stride_ms}")
    print("--------------------------------------")
    print("frames(got):", len(got_map), "frames(exp):", len(exp_map))
    print("missing_ts :", missing)
    print("extra_ts   :", extra)
    print("mismatch   :", mismatch)
    if samples:
        print("\n[mismatch samples] t_ms got exp")
        for t, g, e in samples:
            print(f"  {t:6d}  {g}   {e}")

    if missing or extra or mismatch:
        raise SystemExit(2)

    print("\n[OK] chunked regression matched golden exactly.")


if __name__ == "__main__":
    main()
