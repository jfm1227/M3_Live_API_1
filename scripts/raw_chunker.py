#!/usr/bin/env python3
# scripts/raw_chunker.py
#
# 時間基準のみで raw(40ms格子) を chunk 化し、manifest と chunk raw を出力する。
# - ここでは「chunk raw を作るだけ」(knn/i挿入は別)。
# - dup=0/gap=0 の担保は「join 時に overlap 先頭を捨てる」運用で達成する。
#
# 出力:
#   <out_dir>/manifest.json
#   <out_dir>/chunks/chunk_0000.raw.json ...
#
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class ChunkConfig:
    step_ms: int = 40
    chunk_ms: int = 400
    overlap_ms: int = 200

    @property
    def stride_ms(self) -> int:
        return int(self.chunk_ms - self.overlap_ms)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _index_frames_by_t(raw: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], int, int]:
    frames = raw.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise SystemExit("[raw_chunker][ERROR] raw['frames'] is empty or invalid")

    by_t: Dict[int, Dict[str, Any]] = {}
    for fr in frames:
        if "t_ms" not in fr:
            continue
        t = int(fr["t_ms"])
        by_t[t] = fr

    if not by_t:
        raise SystemExit("[raw_chunker][ERROR] no valid frames with t_ms")

    t_min = min(by_t.keys())
    t_max = max(by_t.keys())
    return by_t, t_min, t_max


def build_chunk_starts(t_min: int, t_max: int, cfg: ChunkConfig) -> List[int]:
    # 例: stride=200ms で t_min..t_max を覆う chunk_start_ms の列を作る
    starts: List[int] = []
    t = int(t_min)
    while t <= int(t_max):
        starts.append(int(t))
        t += cfg.stride_ms
    return starts


def slice_chunk_frames_abs(by_t: Dict[int, Dict[str, Any]], chunk_start_ms: int, cfg: ChunkConfig) -> List[Dict[str, Any]]:
    # chunk窓 [cs, cs+chunk_ms) の t_ms を step_ms 格子で列挙し、
    # raw に存在するフレームだけ集める（t_ms は絶対時刻のまま）
    cs = int(chunk_start_ms)
    out: List[Dict[str, Any]] = []
    for t in range(cs, cs + int(cfg.chunk_ms), int(cfg.step_ms)):
        fr = by_t.get(int(t))
        if fr is not None:
            out.append(fr)
    return out


def build_manifest(
    *,
    raw_path: Path,
    out_dir: Path,
    cfg: ChunkConfig,
    t_min: int,
    t_max: int,
    chunk_starts: List[int],
) -> Dict[str, Any]:
    chunks = []
    chunks_dir = out_dir / "chunks"
    for i, cs in enumerate(chunk_starts):
        p = chunks_dir / f"chunk_{i:04d}.raw.json"
        chunks.append(
            {
                "i": int(i),
                "chunk_start_ms": int(cs),
                "path": str(p).replace("\\", "/"),
            }
        )

    return {
        "schema": "m3.raw_chunk_manifest.v1",
        "raw_path": str(raw_path).replace("\\", "/"),
        "t_min_ms": int(t_min),
        "t_max_ms": int(t_max),
        "chunk_cfg": asdict(cfg) | {"stride_ms": int(cfg.stride_ms)},
        "chunks": chunks,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="input raw json (mouth_timeline.formant.raw.json)")
    ap.add_argument("--out_dir", required=True, help="output dir (manifest.json + chunks/)")
    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--chunk_ms", type=int, default=400)
    ap.add_argument("--overlap_ms", type=int, default=200)
    args = ap.parse_args()

    cfg = ChunkConfig(step_ms=int(args.step_ms), chunk_ms=int(args.chunk_ms), overlap_ms=int(args.overlap_ms))
    if cfg.step_ms <= 0:
        raise SystemExit("[raw_chunker][ERROR] step_ms must be > 0")
    if cfg.chunk_ms <= 0:
        raise SystemExit("[raw_chunker][ERROR] chunk_ms must be > 0")
    if cfg.overlap_ms < 0 or cfg.overlap_ms >= cfg.chunk_ms:
        raise SystemExit("[raw_chunker][ERROR] overlap_ms must satisfy 0 <= overlap_ms < chunk_ms")

    raw_path = Path(args.raw)
    out_dir = Path(args.out_dir)

    raw = _load_json(raw_path)
    by_t, t_min, t_max = _index_frames_by_t(raw)

    chunk_starts = build_chunk_starts(t_min, t_max, cfg)

    # chunk raw 書き出し
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    for i, cs in enumerate(chunk_starts):
        chunk_frames = slice_chunk_frames_abs(by_t, cs, cfg)
        out_obj = {
            "format": raw.get("format", "mouth_timeline.formant.raw.v0"),
            "step_ms": int(cfg.step_ms),
            "frames": chunk_frames,
        }
        _write_json(chunks_dir / f"chunk_{i:04d}.raw.json", out_obj)

    manifest = build_manifest(
        raw_path=raw_path,
        out_dir=out_dir,
        cfg=cfg,
        t_min=t_min,
        t_max=t_max,
        chunk_starts=chunk_starts,
    )
    _write_json(out_dir / "manifest.json", manifest)

    print("[raw_chunker][OK] wrote:")
    print("  manifest:", (out_dir / "manifest.json").as_posix())
    print("  chunks  :", (out_dir / "chunks").as_posix())
    print("  chunks_n:", len(chunk_starts))
    print("  t_min_ms:", t_min, "t_max_ms:", t_max)
    print("  cfg:", asdict(cfg) | {"stride_ms": cfg.stride_ms})


if __name__ == "__main__":
    main()
