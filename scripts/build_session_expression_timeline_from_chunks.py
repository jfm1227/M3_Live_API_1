#!/usr/bin/env python3
# scripts/build_session_expression_timeline_from_chunks.py
#
# expression_chunks.v1.json -> session_expression_timeline_v0.1 互換（M0が読む形）
#
# 出力:
# {
#   "schema_version": "session_expression_timeline_v0.1",
#   "session_id": "...",
#   "step_ms": 40,
#   "timeline": [ {t_ms, expression, source, ...}, ... ],
#   "meta": {...}
# }

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Tuple

def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def save_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--session_id", default="sess_from_chunks")
    ap.add_argument("--drop_hold", action="store_true", help="drop source=='hold' events")
    args = ap.parse_args()

    in_path = Path(args.in_chunks)
    out_path = Path(args.out)

    obj = load_json(in_path)
    step_ms = int(obj.get("step_ms", 40))
    chunks = obj.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        raise SystemExit("[ERROR] chunks empty")

    timeline: List[Dict[str, Any]] = []

    for ch in chunks:
        cs = int(ch.get("chunk_start_ms", 0))
        events = ch.get("events", [])
        if not isinstance(events, list):
            continue
        for ev in events:
            if args.drop_hold and str(ev.get("source", "")) == "hold":
                continue
            rel_t = int(ev.get("t_ms", 0))
            abs_t = cs + rel_t
            out_ev = dict(ev)
            out_ev["t_ms"] = abs_t
            timeline.append(out_ev)

    # sort + de-dup exact duplicates (same t_ms, expression, source)
    timeline.sort(key=lambda e: (int(e.get("t_ms", 0)), str(e.get("expression", "")), str(e.get("source", ""))))
    dedup: List[Dict[str, Any]] = []
    prev: Tuple[int,str,str] | None = None
    for e in timeline:
        k = (int(e.get("t_ms", 0)), str(e.get("expression","")), str(e.get("source","")))
        if k == prev:
            continue
        dedup.append(e)
        prev = k

    out_obj = {
        "schema_version": "session_expression_timeline_v0.1",
        "session_id": args.session_id,
        "step_ms": step_ms,
        "timeline": dedup,
        "meta": {
            "source": str(in_path).replace("\\", "/"),
            "events_n": len(dedup),
            "drop_hold": bool(args.drop_hold),
        }
    }

    save_json(out_path, out_obj)
    print("[build_session_expression_timeline_from_chunks][OK]")
    print(" in :", in_path.as_posix())
    print(" out:", out_path.as_posix())
    print(" step_ms:", step_ms)
    print(" timeline:", len(dedup))

if __name__ == "__main__":
    main()
