# scripts/mouth_insert_intermediate_i.py

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def load_mouth_timeline(path: Path) -> Dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"failed to read JSON from {path}: {e}")
    if "frames" not in data or not isinstance(data["frames"], list):
        raise RuntimeError(f"invalid mouth_timeline format: no 'frames' list in {path}")
    return data


def summarize_counter(counter: Counter) -> str:
    total = sum(counter.values()) or 1
    lines = [f"total frames: {total}"]
    for mid in sorted(counter.keys()):
        cnt = counter[mid]
        ratio = cnt / total * 100.0
        lines.append(f"  mouth_id={mid}: {cnt:5d} frames ({ratio:5.1f}%)")
    return "\n".join(lines)


def collect_runs(frames: List[Dict]) -> List[Tuple[int, int, int]]:
    """
    同じ mouth_id が連続している区間を (start_idx, end_idx, mouth_id) で返す。
    end_idx は「含まない」インデックス（Pythonのスライスと同じ）。
    """
    runs: List[Tuple[int, int, int]] = []
    if not frames:
        return runs

    cur_id = int(frames[0].get("mouth_id", 0))
    start = 0

    for i in range(1, len(frames)):
        mid = int(frames[i].get("mouth_id", 0))
        if mid != cur_id:
            runs.append((start, i, cur_id))
            start = i
            cur_id = mid

    runs.append((start, len(frames), cur_id))
    return runs


def insert_intermediate_i(
    frames: List[Dict],
    step_ms: int,
    min_run_ms: int = 160,
    interval_ms: int = 160,
    max_i_per_run: int = 3,
) -> int:
    """
    連続区間が一定以上長い mouth_id (1,3,4,5) に対して、
    中間口形 i (mouth_id=2) をフレーム単位で挿入する（書き換える）。
    - mouth_id=0 はサイレンス候補なので触らない。
    - mouth_id=2 は既に中間形なので対象外。
    戻り値は、mouth_id を 2 に書き換えたフレーム数。
    """
    if step_ms <= 0:
        raise ValueError(f"step_ms must be positive, got {step_ms}")

    runs = collect_runs(frames)
    changed = 0

    for start, end, mid in runs:
        # 対象外の mouth_id はスキップ
        if mid == 0 or mid == 2:
            continue

        length = end - start
        if length <= 1:
            continue

        run_ms = length * step_ms
        if run_ms < min_run_ms:
            continue

        # この run に挿入する i の個数を決める
        # だいたい interval_ms ごとに 1個、ただし上限 max_i_per_run
        approx_slots = max(run_ms // interval_ms, 1)
        num_i = int(min(max(approx_slots, 1), max_i_per_run))

        # 例えば length=10, num_i=2 なら、
        # (1/3, 2/3) あたりの位置に i を入れるイメージ
        for j in range(num_i):
            idx = start + ( (j + 1) * length // (num_i + 1) )
            # 念のため範囲チェック
            if idx < start or idx >= end:
                continue

            # mouth_id=0/2 は守る（理論上ここには来ないはずだが安全策）
            cur = int(frames[idx].get("mouth_id", 0))
            if cur == 0 or cur == 2:
                continue

            frames[idx]["mouth_id"] = 2
            changed += 1

    return changed


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Insert intermediate mouth shape 'i' (mouth_id=2) "
            "into long runs of the same mouth_id in a mouth_timeline.json."
        ),
    )
    ap.add_argument(
        "input",
        help="path to input mouth_timeline.json",
    )
    ap.add_argument(
        "output",
        help="path to output mouth_timeline.json (with intermediate i inserted)",
    )
    ap.add_argument(
        "--min_run_ms",
        type=int,
        default=160,
        help=(
            "minimum run length in ms to consider inserting 'i' (default: 160ms). "
            "短すぎる区間には挿入しない。"
        ),
    )
    ap.add_argument(
        "--interval_ms",
        type=int,
        default=160,
        help=(
            "target interval in ms between 'i' insertions inside a run (default: 160ms). "
            "長い区間では複数回 i を挿入する。"
        ),
    )
    ap.add_argument(
        "--max_i_per_run",
        type=int,
        default=3,
        help="maximum number of 'i' insertions per run (default: 3).",
    )

    args = ap.parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    data = load_mouth_timeline(in_path)
    frames = data.get("frames", [])
    step_ms = int(data.get("step_ms", 40))

    before_counter = Counter(int(f.get("mouth_id", 0)) for f in frames)

    changed = insert_intermediate_i(
        frames,
        step_ms=step_ms,
        min_run_ms=args.min_run_ms,
        interval_ms=args.interval_ms,
        max_i_per_run=args.max_i_per_run,
    )

    after_counter = Counter(int(f.get("mouth_id", 0)) for f in frames)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[mouth_insert_intermediate_i] input:  {in_path}")
    print(f"[mouth_insert_intermediate_i] output: {out_path}")
    print(f"[mouth_insert_intermediate_i] step_ms={step_ms}")
    print(f"[mouth_insert_intermediate_i] changed frames -> mouth_id=2: {changed}")
    print("\n[before]")
    print(summarize_counter(before_counter))
    print("\n[after]")
    print(summarize_counter(after_counter))


if __name__ == "__main__":
    main()
