# src/m3p/live/expression_chunk_postprocess.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class BlinkConfig:
    enabled: bool = False
    period_ms: int = 15000
    dur_ms: int = 120
    expression: str = "blink"


def _round_to_step(t_ms: int, step_ms: int) -> int:
    # 例: 53ms -> 40ms, 59ms -> 80ms のように丸める
    return int(round(t_ms / step_ms) * step_ms)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _infer_state_at(events_sorted: List[Dict], t_ms: int, default_expr: str) -> str:
    """
    events_sorted: t_ms昇順（同一t_msは入力順）
    t_ms 時点(<=t_ms)での最後の expression を返す
    """
    cur = default_expr
    for ev in events_sorted:
        if int(ev["t_ms"]) > t_ms:
            break
        cur = str(ev["expression"])
    return cur


def build_expression_chunk_events(
    *,
    events_abs: List[Dict],
    chunk_start_ms: int,
    chunk_len_ms: int,
    step_ms: int,
    default_expression: str,
    prev_expression: Optional[str],
    blink: BlinkConfig,
) -> List[Dict]:
    """
    - 入力: 絶対時刻events（例: tool_call由来）
    - 出力: チャンク内相対時刻events（t_ms=0..chunk_len-step）
      かつ t_ms=0 は prev_expression で "hold 1点" を入れる（予約スロット）
    - blink.enabled の場合、チャンク内にルールベースblinkを混在
    """
    if chunk_len_ms <= 0:
        raise ValueError("chunk_len_ms must be > 0")
    if step_ms <= 0:
        raise ValueError("step_ms must be > 0")

    chunk_end_ms = chunk_start_ms + chunk_len_ms
    last_frame_t = chunk_len_ms - step_ms
    if last_frame_t < 0:
        raise ValueError("chunk_len_ms must be >= step_ms")

    # 1) チャンク範囲内のイベントだけ抽出（絶対時刻のまま）
    #    [start, end) に入るもの
    kept_abs: List[Dict] = []
    for ev in events_abs:
        t = int(ev.get("t_ms", 0))
        if chunk_start_ms <= t < chunk_end_ms:
            kept_abs.append(ev)

    kept_abs.sort(key=lambda d: int(d.get("t_ms", 0)))

    # 2) 相対時刻に変換 & step_msへ丸め（0..last_frame_tへクランプ）
    rel_events: List[Dict] = []
    for ev in kept_abs:
        t_abs = int(ev["t_ms"])
        t_rel = t_abs - chunk_start_ms
        t_rel = _round_to_step(t_rel, step_ms)
        t_rel = _clamp(t_rel, 0, last_frame_t)
        rel_events.append(
            {
                "t_ms": t_rel,
                "expression": str(ev.get("expression", default_expression)),
                "source": str(ev.get("source", "event")),
            }
        )

    rel_events.sort(key=lambda d: int(d["t_ms"]))

    # 3) チャンク先頭 hold（t_ms=0 予約）
    hold_expr = prev_expression if (prev_expression is not None and prev_expression != "") else default_expression
    out: List[Dict] = [{"t_ms": 0, "expression": hold_expr, "source": "chunk_hold"}]

    # 4) blink挿入（必要なら）
    if blink.enabled and blink.period_ms > 0 and blink.dur_ms > 0:
        # チャンク内の blink start 時刻（絶対基準で period境界を作る）
        # 例: period=15000ms なら 0,15000,30000... を基準に、
        # チャンクに重なるものだけ入れる。
        first_k = (chunk_start_ms // blink.period_ms) * blink.period_ms
        if first_k < chunk_start_ms:
            first_k += blink.period_ms

        blink_events: List[Dict] = []
        k = first_k
        while k < chunk_end_ms:
            # blink開始/終了（絶対）
            b0_abs = k
            b1_abs = k + blink.dur_ms

            # 相対へ
            b0 = _clamp(_round_to_step(b0_abs - chunk_start_ms, step_ms), 0, last_frame_t)
            b1 = _clamp(_round_to_step(b1_abs - chunk_start_ms, step_ms), 0, last_frame_t)

            # “blink開始時点”の本来表情（blinkで上書きする前）
            # out+rel_events を合流した “ベース” から推定
            base_for_state = (out + rel_events)
            base_for_state.sort(key=lambda d: int(d["t_ms"]))
            restore_expr = _infer_state_at(base_for_state, b0, default_expression)

            blink_events.append({"t_ms": b0, "expression": blink.expression, "source": "blink"})
            blink_events.append({"t_ms": b1, "expression": restore_expr, "source": "blink_end_restore"})

            k += blink.period_ms

        # base rel events に混ぜる（同一t_msは後勝ち＝後から append でOK）
        rel_events = rel_events + blink_events
        rel_events.sort(key=lambda d: int(d["t_ms"]))

    # 5) holdの後にイベントを足す（同一t_msは後勝ち）
    out.extend(rel_events)
    return out
