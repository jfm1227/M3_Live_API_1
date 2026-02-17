# src/m3p/live/expression_chunk_postprocess.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _quantize_ms(t_ms: int, step_ms: int) -> int:
    if step_ms <= 0:
        return t_ms
    return int(round(t_ms / step_ms) * step_ms)


def _sort_events_abs(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(events, key=lambda e: int(e.get("t_ms", 0)))


def _dedup_same_time_keep_last(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    同時刻が複数ある場合は「後勝ち」。
    """
    evs = _sort_events_abs(events)
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(evs):
        t = int(evs[i].get("t_ms", 0))
        j = i
        last = evs[i]
        while j < len(evs) and int(evs[j].get("t_ms", 0)) == t:
            last = evs[j]
            j += 1
        out.append(last)
        i = j
    return out


def _drop_consecutive_same_expression(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    last_expr: Optional[str] = None
    for ev in events:
        expr = str(ev.get("expression", ""))
        if expr == last_expr:
            continue
        out.append(ev)
        last_expr = expr
    return out


def _expression_at_time(events_abs: List[Dict[str, Any]], t_ms: int, default_expr: str) -> str:
    """
    events_abs は t_ms昇順（同時刻は後勝ち済）を想定。
    t_ms 時点の表情（直前のイベントの expression）を返す。
    """
    cur = default_expr
    for ev in events_abs:
        if int(ev.get("t_ms", 0)) > t_ms:
            break
        cur = str(ev.get("expression", default_expr))
    return cur


@dataclass
class BlinkConfig:
    enabled: bool = True
    blink_interval_ms: int = 10000  # 10秒に1回
    blink_duration_ms: int = 160    # 160ms (40ms stepなら4フレーム)
    blink_expression: str = "blink"


@dataclass
class ChunkConfig:
    step_ms: int = 40
    chunk_ms: int = 400
    overlap_ms: int = 200  # 先頭overlap区間は捨てる運用（mouth側と整合）
    default_expression: str = "normal"


def insert_periodic_blinks_abs(
    base_events_abs: List[Dict[str, Any]],
    *,
    duration_ms: int,
    step_ms: int,
    default_expression: str,
    blink_cfg: BlinkConfig,
) -> List[Dict[str, Any]]:
    """
    sparseな表情イベント（絶対t_ms）に、定期blinkを追加する。
    blinkは [t, t+blink_duration_ms) の間 blink_expression にし、
    その後 直前表情に戻すイベントも追加する。
    """
    evs = _dedup_same_time_keep_last(base_events_abs)
    evs = _drop_consecutive_same_expression(evs)

    if not blink_cfg.enabled:
        return evs

    interval = int(blink_cfg.blink_interval_ms)
    dur = int(blink_cfg.blink_duration_ms)
    if interval <= 0 or dur <= 0:
        return evs

    out = list(evs)

    # 0ms起点で interval ごとに blink を入れる（t=0は避ける：initを壊しやすい）
    t = interval
    while t < duration_ms:
        t0 = _quantize_ms(t, step_ms)
        t1 = _quantize_ms(t + dur, step_ms)

        # t0時点の表情を取得（blink前に戻すため）
        prev_expr = _expression_at_time(_dedup_same_time_keep_last(out), t0, default_expression)

        # blink開始
        out.append({"t_ms": int(t0), "expression": str(blink_cfg.blink_expression), "source": "rule_blink"})
        # blink終了→元に戻す
        out.append({"t_ms": int(t1), "expression": str(prev_expr), "source": "rule_blink_end"})

        t += interval

    out = _dedup_same_time_keep_last(out)
    out = _drop_consecutive_same_expression(out)
    return out


def build_chunk_events_with_hold(
    events_abs: List[Dict[str, Any]],
    *,
    chunk_cfg: ChunkConfig,
    chunk_start_ms_list: List[int],
) -> List[Dict[str, Any]]:
    """
    絶対t_msの表情イベントから、chunkごとの相対t_msイベントを作る。
    - 各chunkの t_ms=0 に holdイベントを必ず入れる（前chunk末の表情）。
    - chunk内に存在するイベントは相対化して入れる。
    - 先頭overlap捨ては mouth側の raw結合でやる前提だが、
      expression側も同様の運用に合わせたい場合は、採用開始を overlap_ms にする。
      ここでは「chunk内イベントは全て採用」しつつ、境界だけholdで安定化する。
    """
    step_ms = int(chunk_cfg.step_ms)
    chunk_ms = int(chunk_cfg.chunk_ms)
    default_expr = str(chunk_cfg.default_expression)

    evs = _dedup_same_time_keep_last(events_abs)
    evs = _drop_consecutive_same_expression(evs)

    chunks_out: List[Dict[str, Any]] = []
    last_expr_prev_chunk = default_expr

    for cs in chunk_start_ms_list:
        cs = int(cs)
        ce = cs + chunk_ms

        # chunk開始時点の表情（hold用）
        hold_expr = _expression_at_time(evs, cs, default_expr)

        # chunk内イベント抽出（cs < t < ce）
        rel_events: List[Dict[str, Any]] = []
        for ev in evs:
            t = int(ev.get("t_ms", 0))
            if t < cs:
                continue
            if t >= ce:
                break
            # t==cs は holdで上書きする（後勝ちにしたいので、ここでは入れてもOKだが重複しやすい）
            if t == cs:
                continue
            rel_t = _quantize_ms(t - cs, step_ms)
            rel_events.append({"t_ms": int(rel_t), "expression": str(ev.get("expression", default_expr)), "source": ev.get("source", "event")})

        # 必ず先頭に hold を入れる（t_ms=0）
        # 仕様：chunk末→次chunk先頭で上書き。ここでは「このchunkのcs時点の表情」を確実に置く。
        # もし「前chunk末の表情で上書き」を厳密にやりたい場合も、
        # cs時点の表情=前chunk末の表情（イベントが無ければ）なので一致する。
        rel_events.append({"t_ms": 0, "expression": str(hold_expr), "source": "hold"})

        # t_msソート＆同時刻は後勝ち＆連続同値削除
        rel_events = _dedup_same_time_keep_last(rel_events)
        rel_events = _drop_consecutive_same_expression(rel_events)

        chunks_out.append(
            {
                "chunk_start_ms": cs,
                "step_ms": step_ms,
                "events": rel_events,
                "meta": {
                    "default_expression": default_expr,
                    "hold_expression": str(hold_expr),
                    "prev_chunk_last_expression": str(last_expr_prev_chunk),
                },
            }
        )

        # 次chunk用に「このchunk末時点の表情」を更新
        last_expr_prev_chunk = _expression_at_time(evs, ce - step_ms, default_expr)

    return chunks_out


def postprocess_expression_for_chunks(
    base_events_abs: List[Dict[str, Any]],
    *,
    duration_ms: int,
    chunk_start_ms_list: List[int],
    chunk_cfg: ChunkConfig,
    blink_cfg: BlinkConfig,
) -> Dict[str, Any]:
    """
    まとめ関数：
    1) base_events_abs（tool_call等）に periodic blink を追加（絶対t_ms）
    2) chunkごとの相対イベント（hold付き）を生成
    """
    evs_with_blink = insert_periodic_blinks_abs(
        base_events_abs,
        duration_ms=duration_ms,
        step_ms=chunk_cfg.step_ms,
        default_expression=chunk_cfg.default_expression,
        blink_cfg=blink_cfg,
    )

    chunks = build_chunk_events_with_hold(
        evs_with_blink,
        chunk_cfg=chunk_cfg,
        chunk_start_ms_list=chunk_start_ms_list,
    )

    return {
        "version": "expression_chunk_postprocess_v1",
        "step_ms": int(chunk_cfg.step_ms),
        "duration_ms": int(duration_ms),
        "base_events_abs": _drop_consecutive_same_expression(_dedup_same_time_keep_last(base_events_abs)),
        "events_abs": evs_with_blink,
        "chunks": chunks,
        "cfg": {
            "chunk": {
                "chunk_ms": int(chunk_cfg.chunk_ms),
                "overlap_ms": int(chunk_cfg.overlap_ms),
                "default_expression": str(chunk_cfg.default_expression),
            },
            "blink": {
                "enabled": bool(blink_cfg.enabled),
                "blink_interval_ms": int(blink_cfg.blink_interval_ms),
                "blink_duration_ms": int(blink_cfg.blink_duration_ms),
                "blink_expression": str(blink_cfg.blink_expression),
            },
        },
    }
