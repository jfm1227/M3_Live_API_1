# src/m3p/live/expression_streamer.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _quantize_ms(t_ms: int, step_ms: int) -> int:
    if step_ms <= 0:
        return t_ms
    return int(round(t_ms / step_ms) * step_ms)


@dataclass
class ExpressionStreamerConfig:
    step_ms: int = 40
    default_expression: str = "normal"

    # emo_id(int) -> expression(str)
    # 例: {0:"normal", 1:"smile", 2:"angry", 3:"sad"}
    emo_map: Dict[int, str] = None  # type: ignore

    # 受け取った tool_call がここに無い場合の扱い
    # Noneなら default_expression に落とす
    unknown_emo_to: Optional[str] = None


class ExpressionStreamer:
    """
    tool_call (emo_id) を受け取り、expression_timeline を sparse に生成する。

    出力は「イベント配列」を基本とする（M0は list / dict.timeline / dict.frames を受けられる想定）。
    形式例（contracts案）:
      [
        {"t_ms": 0, "expression": "normal", "source": "init"},
        {"t_ms": 1200, "expression": "blink", "source": "tool_call"},
      ]
    """

    def __init__(self, cfg: ExpressionStreamerConfig):
        if cfg.emo_map is None:
            cfg.emo_map = {}
        self.cfg = cfg
        self.events: List[Dict[str, Any]] = []
        self._last_expr: Optional[str] = None

        # --- Step2 hookup (optional) ---
        # live_chunk_loop_step2.py 側の LiveInputs を受け取って、
        # tool_call確定ごとに push_expression_event() へ流す。
        self._live_inputs = None  # type: ignore
        self._base_offset_ms = 0

    def set_live_inputs(self, live_inputs: Any, *, base_offset_ms: int = 0) -> None:
        """
        Step2 用: LiveInputs を接続する。
        - live_inputs: live_chunk_loop_step2.LiveInputs 相当（push_expression_event を持つこと）
        - base_offset_ms: global_now_ms の基準オフセット（単調増加なら絶対値は不問）
        """
        self._live_inputs = live_inputs
        self._base_offset_ms = int(base_offset_ms)

    def init_at(self, t_ms: int = 0, expr: Optional[str] = None) -> None:
        """最初の基準表情イベントを入れる（任意だが入れるの推奨）"""
        if expr is None:
            expr = self.cfg.default_expression
        t_q = _quantize_ms(int(t_ms), self.cfg.step_ms)
        self._append_if_changed(t_q, expr, source="init")

    def on_emo_id(self, emo_id: int, t_ms: int, *, source: str = "tool_call") -> Optional[Dict[str, Any]]:
        """tool_call 由来の emo_id を受け取り、必要ならイベントを追加"""
        expr = self.cfg.emo_map.get(int(emo_id))
        if expr is None:
            if self.cfg.unknown_emo_to is None:
                expr = self.cfg.default_expression
            else:
                expr = self.cfg.unknown_emo_to

        t_q = _quantize_ms(int(t_ms), self.cfg.step_ms)
        ev = self._append_if_changed(t_q, expr, source=source)

        # Step2: connect to chunk loop (optional)
        if ev is not None and self._live_inputs is not None:
            try:
                t_abs = int(self._base_offset_ms) + int(t_q)
                self._live_inputs.push_expression_event(
                    t_abs_ms=t_abs,
                    expression=str(expr),
                    source=str(source),
                )
            except Exception:
                pass

        return ev

    def _append_if_changed(self, t_ms: int, expr: str, *, source: str) -> Optional[Dict[str, Any]]:
        if expr == self._last_expr:
            return None
        ev = {"t_ms": int(t_ms), "expression": str(expr), "source": str(source)}
        self.events.append(ev)
        self._last_expr = str(expr)
        return ev

    def to_json(self) -> List[Dict[str, Any]]:
        # 念のためソート（同時刻は後勝ち）
        evs = sorted(self.events, key=lambda x: int(x.get("t_ms", 0)))
        merged: List[Dict[str, Any]] = []
        i = 0
        while i < len(evs):
            t = int(evs[i]["t_ms"])
            same = []
            while i < len(evs) and int(evs[i]["t_ms"]) == t:
                same.append(evs[i])
                i += 1
            merged.append(same[-1])  # 後勝ち
        # 連続同値を削除
        out: List[Dict[str, Any]] = []
        last = None
        for ev in merged:
            if ev.get("expression") == last:
                continue
            out.append(ev)
            last = ev.get("expression")
        return out

    def write_json(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_json()

        tmp = out_path.with_suffix(out_path.suffix + f".tmp.{int(time.time()*1000)}")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_path)