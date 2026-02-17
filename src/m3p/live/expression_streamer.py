# src/m3p/live/expression_streamer.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _quantize_ms(t_ms: int, step_ms: int) -> int:
    if step_ms <= 0:
        return t_ms
    return int(round(t_ms / step_ms) * step_ms)


def _emo_id_to_str(emo_id: Union[str, int]) -> str:
    """
    Accept both "1_2" (str) and 12 (int-like) inputs.
    Canonical key is str.

    - If emo_id is already str: keep as-is (strip spaces).
    - If int: convert to str.
    """
    if isinstance(emo_id, str):
        return emo_id.strip()
    return str(int(emo_id))


@dataclass
class ExpressionStreamerConfig:
    step_ms: int = 40
    default_expression: str = "normal"

    # emo_id(str) -> expression(str)
    # 例: {"1_0":"normal", "1_1":"smile", "2_0":"angry"}
    emo_map: Dict[str, str] = None  # type: ignore

    # 受け取った emo_id がここに無い場合の扱い
    # Noneなら default_expression に落とす
    unknown_emo_to: Optional[str] = None


# --- 修正点・差分反映箇所 ---
EMO_MAP = {
    "1_0": "normal",
    "1_1": "smile",
    "1_2": "sad",
    "2_0": "angry",
    # 追加・変更はここだけ
}

cfg = ExpressionStreamerConfig(
    step_ms=40,
    default_expression="normal",
    emo_map=EMO_MAP,
    unknown_emo_to="normal",  # 未定義emo_idは必ずここに落とす
)
# --------------------------


class ExpressionStreamer:
    """
    tool_call (emo_id) を受け取り、expression_timeline を sparse に生成する.

    出力は「イベント配列」を基本とする（M0は list / dict.timeline / dict.frames を受けられる想定）。
    形式例（contracts案）:
      [
        {"t_ms": 0, "expression": "normal", "source": "init"},
        {"t_ms": 1200, "expression": "blink", "source": "rule"},
      ]
    """

    def __init__(self, cfg: ExpressionStreamerConfig):
        if cfg.emo_map is None:
            cfg.emo_map = {}
        # 念のため key/value を str 化
        cfg.emo_map = {str(k).strip(): str(v) for k, v in cfg.emo_map.items()}
        self.cfg = cfg
        self.events: List[Dict[str, Any]] = []
        self._last_expr: Optional[str] = None

    def init_at(self, t_ms: int = 0, expr: Optional[str] = None) -> None:
        """最初の基準表情イベントを入れる（任意だが入れるの推奨）"""
        if expr is None:
            expr = self.cfg.default_expression
        t_q = _quantize_ms(int(t_ms), self.cfg.step_ms)
        self._append_if_changed(t_q, expr, source="init")

    def on_emo_id(
        self,
        emo_id: Union[str, int],
        t_ms: int,
        *,
        source: str = "tool_call",
    ) -> Optional[Dict[str, Any]]:
        """tool_call 由来の emo_id を受け取り、必要ならイベントを追加"""
        key = _emo_id_to_str(emo_id)
        expr = self.cfg.emo_map.get(key)

        if expr is None:
            if self.cfg.unknown_emo_to is None:
                expr = self.cfg.default_expression
            else:
                expr = self.cfg.unknown_emo_to

        t_q = _quantize_ms(int(t_ms), self.cfg.step_ms)
        return self._append_if_changed(t_q, expr, source=source)

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