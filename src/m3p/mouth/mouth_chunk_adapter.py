# src/m3p/mouth/mouth_chunk_adapter.py
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# TODO: あなたのrepoの実パスに合わせてここだけ直す
from m3p.live.mouth_streamer_oc import MouthStreamerOC, MouthOCConfig  # type: ignore


@dataclass
class ChunkSpec:
    step_ms: int = 40
    chunk_ms: int = 400
    overlap_ms: int = 200

    @property
    def stride_ms(self) -> int:
        return int(self.chunk_ms - self.overlap_ms)

    @property
    def frames_per_chunk(self) -> int:
        return int(self.chunk_ms // self.step_ms)

    @property
    def frames_overlap(self) -> int:
        return int(self.overlap_ms // self.step_ms)


class MouthChunkAdapter:
    """
    目的:
      PCM(stream) -> 400ms chunk(+overlap) -> raw frames(chunk local) を返す
    制約:
      - VAD/formant/kNN は再実装しない
      - raw生成は MouthStreamerOC(合格コード) を黒箱として呼ぶ
    """

    def __init__(
        self,
        spec: ChunkSpec = ChunkSpec(),
        input_sr_default: int = 24000,
        analysis_sr: int = 16000,
        vowel_mode: str = "formant",
        formant_window_ms: int = 200,
        formant_max_hz: int = 5500,
        open_id: int = 2,
        close_id: int = 0,
    ) -> None:
        self.spec = spec
        self.input_sr_default = int(input_sr_default)

        self._buf = bytearray()
        self._chunk_index = 0

        # MouthStreamerOC の設定は「既存合格raw生成」と同等の値に寄せる
        self._streamer_cfg = MouthOCConfig(
            step_ms=int(spec.step_ms),
            input_sr_default=int(input_sr_default),
            analysis_sr=int(analysis_sr),
            vowel_mode=str(vowel_mode),
            formant_window_ms=int(formant_window_ms),
            formant_max_hz=int(formant_max_hz),
            open_id=int(open_id),
            close_id=int(close_id),
        )

    def push_pcm16_mono(self, pcm: bytes, sr: Optional[int] = None) -> None:
        # sr は chunk 内処理時に MouthStreamerOC.push_pcm16_mono(input_sr=...) に渡す
        # ここではバッファリングだけ
        if not pcm:
            return
        self._buf.extend(pcm)

    def _bytes_for_ms(self, sr: int, ms: int) -> int:
        # PCM16 mono: 2 bytes/sample
        samples = int(round(sr * (ms / 1000.0)))
        return int(samples * 2)

    def pop_next_chunk_raw(self, sr: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        chunk単位で raw(json相当) を返す。
        - 返却 raw は chunk-local t_ms (0..chunk_ms-step_ms)
        - chunk1以降は先頭overlap区間を捨てる
        """
        if sr is None:
            sr = self.input_sr_default
        sr = int(sr)

        need_bytes = self._bytes_for_ms(sr, self.spec.chunk_ms)
        if len(self._buf) < need_bytes:
            return None

        # chunk取り出し（overlapを含むウィンドウ）
        chunk_pcm = bytes(self._buf[:need_bytes])

        # 次のchunkへ: stride分だけ進める（= overlap を残す）
        stride_bytes = self._bytes_for_ms(sr, self.spec.stride_ms)
        del self._buf[:stride_bytes]

        raw = self._process_chunk_pcm_to_raw(chunk_pcm=chunk_pcm, sr=sr)

        # chunk1以降は先頭overlapフレームを捨てる（あなたの合意仕様）
        if self._chunk_index > 0 and self.spec.frames_overlap > 0:
            raw["frames"] = raw.get("frames", [])[self.spec.frames_overlap :]

            # t_ms を 0 始まりに詰め直し
            frames = raw.get("frames", [])
            for i, fr in enumerate(frames):
                fr["t_ms"] = i * self.spec.step_ms

        raw["chunk_index"] = int(self._chunk_index)
        base = int(self._chunk_index * self.spec.stride_ms)
        if self._chunk_index > 0:
            base += int(self.spec.overlap_ms)   # ★保持開始=overlap捨て後の位置
        raw["chunk_start_ms"] = base

        self._chunk_index += 1
        return raw

    def _process_chunk_pcm_to_raw(self, chunk_pcm: bytes, sr: int) -> Dict[str, Any]:
        """
        ここが肝:
          既存合格の raw 生成（MouthStreamerOC）を黒箱として呼ぶだけ
        """
        # MouthStreamerOC は flush() で out_json に dump する :contentReference[oaicite:2]{index=2}
        # いったんテンポラリに吐かせて読み戻す（内部実装に依存しすぎない）
        fd, tmp = tempfile.mkstemp(prefix="m3_raw_chunk_", suffix=".json")
        os.close(fd)
        try:
            streamer = MouthStreamerOC(session_id=f"chunk_{self._chunk_index:06d}", cfg=self._streamer_cfg, out_json=tmp)

            streamer.push_pcm16_mono(chunk_pcm, input_sr=sr)
            streamer.finalize()  # finalize() は emit→flush する :contentReference[oaicite:3]{index=3}

            with open(tmp, "r", encoding="utf-8") as f:
                obj = json.load(f)

            # obj["frames"] は raw互換のキー（vad_active/f1_hz/f2_hz/rms/mouth_id_raw等）で出ている :contentReference[oaicite:4]{index=4}
            frames = obj.get("frames", [])
            out_raw = {
                "format": "mouth_timeline.formant.raw.v0",
                "step_ms": int(self.spec.step_ms),
                "frames": frames,
            }
            return out_raw
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass
