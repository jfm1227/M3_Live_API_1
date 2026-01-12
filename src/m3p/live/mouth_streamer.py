# src/m3p/live/mouth_streamer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Iterable
import math
import numpy as np


@dataclass
class MouthFrame:
    t_ms: int
    dur_ms: int
    mouth_id: int  # 0..5 (PoCでは 0=close, 1=open(a))


class AudioRing:
    """Fixed-size ring buffer for mono float32 audio [-1,1]."""
    def __init__(self, capacity_samples: int):
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be > 0")
        self._buf = np.zeros((capacity_samples,), dtype=np.float32)
        self._cap = capacity_samples
        self._size = 0
        self._wpos = 0

    def push(self, x: np.ndarray) -> None:
        if x.ndim != 1:
            raise ValueError("AudioRing expects mono 1D array")
        n = int(x.shape[0])
        if n <= 0:
            return
        # If pushing more than capacity, keep last cap samples
        if n >= self._cap:
            x = x[-self._cap:]
            n = int(x.shape[0])

        end = self._wpos + n
        if end <= self._cap:
            self._buf[self._wpos:end] = x
        else:
            k = self._cap - self._wpos
            self._buf[self._wpos:] = x[:k]
            self._buf[:end - self._cap] = x[k:]
        self._wpos = (self._wpos + n) % self._cap
        self._size = min(self._cap, self._size + n)

    def get_latest(self, n: int) -> np.ndarray:
        n = min(int(n), self._size)
        if n <= 0:
            return np.zeros((0,), dtype=np.float32)
        start = (self._wpos - n) % self._cap
        if start < self._wpos:
            return self._buf[start:self._wpos].copy()
        return np.concatenate([self._buf[start:], self._buf[:self._wpos]]).copy()

    @property
    def size(self) -> int:
        return self._size


def pcm16le_to_float32_mono(pcm_bytes: bytes, channels: int = 1) -> np.ndarray:
    """PCM16LE bytes -> mono float32 [-1,1]. If channels>1, average to mono."""
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if channels <= 1:
        return x
    # reshape interleaved: [L,R,L,R,...]
    n = (x.shape[0] // channels) * channels
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    x = x[:n].reshape((-1, channels))
    return x.mean(axis=1).astype(np.float32)


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Simple linear resampler (mono). Good enough for PoC."""
    if sr_in == sr_out:
        return x
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    if sr_in <= 0 or sr_out <= 0:
        raise ValueError("sr_in/sr_out must be > 0")

    # Target length
    n_out = int(round(x.size * (sr_out / sr_in)))
    if n_out <= 1:
        return np.zeros((0,), dtype=np.float32)

    # Interpolate
    xp = np.linspace(0.0, 1.0, num=x.size, endpoint=False, dtype=np.float64)
    fp = x.astype(np.float64, copy=False)
    xq = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
    y = np.interp(xq, xp, fp).astype(np.float32)
    return y


class MouthStreamerOpenClose:
    """
    PoC streamer:
      - audio chunk (mono) -> ring buffer (analysis_sr)
      - every step_ms: compute RMS over window_ms
      - mouth_id: 0 if rms < thr else 1
      - emits frames with t_ms quantized to step_ms and strictly increasing
    """
    def __init__(
        self,
        step_ms: int = 40,
        analysis_sr: int = 16000,
        window_ms: int = 240,
        rms_thr: float = 0.015,
        open_id: int = 1,
        close_id: int = 0,
    ):
        if step_ms <= 0:
            raise ValueError("step_ms must be > 0")
        if analysis_sr <= 0:
            raise ValueError("analysis_sr must be > 0")
        if window_ms <= 0:
            raise ValueError("window_ms must be > 0")
        if not (0.0 <= rms_thr <= 1.0):
            raise ValueError("rms_thr must be in [0,1]")

        self.step_ms = int(step_ms)
        self.analysis_sr = int(analysis_sr)
        self.window_ms = int(window_ms)
        self.rms_thr = float(rms_thr)
        self.open_id = int(open_id)
        self.close_id = int(close_id)

        cap = int(math.ceil((self.window_ms / 1000.0) * self.analysis_sr)) * 3  #余裕
        self.ring = AudioRing(max(1, cap))

        self._total_samples_in_analysis_sr = 0  # after resample
        self._last_emitted_t_ms: Optional[int] = None
        self._frames: List[MouthFrame] = []

    def _current_time_ms(self) -> int:
        # time based on samples accumulated in analysis_sr
        return int(round(self._total_samples_in_analysis_sr * 1000.0 / self.analysis_sr))

    def ingest_audio_chunk(self, pcm_mono_f32: np.ndarray, sr_in: int) -> None:
        if pcm_mono_f32.ndim != 1:
            raise ValueError("pcm_mono_f32 must be mono 1D")
        x = pcm_mono_f32.astype(np.float32, copy=False)
        x16 = resample_linear(x, sr_in=sr_in, sr_out=self.analysis_sr)
        self.ring.push(x16)
        self._total_samples_in_analysis_sr += int(x16.size)

    def _compute_rms(self) -> float:
        n = int(round(self.window_ms * self.analysis_sr / 1000.0))
        w = self.ring.get_latest(n)
        if w.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(w)) + 1e-12))

    def _quantize_t_ms(self, t_ms: int) -> int:
        return (t_ms // self.step_ms) * self.step_ms

    def try_emit_frames(self) -> List[MouthFrame]:
        """
        Emit new frames up to 'now' on step_ms grid.
        Strategy:
          - Determine current t_ms_q
          - Emit each step boundary strictly after last emitted
        """
        now_ms = self._current_time_ms()
        now_q = self._quantize_t_ms(now_ms)

        if self._last_emitted_t_ms is None:
            # start at 0
            next_t = 0
        else:
            next_t = self._last_emitted_t_ms + self.step_ms

        out: List[MouthFrame] = []
        while next_t <= now_q:
            rms = self._compute_rms()
            mouth_id = self.open_id if rms >= self.rms_thr else self.close_id
            f = MouthFrame(t_ms=next_t, dur_ms=self.step_ms, mouth_id=mouth_id)
            self._frames.append(f)
            out.append(f)
            self._last_emitted_t_ms = next_t
            next_t += self.step_ms
        return out

    def finalize_tail(self) -> List[MouthFrame]:
        """
        Call at end of stream:
          - emit frames up to the final quantized time
          - optionally append a final close frame (not strictly necessary)
        """
        out = self.try_emit_frames()
        # Optionally ensure last frame is close at end:
        # (PoCでは不要。必要ならここで追加)
        return out

    def to_mouth_timeline_json(self) -> Dict[str, Any]:
        # v0.1: event list {t_ms, dur_ms, mouth_id}
        return {
            "version": "mouth_timeline.v0.1",
            "step_ms": self.step_ms,
            "frames": [{"t_ms": f.t_ms, "dur_ms": f.dur_ms, "mouth_id": f.mouth_id} for f in self._frames],
        }

    @property
    def frames_count(self) -> int:
        return len(self._frames)

    @property
    def last_t_ms(self) -> Optional[int]:
        return self._last_emitted_t_ms
