# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


def _frame_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


@dataclass
class EnergyVADStreamConfig:
    analysis_sr: int = 16000

    # short-time energy
    frame_ms: int = 20
    hop_ms: int = 10
    energy_thr: float = 1e-4

    # smoothing
    min_speech_ms: int = 100
    min_silence_ms: int = 100

    # output grid
    step_ms: int = 40


class EnergyVADStream:
    """
    Streaming energy VAD:
      - keep analysis_sr audio buffer
      - compute raw active on hop grid (10ms)
      - smooth with min_speech/min_silence
      - resample to 40ms step by majority vote over [t, t+step)
    """

    def __init__(self, cfg: EnergyVADStreamConfig):
        self.cfg = cfg
        self._buf = np.zeros((0,), dtype=np.float32)
        self._raw_hop_active: List[int] = []   # per-hop (10ms)
        self._hop_ms = self.cfg.hop_ms
        self._frame_ms = self.cfg.frame_ms

        self._step_active: List[int] = []      # per-step (40ms)
        self._step_ms = self.cfg.step_ms

        self._frame_samples = int(round(self.cfg.analysis_sr * (self._frame_ms / 1000.0)))
        self._hop_samples = int(round(self.cfg.analysis_sr * (self._hop_ms / 1000.0)))
        self._step_samples = int(round(self.cfg.analysis_sr * (self._step_ms / 1000.0)))
        if self._frame_samples <= 0 or self._hop_samples <= 0 or self._step_samples <= 0:
            raise ValueError("bad VAD config (samples <= 0)")

    @property
    def step_mask(self) -> List[int]:
        return self._step_active

    def push_audio(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        self._buf = np.concatenate([self._buf, x.astype(np.float32, copy=False)], axis=0)
        self._update_raw_hop()
        self._update_step_mask()

    def ensure_steps(self, n_steps: int) -> None:
        # we can't fabricate; just make sure pipeline ran
        # caller should only request steps that are already available by lookahead
        if len(self._step_active) >= n_steps:
            return

    # ---------- internals ----------

    def _update_raw_hop(self) -> None:
        """
        Extend raw hop activity as much as possible.
        """
        # hop index k uses window [k*hop, k*hop + frame)
        while True:
            k = len(self._raw_hop_active)
            s0 = k * self._hop_samples
            s1 = s0 + self._frame_samples
            if s1 > self._buf.size:
                break
            seg = self._buf[s0:s1]
            rms = _frame_rms(seg)
            a = 1 if rms >= self.cfg.energy_thr else 0
            self._raw_hop_active.append(a)

        # smoothing on the whole sequence (cheap enough at this scale)
        self._raw_hop_active = self._smooth_min_durations(self._raw_hop_active)

    def _smooth_min_durations(self, act: List[int]) -> List[int]:
        """
        Enforce min_speech_ms and min_silence_ms on hop grid.
        """
        if not act:
            return act

        min_speech_hops = max(1, int(round(self.cfg.min_speech_ms / self._hop_ms)))
        min_silence_hops = max(1, int(round(self.cfg.min_silence_ms / self._hop_ms)))

        out = act[:]
        i = 0
        n = len(out)
        while i < n:
            v = out[i]
            j = i + 1
            while j < n and out[j] == v:
                j += 1
            run = j - i

            if v == 1 and run < min_speech_hops:
                # short speech -> flip to silence
                for k in range(i, j):
                    out[k] = 0
            elif v == 0 and run < min_silence_hops:
                # short silence -> flip to speech
                for k in range(i, j):
                    out[k] = 1

            i = j
        return out

    def _update_step_mask(self) -> None:
        """
        Convert hop grid to step grid by majority vote over [t, t+step).
        """
        # hop time origin is 0ms; hop index k at t=k*hop_ms
        # step index s at t=s*step_ms, take hops in [t, t+step)
        while True:
            s = len(self._step_active)
            t0_ms = s * self._step_ms
            t1_ms = t0_ms + self._step_ms

            k0 = int(np.floor(t0_ms / self._hop_ms))
            k1 = int(np.ceil(t1_ms / self._hop_ms))

            if k1 > len(self._raw_hop_active):
                break

            window = self._raw_hop_active[k0:k1]
            if not window:
                break
            ones = sum(window)
            v = 1 if ones >= (len(window) - ones) else 0  # majority
            self._step_active.append(int(v))
