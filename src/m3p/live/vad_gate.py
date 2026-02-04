"""Lightweight VAD-like gate for 40ms frame grids.

Goal:
- Provide a stable voiced/unvoiced decision using only RMS (or an energy-like scalar).
- Avoid chattering via hysteresis + hangover + minimum on/off durations.

This is NOT a neural VAD; it's a deterministic gate suitable for real-time use.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VADGateConfig:
    # Frame grid
    step_ms: int = 40

    # Hysteresis thresholds on the energy scalar (e.g., RMS)
    on_thr: float = 0.020
    off_thr: float = 0.012

    # Keep voiced=True for a short time after energy drops below off_thr
    hangover_ms: int = 160

    # Minimum durations to accept a state change (debounce)
    min_on_ms: int = 80
    min_off_ms: int = 80


class VADGate:
    """Deterministic VAD-like gate.

    Usage:
        gate = VADGate(VADGateConfig(...))
        voiced = gate.update(rms)

    Internals:
        - Pending transitions (to_on/to_off) accumulate time until min_* satisfied.
        - Hangover extends voiced state after falling below off_thr.
    """

    def __init__(self, cfg: VADGateConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._voiced: bool = False
        self._pending_on_ms: int = 0
        self._pending_off_ms: int = 0
        self._hangover_left_ms: int = 0

    @property
    def voiced(self) -> bool:
        return self._voiced

    def update(self, energy: float) -> bool:
        """Update gate with per-frame energy (RMS or similar).

        Returns:
            bool: voiced decision for this frame.
        """
        step = int(self.cfg.step_ms)

        if self._voiced:
            # If energy is clearly high, refresh hangover.
            if energy >= self.cfg.off_thr:
                self._hangover_left_ms = int(self.cfg.hangover_ms)
            else:
                self._hangover_left_ms = max(0, self._hangover_left_ms - step)

            # Consider turning OFF only if:
            # - energy below off_thr AND
            # - hangover consumed AND
            # - below-thr persists long enough (min_on_ms)
            if energy < self.cfg.off_thr and self._hangover_left_ms == 0:
                self._pending_off_ms += step
                self._pending_on_ms = 0
                if self._pending_off_ms >= self.cfg.min_off_ms:
                    self._voiced = False
                    self._pending_off_ms = 0
                    self._pending_on_ms = 0
            else:
                self._pending_off_ms = 0

        else:
            # Consider turning ON only if energy above on_thr persists long enough (min_off_ms)
            if energy >= self.cfg.on_thr:
                self._pending_on_ms += step
                self._pending_off_ms = 0
                if self._pending_on_ms >= self.cfg.min_on_ms:
                    self._voiced = True
                    self._hangover_left_ms = int(self.cfg.hangover_ms)
                    self._pending_on_ms = 0
                    self._pending_off_ms = 0
            else:
                self._pending_on_ms = 0

        return self._voiced
