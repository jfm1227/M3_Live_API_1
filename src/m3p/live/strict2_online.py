# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Strict2OnlineConfig:
    default_open_id: int = 3


class Strict2Online:
    def __init__(self, cfg: Strict2OnlineConfig):
        self.cfg = cfg
        self._last_nonzero = cfg.default_open_id

    def apply(self, vad_active: int, mouth_id: int) -> int:
        # track last non-zero
        if mouth_id != 0:
            self._last_nonzero = int(mouth_id)

        if vad_active == 0:
            return 0

        # vad=1 => must be non-zero
        if mouth_id == 0:
            mouth_id = self._last_nonzero if self._last_nonzero != 0 else self.cfg.default_open_id

        # final guarantee
        if mouth_id == 0:
            mouth_id = self.cfg.default_open_id

        return int(mouth_id)
