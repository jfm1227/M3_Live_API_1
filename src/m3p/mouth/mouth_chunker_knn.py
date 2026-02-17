# src/m3p/mouth/mouth_chunker_knn.py
from __future__ import annotations

import glob
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from m3p.live.energy_vad_stream import EnergyVADStream, EnergyVADStreamConfig
from m3p.live.strict2_online import Strict2Online, Strict2OnlineConfig
from m3p.live.formant_extractor import extract_f1_f2


VALID_VOWEL_IDS = {1, 2, 3, 4, 5}


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and (x == x)


class _KNNVowelDB:
    """
    kNN を runtime に持ち込むための最小実装。
    学習DBは data/knn_db/*.f1f2.json を読む（Step0と同じ資産）
    - Zスコア正規化
    - 距離重み投票 (1/(d^2+eps))
    """
    def __init__(self, db: List[Tuple[float, float, int]], k: int = 5):
        if not db:
            raise RuntimeError("empty kNN db")
        self.k = int(k)
        self._db = db
        self._z = self._compute_z(db)
        self._db_z = [(self._z_point(f1, f2, self._z), vid) for (f1, f2, vid) in db]

    @staticmethod
    def from_f1f2_jsons(paths: List[str], k: int = 5) -> "_KNNVowelDB":
        db: List[Tuple[float, float, int]] = []
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            for fr in obj.get("frames", []):
                vid = fr.get("vowel_id")
                f1 = fr.get("f1_hz")
                f2 = fr.get("f2_hz")
                if vid in VALID_VOWEL_IDS and _is_num(f1) and _is_num(f2):
                    db.append((float(f1), float(f2), int(vid)))
        return _KNNVowelDB(db=db, k=k)

    @staticmethod
    def _compute_z(db: List[Tuple[float, float, int]]) -> Tuple[float, float, float, float]:
        n = len(db)
        mu1 = sum(x for x, _, _ in db) / n
        mu2 = sum(y for _, y, _ in db) / n
        v1 = sum((x - mu1) ** 2 for x, _, _ in db) / n
        v2 = sum((y - mu2) ** 2 for _, y, _ in db) / n
        s1 = math.sqrt(v1) if v1 > 0 else 1.0
        s2 = math.sqrt(v2) if v2 > 0 else 1.0
        return mu1, mu2, s1, s2

    @staticmethod
    def _z_point(f1: float, f2: float, z: Tuple[float, float, float, float]) -> Tuple[float, float]:
        mu1, mu2, s1, s2 = z
        return ((f1 - mu1) / s1, (f2 - mu2) / s2)

    def predict(self, f1: float, f2: float) -> int:
        q1, q2 = self._z_point(float(f1), float(f2), self._z)
        dlist: List[Tuple[float, int]] = []
        for (x1, x2), vid in self._db_z:
            d2 = (q1 - x1) ** 2 + (q2 - x2) ** 2
            dlist.append((d2, vid))
        dlist.sort(key=lambda t: t[0])

        kk = min(self.k, len(dlist))
        eps = 1e-9
        votes: Dict[int, float] = {}
        for i in range(kk):
            d2, vid = dlist[i]
            w = 1.0 / (d2 + eps)
            votes[vid] = votes.get(vid, 0.0) + w

        items = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
        return int(items[0][0])


@dataclass
class MouthChunkerKNNConfig:
    step_ms: int = 40
    chunk_len_ms: int = 400
    ring_start_ms: int = 600          # >=600ms たまってから chunk を吐き始める
    ring_keep_ms: int = 2000          # 使い回し安全用（smokeでは十分大きく）
    formant_window_ms: int = 200
    fallback_id_active: int = 2       # vad=1 で欠損時は i(2)
    knn_k: int = 5
    knn_db_glob: str = "data/knn_db/*.f1f2.json"

    # EnergyVADStream は 16k 前提なので、入力を16kへ resample して push する
    vad_cfg: EnergyVADStreamConfig = EnergyVADStreamConfig(
        analysis_sr=16000,
        frame_ms=20,
        hop_ms=10,
        energy_thr=1e-4,
        min_speech_ms=100,
        min_silence_ms=100,
        step_ms=40,
    )


class MouthChunkerKNN:
    """
    400ms fixed chunk adapter.
    - VAD: EnergyVADStream（入力PCMをpushするだけ。参照は step_mask）
    - strict2: vad=1なら非0保証（open_idはfallback_id_active）
    - formant: extract_f1_f2（float32[-1,1]）
    - kNN: data/knn_db をロードして分類
    NOTE:
      - i挿入はここではしない（結合後一括）
      - t_ms は最終出力で 0.. の連番（Step1/Step1.5用）
    """
    def __init__(self, sample_rate: int, cfg: MouthChunkerKNNConfig = MouthChunkerKNNConfig()):
        assert cfg.chunk_len_ms % cfg.step_ms == 0
        self.sample_rate = int(sample_rate)
        self.cfg = cfg
        self.frames_per_chunk = cfg.chunk_len_ms // cfg.step_ms

        # ring buffer: float32 mono [-1,1]
        self._ring = np.zeros((0,), dtype=np.float32)

        # output cursor (abs)
        self._out_abs_ms = 0

        # VAD (stream)
        self._vad = EnergyVADStream(cfg.vad_cfg)

        # strict2 (non-zero guarantee when vad=1)
        self._strict2 = Strict2Online(Strict2OnlineConfig(default_open_id=int(cfg.fallback_id_active)))

        # kNN DB
        db_paths = sorted(glob.glob(cfg.knn_db_glob))
        if not db_paths:
            raise RuntimeError(f"kNN DB not found: {cfg.knn_db_glob}")
        self._knn = _KNNVowelDB.from_f1f2_jsons(db_paths, k=cfg.knn_k)

        self._last_mouth_id = 0
        self._chunks: List[List[Dict[str, int]]] = []

    def push_pcm16_mono(self, pcm16: bytes, t_ms: int) -> None:
        # int16 -> float32 [-1,1]
        pcm = (np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0)
        if pcm.size == 0:
            return

        # ---- VAD stream push（rawと同じ思想：入力PCMを順に一度だけ入れる） ----
        x16 = self._resample_linear(pcm, self.sample_rate, self.cfg.vad_cfg.analysis_sr)
        self._vad.push_audio(x16)

        # ring append
        self._ring = np.concatenate([self._ring, pcm], axis=0)

        # ring trim (keep_ms)
        keep_samples = int(self.sample_rate * self.cfg.ring_keep_ms / 1000)
        if self._ring.size > keep_samples:
            drop = self._ring.size - keep_samples
            self._ring = self._ring[drop:]

        self._try_emit_chunks()

    def finalize(self) -> Dict[str, Any]:
        frames: List[Dict[str, int]] = []
        abs_t = 0
        for ch in self._chunks:
            for fr in ch:
                frames.append({"t_ms": abs_t, "mouth_id": int(fr["mouth_id"])})
                abs_t += self.cfg.step_ms
        return {"version": "mouth_v0.1", "step_ms": self.cfg.step_ms, "frames": frames}

    # ---------------- internal ----------------
    def _try_emit_chunks(self) -> None:
        ring_ms = int(1000 * self._ring.size / self.sample_rate)

        # start guard
        while ring_ms >= self.cfg.ring_start_ms and ring_ms >= self.cfg.chunk_len_ms:
            # VAD step_mask guard: このchunkの最後stepまで参照可能であること
            need_last_step = int((self._out_abs_ms + (self.cfg.chunk_len_ms - self.cfg.step_ms)) // self.cfg.step_ms)
            if len(self._vad.step_mask) <= need_last_step:
                break

            self._emit_one_chunk()
            ring_ms = int(1000 * self._ring.size / self.sample_rate)

    def _emit_one_chunk(self) -> None:
        chunk_samples = int(self.sample_rate * self.cfg.chunk_len_ms / 1000)
        chunk = self._ring[:chunk_samples].copy()
        self._ring = self._ring[chunk_samples:]

        chunk_start_ms = int(self._out_abs_ms)
        self._out_abs_ms += self.cfg.chunk_len_ms

        frames: List[Dict[str, int]] = []

        for i in range(self.frames_per_chunk):
            t_rel = i * self.cfg.step_ms

            if i == 0:
                # hold-last at t=0 (overwrite semantics; we keep 10 frames total)
                frames.append({"t_ms": 0, "mouth_id": int(self._last_mouth_id)})
                continue

            step_abs = int((chunk_start_ms + t_rel) // self.cfg.step_ms)
            vad_active = int(self._vad.step_mask[step_abs]) if step_abs < len(self._vad.step_mask) else 0

            if vad_active == 0:
                mouth_id = 0
            else:
                # formant: chunk is float32[-1,1]
                f1, f2 = extract_f1_f2(
                    chunk,
                    self.sample_rate,   # sr
                    t_rel,              # center_ms
                    window_ms=self.cfg.formant_window_ms,
                )
                if f1 is None or f2 is None:
                    mouth_id = int(self.cfg.fallback_id_active)
                else:
                    mouth_id = int(self._knn.predict(float(f1), float(f2)))

                # strict2 non-zero guarantee
                mouth_id = int(self._strict2.apply(vad_active, mouth_id))

            frames.append({"t_ms": int(t_rel), "mouth_id": int(mouth_id)})

        self._last_mouth_id = int(frames[-1]["mouth_id"]) if frames else self._last_mouth_id
        self._chunks.append(frames)

    @staticmethod
    def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        if x.size == 0:
            return x.astype(np.float32, copy=False)
        if sr_in == sr_out:
            return x.astype(np.float32, copy=False)

        dur = x.size / float(sr_in)
        n_out = int(round(dur * sr_out))
        if n_out <= 1:
            return np.zeros((0,), dtype=np.float32)

        t_in = np.linspace(0.0, dur, num=x.size, endpoint=False, dtype=np.float64)
        t_out = np.linspace(0.0, dur, num=n_out, endpoint=False, dtype=np.float64)
        y = np.interp(t_out, t_in, x.astype(np.float64))
        return y.astype(np.float32)
