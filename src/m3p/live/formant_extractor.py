# src/m3p/live/formant_extractor.py
import numpy as np
import parselmouth


def extract_f1_f2(
    wav_f: np.ndarray,
    sr: int,
    center_ms: int,
    window_ms: int = 200,
) -> tuple[float | None, float | None]:
    """
    wav_f: float32 mono [-1,1]
    sr: sampling rate
    center_ms: 中心時刻
    """
    half = window_ms // 2
    start = int((center_ms - half) * sr / 1000)
    end   = int((center_ms + half) * sr / 1000)

    if start < 0 or end > wav_f.size or end <= start:
        return None, None

    seg = wav_f[start:end]
    if seg.size < sr * 0.05:  # 50ms未満は捨てる
        return None, None

    snd = parselmouth.Sound(seg, sr)
    formant = snd.to_formant_burg(
        time_step=0.01,
        max_number_of_formants=5,
        maximum_formant=5500,
    )

    t = snd.get_total_duration() / 2.0
    f1 = formant.get_value_at_time(1, t)
    f2 = formant.get_value_at_time(2, t)

    if f1 is None or f2 is None:
        return None, None

    return float(f1), float(f2)
