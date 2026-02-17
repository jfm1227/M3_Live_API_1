#!/usr/bin/env python3
# scripts/dev_pcm_to_m3_stream_test.py
#
# WAV(16k/mono/PCM16) を 40ms フレームで PCM push し、
# M3(MouthStreamerOC) の出力を「formant raw」形式に整形して保存する。
#
# 出力: mouth_timeline.formant.raw.v0
#  - frames[*].vad_active / f1_hz / f2_hz をトップレベルに置く（KNN段が読む想定）

from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

# NOTE: リポジトリ側の配置に合わせて import
# 例: src/m3p/live/mouth_streamer_oc.py に MouthStreamerOC / MouthOCConfig がいる想定
from m3p.live.mouth_streamer_oc import MouthStreamerOC, MouthOCConfig  # type: ignore


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _wav_read_pcm16_mono_bytes(wav_path: Path) -> tuple[bytes, int]:
    with wave.open(str(wav_path), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        if nch != 1:
            raise SystemExit(f"[ERROR] wav must be mono (channels=1). got {nch}")
        if sw != 2:
            raise SystemExit(f"[ERROR] wav must be PCM16 (sampwidth=2). got {sw}")
        pcm = wf.readframes(nframes)
    return pcm, int(sr)


def _stream_pcm_in_frames(
    ms: MouthStreamerOC,
    pcm: bytes,
    input_sr: int,
    frame_ms: int,
    sleep_rt: bool,
) -> None:
    # PCM16 => 1 sample = 2 bytes
    samples_per_frame = int(round(input_sr * frame_ms / 1000.0))
    bytes_per_frame = samples_per_frame * 2

    if bytes_per_frame <= 0:
        raise SystemExit("[ERROR] invalid frame size")

    total = len(pcm)
    off = 0
    while off < total:
        chunk = pcm[off : min(off + bytes_per_frame, total)]
        if not chunk:
            break
        ms.push_pcm16_mono(chunk, input_sr=input_sr)
        off += len(chunk)
        if sleep_rt:
            time.sleep(frame_ms / 1000.0)


def _to_knn_raw_shape(ms_out: Dict[str, Any], *, session_id: str) -> Dict[str, Any]:
    # MouthStreamerOC は frames[*].meta に f1/f2/vad を持つ :contentReference[oaicite:2]{index=2}
    step_ms = int(ms_out.get("step_ms", ms_out.get("meta", {}).get("step_ms", 40)))
    frames_in = ms_out.get("frames", [])
    if not isinstance(frames_in, list) or not frames_in:
        raise SystemExit("[ERROR] MouthStreamerOC output frames empty")

    frames_out: List[Dict[str, Any]] = []
    for fr in frames_in:
        if not isinstance(fr, dict) or "t_ms" not in fr:
            continue
        meta = fr.get("meta", {}) if isinstance(fr.get("meta", {}), dict) else {}
        frames_out.append(
            {
                "t_ms": int(fr["t_ms"]),
                # KNN段が読むキー（トップレベル）
                "vad_active": int(meta.get("vad_active", 0)),
                "f1_hz": meta.get("f1_hz", None),
                "f2_hz": meta.get("f2_hz", None),
                # デバッグ用に残す（任意）
                "rms": meta.get("rms", None),
            }
        )

    out = {
        "schema": "mouth_timeline.formant.raw.v0",
        "session_id": str(session_id),
        "step_ms": int(step_ms),
        "frames": frames_out,
        "meta": {
            "source_schema": str(ms_out.get("schema_version", ms_out.get("schema", "unknown"))),
            "source_meta": ms_out.get("meta", {}),
        },
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="16kHz mono PCM16 wav")
    ap.add_argument("--out_raw", required=True, help="output raw json (for KNN stage)")
    ap.add_argument("--session_id", default="sess_e2e_pcm")
    ap.add_argument("--input_sr", type=int, default=16000)
    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--window_ms", type=int, default=80)
    ap.add_argument("--frame_ms", type=int, default=40, help="PCM push frame size (ms)")
    ap.add_argument("--sleep_rt", action="store_true", help="sleep to simulate realtime")
    ap.add_argument("--vowel_mode", default="formant", choices=["formant", "simple"])
    ap.add_argument("--formant_window_ms", type=int, default=200)
    ap.add_argument("--formant_max_hz", type=int, default=5500)
    args = ap.parse_args()

    wav_path = Path(args.wav)
    out_raw = Path(args.out_raw)

    pcm, wav_sr = _wav_read_pcm16_mono_bytes(wav_path)
    if int(wav_sr) != int(args.input_sr):
        raise SystemExit(f"[ERROR] wav sr mismatch: wav={wav_sr} expected={args.input_sr} (用意したwavを16kにしてください)")

    cfg = MouthOCConfig(
        step_ms=int(args.step_ms),
        window_ms=int(args.window_ms),
        analysis_sr=16000,          # M3側解析SR（現状は16k前提でOK）
        input_sr_default=int(args.input_sr),
        vowel_mode=str(args.vowel_mode),
        formant_window_ms=int(args.formant_window_ms),
        formant_max_hz=int(args.formant_max_hz),
    )

    tmp_out = out_raw.with_suffix(out_raw.suffix + ".ms_tmp.json")
    ms = MouthStreamerOC(out_json=str(tmp_out), session_id=str(args.session_id), cfg=cfg)

    _stream_pcm_in_frames(
        ms=ms,
        pcm=pcm,
        input_sr=int(args.input_sr),
        frame_ms=int(args.frame_ms),
        sleep_rt=bool(args.sleep_rt),
    )

    # flush & finalize（MouthStreamerOCが out_json に書く）
    ms.flush()
    ms.finalize()

    ms_out = _load_json(tmp_out)
    raw_obj = _to_knn_raw_shape(ms_out, session_id=str(args.session_id))
    _write_json(out_raw, raw_obj)

    print("[dev_pcm_to_m3_stream_test][OK]")
    print(" wav    :", wav_path.as_posix())
    print(" tmp_out:", tmp_out.as_posix())
    print(" out_raw:", out_raw.as_posix())
    print(" frames :", len(raw_obj.get("frames", [])))


if __name__ == "__main__":
    main()
