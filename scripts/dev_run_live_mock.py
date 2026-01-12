# scripts/dev_run_live_mock.py
from __future__ import annotations

import argparse
import json
import os
import time
import wave
from typing import Tuple

import numpy as np

from m3p.live.mouth_streamer import MouthStreamerOpenClose, pcm16le_to_float32_mono


def atomic_write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_wav_pcm16(path: str) -> Tuple[int, int, bytes]:
    """
    Read WAV via stdlib wave.
    Returns: (sr, channels, pcm16le_bytes_interleaved)
    NOTE: supports PCM16 only.
    """
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        comptype = wf.getcomptype()

        if comptype != "NONE":
            raise RuntimeError(f"WAV compression not supported: comptype={comptype}")
        if sampwidth != 2:
            raise RuntimeError(f"Only PCM16 supported for PoC. sampwidth={sampwidth}")

        pcm = wf.readframes(nframes)
        return sr, channels, pcm


def chunk_bytes(pcm: bytes, bytes_per_chunk: int):
    for i in range(0, len(pcm), bytes_per_chunk):
        yield pcm[i:i + bytes_per_chunk]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="Input WAV (PCM16). ex: in/sample.wav")
    ap.add_argument("--out_dir", required=True, help="Output dir. ex: out/live/sess_mock_01")
    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--chunk_ms", type=int, default=20, help="Mock stream chunk size in ms")
    ap.add_argument("--resample_to", type=int, default=16000, help="Analysis SR")
    ap.add_argument("--window_ms", type=int, default=240, help="RMS window for open/close")
    ap.add_argument("--rms_thr", type=float, default=0.015, help="RMS threshold for open")
    ap.add_argument("--realtime", action="store_true", help="Sleep to simulate real-time streaming")
    ap.add_argument("--progress_every", type=int, default=25, help="Log every N emitted frames")
    args = ap.parse_args()

    out_mouth = os.path.join(args.out_dir, "mouth_timeline.live.json")
    out_log = os.path.join(args.out_dir, "mock.log.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)

    sr, channels, pcm = read_wav_pcm16(args.wav)

    # bytes per chunk (PCM16LE interleaved)
    bytes_per_sample = 2
    bytes_per_frame = bytes_per_sample * channels
    samples_per_chunk = int(round(sr * (args.chunk_ms / 1000.0)))
    if samples_per_chunk <= 0:
        raise ValueError("chunk_ms too small")
    bytes_per_chunk = samples_per_chunk * bytes_per_frame

    streamer = MouthStreamerOpenClose(
        step_ms=args.step_ms,
        analysis_sr=args.resample_to,
        window_ms=args.window_ms,
        rms_thr=args.rms_thr,
        open_id=1,   # PoC: open -> 'a'
        close_id=0,
    )

    t0 = time.time()
    emitted_total = 0

    with open(out_log, "w", encoding="utf-8") as flog:
        # initial write (empty)
        atomic_write_json(out_mouth, streamer.to_mouth_timeline_json())

        for bi, chunk in enumerate(chunk_bytes(pcm, bytes_per_chunk)):
            if not chunk:
                continue

            x = pcm16le_to_float32_mono(chunk, channels=channels)  # mono float32
            streamer.ingest_audio_chunk(x, sr_in=sr)

            new_frames = streamer.try_emit_frames()
            if new_frames:
                emitted_total += len(new_frames)
                atomic_write_json(out_mouth, streamer.to_mouth_timeline_json())

                if (emitted_total // args.progress_every) != ((emitted_total - len(new_frames)) // args.progress_every):
                    last_t = streamer.last_t_ms
                    msg = {
                        "kind": "progress",
                        "emitted_frames": emitted_total,
                        "last_t_ms": last_t,
                        "out_mouth": out_mouth,
                    }
                    print(msg)
                    flog.write(json.dumps(msg, ensure_ascii=False) + "\n")

            if args.realtime:
                # rough real-time pacing based on chunk_ms
                time.sleep(args.chunk_ms / 1000.0)

        # finalize
        tail = streamer.finalize_tail()
        if tail:
            emitted_total += len(tail)
            atomic_write_json(out_mouth, streamer.to_mouth_timeline_json())

        dt = time.time() - t0
        done = {
            "kind": "done",
            "wav": args.wav,
            "sr": sr,
            "channels": channels,
            "analysis_sr": args.resample_to,
            "step_ms": args.step_ms,
            "chunk_ms": args.chunk_ms,
            "window_ms": args.window_ms,
            "rms_thr": args.rms_thr,
            "emitted_frames": emitted_total,
            "elapsed_s": dt,
            "out_mouth": out_mouth,
        }
        print(done)
        flog.write(json.dumps(done, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
