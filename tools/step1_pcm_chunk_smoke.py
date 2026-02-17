#!/usr/bin/env python3
# tools/step1_pcm_chunk_smoke.py
import argparse
import wave

from m3p.mouth.mouth_chunker_knn import MouthChunkerKNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--knn_db_glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--chunk_len_ms", type=int, default=400)
    args = ap.parse_args()

    with wave.open(args.wav, "rb") as wf:
        ch = wf.getnchannels()
        assert wf.getsampwidth() == 2
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    import numpy as np

    pcm = np.frombuffer(raw, dtype=np.int16)

    if ch == 2:
        # stereo -> mono (L+R)/2
        pcm = pcm.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif ch == 1:
        pass
    else:
        raise RuntimeError(f"unsupported channels: {ch}")

    chunker = MouthChunkerKNN(
        sample_rate=sr,
        step_ms=args.step_ms,
        chunk_len_ms=args.chunk_len_ms,
        knn_db_glob=args.knn_db_glob,
    )

    # --- 分割 push（本番相当） ---
    # 例：push を 160ms 単位で流す（step_ms=40 の4フレーム分）
    PUSH_MS = 160

    samples_per_push = int(sr * PUSH_MS / 1000)
    total = len(pcm)  # pcm: mono int16 numpy array

    t_ms = 0
    pos = 0
    while pos < total:
        blk = pcm[pos:pos + samples_per_push]
        if blk.size == 0:
            break
        chunker.push_pcm16_mono(blk.tobytes(), t_ms=t_ms)
        pos += blk.size
        t_ms += int(1000 * blk.size / sr)

    out = chunker.finalize()

    with open(args.out, "w", encoding="utf-8") as f:
        import json
        json.dump(out, f, ensure_ascii=False, indent=2)

    # basic checks
    frames = out["frames"]
    assert out["step_ms"] == args.step_ms
    print(f"[OK] frames={len(frames)} step_ms={args.step_ms}")
    print("[OK] Step1 PCM->chunk smoke passed")


if __name__ == "__main__":
    main()
