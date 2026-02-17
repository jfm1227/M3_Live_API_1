import argparse
import json
import os
import sys
import wave
from typing import Dict, Any, List, Optional

# src へのパスを通す (必要に応じて)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from m3p.live.mouth_streamer_oc import MouthStreamerOC, MouthOCConfig

def _to_float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def _project_streamer_out_to_raw(streamer_json_path: str) -> Dict[str, Any]:
    """
    MouthStreamerOC が出した JSON (mouth_timeline_v0) を
    従来の raw.json (m3p.mouth.timeline 形式) へ射影する。
    """
    with open(streamer_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames_in = data.get("frames", [])
    frames_out = []

    for fr in frames_in:
        # 修正：metaを参照せず、streamer_outのトップレベルから取得し
        # 必要最小限のキーに固定する
        t_ms = fr.get("t_ms")
        
        frames_out.append(
            {
                "t_ms": t_ms,
                "vad_active": int(fr.get("vad_active", 0) or 0),
                "f1_hz": fr.get("f1_hz", None),
                "f2_hz": fr.get("f2_hz", None),
                "src": "mouth_streamer_oc",
            }
        )

    out = {
        "version": "m3p.mouth.timeline.v1",
        "step_ms": data.get("step_ms", 40),
        "frames": frames_out,
        "meta": data.get("meta", {}),
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="Input mono wav (PCM16)")
    ap.add_argument("--out_raw", required=True, help="Output raw.json")
    ap.add_argument("--tmp_streamer_json", default=None, help="mouth_streamer_oc の生出力（meta入り）を保存したい場合")
    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--input_sr", type=int, default=16000)
    ap.add_argument("--vad_energy_thr", type=float, default=0.0004) # (0.02 RMS)^2
    args = ap.parse_args()

    if not os.path.exists(args.wav):
        print(f"Error: wav not found: {args.wav}")
        sys.exit(1)

    # 出力先ディレクトリの準備
    out_dir = os.path.dirname(args.out_raw)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 1. MouthStreamerOC を使って逐次処理
    # (ここでは wav 全体を一気に読み込んでいるが、内部で 40ms ずつ push することでストリーム動作を模倣)
    tmp_streamer_json = args.tmp_streamer_json or (args.out_raw + ".streamer.json")

    cfg_kwargs = dict(
        step_ms=args.step_ms,
        analysis_sr=args.input_sr,
        input_sr_default=args.input_sr,
    )
    if args.vad_energy_thr is not None:
        cfg_kwargs["vad_energy_thr"] = args.vad_energy_thr

    ms = MouthStreamerOC(
        out_json=tmp_streamer_json,
        session_id="dev_tts_stream_to_m3",
        cfg=MouthOCConfig(**cfg_kwargs),
    )

    with wave.open(args.wav, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        if sw != 2 or ch != 1:
            print(f"Warning: input wav is ch={ch} sw={sw}. Expected Mono/PCM16. Processing anyway...")
        
        # 40ms 単位で読み込み
        chunk_samples = int(sr * args.step_ms / 1000)
        while True:
            b = wf.readframes(chunk_samples)
            if not b:
                break
            ms.push_pcm16_mono(b, input_sr=sr)

    ms.finalize()

    # 2. 生成された streamer_json を raw.json 形式に変換
    raw_data = _project_streamer_out_to_raw(tmp_streamer_json)
    with open(args.out_raw, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)

    print(f"[Done] raw.json written to: {args.out_raw}")

if __name__ == "__main__":
    main()