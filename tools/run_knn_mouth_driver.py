#!/usr/bin/env python3
# tools/run_knn_mouth_driver.py
import argparse
import os
import subprocess
import sys
from typing import List

def _run(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ids",
        nargs="+",
        default=["1_0_2", "1_0_5", "1_0_72", "1_1_4", "1_2_1"],
        help="utterance ids",
    )

    # repo-relative defaults（現状の配置に合わせて固定）
    ap.add_argument("--raw_dir", default="out/live_pseudo")
    ap.add_argument("--gt_glob", default="out/knn_data/train/*.f1f2.json")
    ap.add_argument("--wav_dir", default="mnt/data")
    ap.add_argument("--out_dir", default="out/knn_mouth")

    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--fallback_id_active", type=int, default=2)
    ap.add_argument("--min_conf_ratio", type=float, default=1.15)

    # i挿入パラメータ（あなたの実行例をデフォルト化）
    ap.add_argument("--min_run_ms", type=int, default=160)
    ap.add_argument("--interval_ms", type=int, default=160)
    ap.add_argument("--max_i_per_run", type=int, default=3)

    # audio出力の運用切替
    # - wavpath : audio="mnt/data/<ID>.wav" のようにパスを書きたい場合
    # - wavname : audio="<ID>.wav" のように「ファイル名だけ」を書く（M0方式）
    # - live    : audio="live" を書く（Live/OBS想定）
    # - empty   : audio="" を書く（Live/OBS想定）
    ap.add_argument("--audio_mode", choices=["wavpath", "wavname", "live", "empty"], default="wavname")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    py = sys.executable  # venv/codespacesでも安全

    for utt_id in args.ids:
        raw_path = os.path.join(args.raw_dir, f"{utt_id}.mouth_timeline.formant.raw.json")
        wav_path = os.path.join(args.wav_dir, f"{utt_id}.wav")

        out_knn = os.path.join(args.out_dir, f"{utt_id}.mouth_timeline.knn.json")
        out_knn_i = os.path.join(args.out_dir, f"{utt_id}.mouth_timeline.knn.i.json")
        
        if args.audio_mode == "wavpath":
            audio_arg = wav_path
        elif args.audio_mode == "wavname":
            audio_arg = f"{utt_id}.wav"
        elif args.audio_mode == "live":
            audio_arg = "live"
        else:
            audio_arg = ""

        # 1) raw -> kNN mouth_timeline
        _run([
            py, "tools/knn_from_formant_raw_to_mouth_timeline.py",
            "--raw", raw_path,
            "--gt_glob", args.gt_glob,
            "--out", out_knn,
            "--audio", audio_arg,
            "--step_ms", str(args.step_ms),
            "--k", str(args.k),
            "--fallback_id_active", str(args.fallback_id_active),
            "--min_conf_ratio", str(args.min_conf_ratio),
        ])

        # 2) i挿入
        _run([
            py, "scripts/mouth_insert_intermediate_i.py",
            out_knn,
            out_knn_i,
            "--min_run_ms", str(args.min_run_ms),
            "--interval_ms", str(args.interval_ms),
            "--max_i_per_run", str(args.max_i_per_run),
        ])

    print("\n[OK] driver done.")
    print("out_dir:", args.out_dir)

if __name__ == "__main__":
    main()
