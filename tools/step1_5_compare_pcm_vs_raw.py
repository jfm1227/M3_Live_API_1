# tools/step1_5_compare_pcm_vs_raw.py
import argparse
import json


def _load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _frames_to_map(frames):
    m = {}
    for fr in frames:
        t = fr.get("t_ms")
        mid = fr.get("mouth_id")
        if t is None or mid is None:
            continue
        m[int(t)] = int(mid)  # same t_ms: last wins
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcm_json", required=True, help="Step1 output (PCM->chunk) mouth json")
    ap.add_argument("--raw_json", required=True, help="Step0 output (raw->kNN) mouth json")
    ap.add_argument("--limit", type=int, default=50, help="print first N mismatches")
    args = ap.parse_args()

    pcm = _load(args.pcm_json)
    raw = _load(args.raw_json)

    step_pcm = int(pcm.get("step_ms", 40))
    step_raw = int(raw.get("step_ms", 40))
    if step_pcm != step_raw:
        print(f"[WARN] step_ms mismatch pcm={step_pcm} raw={step_raw}")

    pcm_map = _frames_to_map(pcm.get("frames", []))
    raw_map = _frames_to_map(raw.get("frames", []))

    keys = sorted(set(pcm_map.keys()) | set(raw_map.keys()))
    mouth_mis = 0
    vad_mis = 0
    head = []

    for t in keys:
        p = int(pcm_map.get(t, 0))
        r = int(raw_map.get(t, 0))
        if p != r:
            mouth_mis += 1
            if len(head) < args.limit:
                head.append((t, r, p))
        pv = 1 if p != 0 else 0
        rv = 1 if r != 0 else 0
        if pv != rv:
            vad_mis += 1

    print("===== Step1.5 compare (PCM chunk vs RAW kNN) =====")
    print(f"- pcm : {args.pcm_json}")
    print(f"- raw : {args.raw_json}")
    print(f"- total_t_ms={len(keys)}  mouth_mismatch={mouth_mis}  vad_mismatch={vad_mis}")

    if head:
        print("[mismatch head] t_ms raw pcm")
        for t, r, p in head:
            print(f"  {t:6d}  {r}  {p}")

    if vad_mis != 0:
        raise SystemExit("[FAIL] VAD mismatch (must be 0)")
    if mouth_mis != 0:
        raise SystemExit("[FAIL] mouth_id mismatch (must be 0 in Step1.5)")
    print("[PASS] Step1.5 matches (pre i-insert).")


if __name__ == "__main__":
    main()
