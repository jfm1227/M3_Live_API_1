# scripts/dev_expression_chunk_test.py
from m3p.live.expression_streamer import ExpressionStreamer, ExpressionStreamerConfig
from m3p.live.expression_chunk_postprocess import (
    postprocess_expression_for_chunks,
    ChunkConfig,
    BlinkConfig,
)

# ---- 設定 ----
STEP_MS = 40
DURATION_MS = 30000          # 30秒
CHUNK_MS = 400
OVERLAP_MS = 200
CHUNK_STARTS = list(range(0, DURATION_MS, CHUNK_MS - OVERLAP_MS))

EMO_MAP = {
    "1_0": "normal",
    "1_1": "smile",
}

# ---- expression生成（tool_call相当）----
es = ExpressionStreamer(
    ExpressionStreamerConfig(
        step_ms=STEP_MS,
        default_expression="normal",
        emo_map=EMO_MAP,
        unknown_emo_to="normal",
    )
)
es.init_at(0)
es.on_emo_id("1_1", 1000)   # 1秒で smile に変化

base_events = es.to_json()

# ---- chunk + blink 後処理 ----
out = postprocess_expression_for_chunks(
    base_events,
    duration_ms=DURATION_MS,
    chunk_start_ms_list=CHUNK_STARTS,
    chunk_cfg=ChunkConfig(
        step_ms=STEP_MS,
        chunk_ms=CHUNK_MS,
        overlap_ms=OVERLAP_MS,
        default_expression="normal",
    ),
    blink_cfg=BlinkConfig(
        enabled=True,
        blink_interval_ms=10000,   # 10秒に1回
        blink_duration_ms=160,
    ),
)

# ---- 確認用出力 ----
print("BASE EVENTS:", base_events[:5])
print("CHUNK[0] EVENTS:", out["chunks"][0]["events"])
print("CHUNK[1] EVENTS:", out["chunks"][1]["events"])
print("CHUNK[50] EVENTS:", out["chunks"][50]["events"])  # 50*200ms=10000ms付近
print("CHUNK[51] EVENTS:", out["chunks"][51]["events"])
