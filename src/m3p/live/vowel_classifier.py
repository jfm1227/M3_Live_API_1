# src/m3p/live/vowel_classifier.py
import math
import os
import json
from typing import Iterable
from typing import Dict
from typing import Optional, Dict, Any, Tuple
import atexit

# === STEP1: global normalization params (from GT stats) ===
GLOBAL_F1_MEDIAN = 500.0
GLOBAL_F1_IQR    = 250.0

GLOBAL_F2_MEDIAN = 1650.0
GLOBAL_F2_IQR    = 800.0


VOWEL_TABLE = {
    1: ("a", 650, 1550),
    2: ("i", 450, 1800),
    3: ("u", 450, 1600),
    4: ("e", 500, 1800),
    5: ("o", 480, 1350),
}

# 共分散（対角近似）
# 値は GT ログから IQR / 分散を見て調整する前提
VOWEL_COV = {
    1: (120.0**2, 180.0**2),  # a
    2: ( 90.0**2, 260.0**2),  # i
    3: (100.0**2, 200.0**2),  # u
    4: (110.0**2, 220.0**2),  # e
    5: (130.0**2, 170.0**2),  # o
}

# デバッグ用：一度だけ設定を表示するためのフラグ
_PRINTED = False

# ===== Hold-last (previous vowel) cache =====
_LAST_VOWEL_ID: int = 0

# ===== Gauss (Mahalanobis) model cache =====
_GAUSS: Optional[Dict[str, Any]] = None
_GAUSS_PATH: Optional[str] = None
_GAUSS_LOAD_ERROR: Optional[str] = None

# ===== Optional Bayesian prior (from GT distribution) =====
# If VOWEL_PRIOR_JSON is set, apply:
#    score = d2 - beta * log(prior[vowel_id])
# beta = VOWEL_PRIOR_BETA (default 1.0)
_PRIOR: Optional[Dict[int, float]] = None
_PRIOR_SRC: Optional[str] = None
_PRIOR_BETA: float = 1.0

# --- ambiguity stats ---
_STAT_TOTAL: int = 0
_STAT_AMBIG: int = 0

# ===== Two-stage group split (front vs back) =====
# Goal: avoid large-class mistakes among:
#    - a (1)
#    - i/e (2,4)  [front]
#    - u/o (3,5)  [back]
# We split by F2 threshold (Hz) and only run maha within the selected group.
# If F2 is near the threshold, we consider both groups and mark as ambiguous.
_TWO_STAGE_ENABLED: Optional[bool] = None
_F2_THR_HZ: float = 1600.0
_F2_MARGIN_HZ: float = 150.0
_AMBIG_RATIO_THR: float = 1.25  # second_best_d2 / best_d2 < thr => ambiguous
_AMBIG_D2_MARGIN: float = 0.0   # (d2_2nd - d2_best) <= margin => ambiguous (0 disables)


# Print stats at process exit (once)
def _print_ambiguity_stats() -> None:
    if _STAT_TOTAL <= 0:
        return
    rate = 100.0 * float(_STAT_AMBIG) / float(_STAT_TOTAL)
    print(
        f"[vowel_classifier][stats] ambiguous={_STAT_AMBIG} / total={_STAT_TOTAL} "
        f"({rate:.2f}%)",
        flush=True,
    )

atexit.register(_print_ambiguity_stats)


def _load_gauss_from_env() -> None:
    """
    Load vowel gaussian model from env var VOWEL_GAUSS_JSON.
    Expected schema (from fit_vowel_gauss_from_wavs.py):
      {
        "space": "mel",
        "step_ms": 40,
        "vowels": {
          "1": {"mu":[..,..], "inv":[[..,..],[..,..]], "tau": float, "n": int},
          ...
        },
        "uncertain": {"ratio": float, "tau": float},
        "meta": {...}
      }
    """
    global _GAUSS, _GAUSS_PATH, _GAUSS_LOAD_ERROR
    if _GAUSS is not None or _GAUSS_LOAD_ERROR is not None:
        return
    path = os.environ.get("VOWEL_GAUSS_JSON", "").strip()
    if not path:
        _GAUSS_LOAD_ERROR = "VOWEL_GAUSS_JSON not set"
        return
    _GAUSS_PATH = path
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # minimal sanity checks
        if not isinstance(data, dict):
            raise ValueError("gauss json is not a dict")
        if data.get("space") not in (None, "mel"):
            raise ValueError(f"unsupported space: {data.get('space')}")
        vowels = data.get("vowels")
        if not isinstance(vowels, dict) or not vowels:
            raise ValueError("missing/empty vowels")
        # ensure required keys exist for each vowel
        for k, v in vowels.items():
            if not isinstance(v, dict):
                raise ValueError(f"vowels[{k}] is not a dict")
            if "mu" not in v or "inv" not in v:
                raise ValueError(f"vowels[{k}] missing mu/inv")
        _GAUSS = data
    except Exception as e:
        _GAUSS = None
        _GAUSS_LOAD_ERROR = f"{type(e).__name__}: {e}"


def _load_prior_from_env() -> None:
    """
    Load prior JSON from env var VOWEL_PRIOR_JSON (optional).
    Expected:
      { "prior": {"1":0.2,"2":0.2,"3":0.2,"4":0.2,"5":0.2}, ... }
    or directly that dict.
    """
    global _PRIOR, _PRIOR_SRC, _PRIOR_BETA
    if _PRIOR is not None:
        return
    _PRIOR = {}
    _PRIOR_SRC = None

    # eps smoothing (avoid zero-prob / extreme prior)
    try:
        eps = float(os.getenv("VOWEL_PRIOR_EPS", "1e-6"))
    except Exception:
        eps = 1e-6
    if eps < 0:
        eps = 0.0
    try:
        _PRIOR_BETA = float(os.environ.get("VOWEL_PRIOR_BETA", "1.0"))
    except Exception:
        _PRIOR_BETA = 1.0

    path = os.environ.get("VOWEL_PRIOR_JSON", "").strip()
    if not path:
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        prior_obj = obj.get("prior", obj) if isinstance(obj, dict) else {}
        tmp: Dict[int, float] = {}
        if isinstance(prior_obj, dict):
            for k, v in prior_obj.items():
                try:
                    vid = int(k)
                    pv = float(v)
                except Exception:
                    continue
                if pv > 0:
                    tmp[vid] = pv
        # eps smoothing: 未出現/ゼロの母音にも少量の質量を付与（priorのゼロ割/極端化防止）
        if eps > 0:
            for vid in range(1, 6):
                tmp[vid] = tmp.get(vid, 0.0) + eps
        s = float(sum(tmp.values()))
        if s > 0:
            _PRIOR = {k: (v / s) for k, v in tmp.items()}
            _PRIOR_SRC = path
    except Exception:
        _PRIOR = {}
        _PRIOR_SRC = None


def _load_two_stage_from_env() -> None:
    """Load two-stage (group split) params from env vars (once)."""
    global _TWO_STAGE_ENABLED, _F2_THR_HZ, _F2_MARGIN_HZ, _AMBIG_RATIO_THR, _AMBIG_D2_MARGIN
    if _TWO_STAGE_ENABLED is not None:
        return
    v = os.environ.get("VOWEL_TWO_STAGE", "1").strip().lower()
    _TWO_STAGE_ENABLED = v not in ("0", "false", "off", "no")
    try:
        _F2_THR_HZ = float(os.environ.get("VOWEL_GROUP_F2_THR_HZ", "1600"))
    except Exception:
        _F2_THR_HZ = 1600.0
    try:
        _F2_MARGIN_HZ = float(os.environ.get("VOWEL_GROUP_F2_MARGIN_HZ", "150"))
    except Exception:
        _F2_MARGIN_HZ = 150.0
    try:
        _AMBIG_RATIO_THR = float(os.environ.get("VOWEL_AMBIG_RATIO", "1.25"))
    except Exception:
        _AMBIG_RATIO_THR = 1.25
    try:
        _AMBIG_D2_MARGIN = float(os.environ.get("VOWEL_AMBIG_D2_MARGIN", "0.0"))
    except Exception:
        _AMBIG_D2_MARGIN = 0.0


def _two_stage_candidates(f2_hz: float) -> Tuple[Tuple[int, ...], bool]:
    """Return (candidate_ids, ambiguous_group). Always includes 'a'(1)."""
    if not _TWO_STAGE_ENABLED:
        return (1, 2, 3, 4, 5), True
    thr = float(_F2_THR_HZ)
    m = float(_F2_MARGIN_HZ)
    if f2_hz >= thr + m:
        return (1, 2, 4), False  # front (i/e)
    if f2_hz <= thr - m:
        return (1, 3, 5), False  # back (u/o)
    return (1, 2, 3, 4, 5), True


def _hz_to_mel(hz: float) -> float:
    # Standard mel scale
    # mel = 2595 * log10(1 + hz/700)
    if not math.isfinite(hz) or hz <= 0:
        return float("nan")
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _group_of_vowel_id(vid: int) -> str:
    # 3-way groups that matter to you:
    #    - "a"         : 1
    #    - "front"     : 2(i),4(e)
    #    - "back"      : 3(u),5(o)
    if vid == 1:
        return "a"
    if vid in (2, 4):
        return "front"
    if vid in (3, 5):
        return "back"
    return "other"


def _group_prior_from_prior(prior: Dict[int, float]) -> Dict[str, float]:
    pa = float(prior.get(1, 0.0))
    pf = float(prior.get(2, 0.0)) + float(prior.get(4, 0.0))
    pb = float(prior.get(3, 0.0)) + float(prior.get(5, 0.0))
    # normalize (keep stable even if some keys missing)
    s = pa + pf + pb
    if s <= 0:
        return {"a": 1.0/3.0, "front": 1.0/3.0, "back": 1.0/3.0}
    return {"a": pa/s, "front": pf/s, "back": pb/s}


def _min_d2_in_group(d2_map: Dict[int, float], group_name: str) -> Tuple[Optional[int], float]:
    best_vid: Optional[int] = None
    best_d2: float = float("inf")
    for vid, d2 in d2_map.items():
        if _group_of_vowel_id(int(vid)) != group_name:
            continue
        if float(d2) < best_d2:
            best_d2 = float(d2)
            best_vid = int(vid)
    return best_vid, best_d2


def _get_uncertain_params(best_tau: Optional[float]) -> Tuple[float, float]:
    """
    Return (ratio, tau_floor).
    Priority:
      1) env vars VOWEL_UNCERTAIN_RATIO / VOWEL_UNCERTAIN_TAU_FLOOR
      2) gauss_json["uncertain"] fields
      3) defaults (ratio=1.2, tau_floor=16.0)
    """
    ratio = None
    tau_floor = None
    try:
        if os.environ.get("VOWEL_UNCERTAIN_RATIO"):
            ratio = float(os.environ["VOWEL_UNCERTAIN_RATIO"])
    except Exception:
        ratio = None
    try:
        if os.environ.get("VOWEL_UNCERTAIN_TAU_FLOOR"):
            tau_floor = float(os.environ["VOWEL_UNCERTAIN_TAU_FLOOR"])
    except Exception:
        tau_floor = None

    if _GAUSS is not None:
        _unc = _GAUSS.get("uncertain", {}) if isinstance(_GAUSS, dict) else {}
        if ratio is None:
            try:
                ratio = float(_unc.get("ratio", 1.2))
            except Exception:
                ratio = 1.2
        if tau_floor is None:
            try:
                tau_floor = float(_unc.get("tau", 16.0))
            except Exception:
                tau_floor = 16.0
    else:
        if ratio is None:
            ratio = 1.2
        if tau_floor is None:
            tau_floor = 16.0

    return float(ratio), float(tau_floor)


def _maha2(x0: float, x1: float, mu0: float, mu1: float, inv: Any) -> float:
    """
    2D Mahalanobis distance: (x-mu)^T * inv * (x-mu)
    inv is expected to be [[a,b],[c,d]].
    """
    dx0 = x0 - mu0
    dx1 = x1 - mu1
    a = float(inv[0][0])
    b = float(inv[0][1])
    c = float(inv[1][0])
    d = float(inv[1][1])
    return (dx0 * (a * dx0 + b * dx1)) + (dx1 * (c * dx0 + d * dx1))


def classify_vowel(f1: Optional[float], f2: Optional[float]) -> int:
    # デバッグ用：初回呼び出し時に設定を出力
    global _PRINTED, _LAST_VOWEL_ID
    global _STAT_TOTAL, _STAT_AMBIG

    _STAT_TOTAL += 1

    _load_prior_from_env()
    _load_two_stage_from_env()
    if not _PRINTED:
        _load_gauss_from_env()
        if _GAUSS is not None:
            print(f"[vowel_classifier] mode=maha gauss={_GAUSS_PATH}", flush=True)
            if _PRIOR_SRC:
                print(f"[vowel_classifier] prior={_PRIOR_SRC} beta={_PRIOR_BETA}", flush=True)
            if _TWO_STAGE_ENABLED is not None:
                print(f"[vowel_classifier] two_stage={_TWO_STAGE_ENABLED} f2_thr_hz={_F2_THR_HZ} f2_margin_hz={_F2_MARGIN_HZ} amb_ratio={_AMBIG_RATIO_THR} amb_d2_margin={_AMBIG_D2_MARGIN}", flush=True)
        else:
            # fallback: keep old log shape for quick visual identification
            w2 = 3.0
            print(f"[vowel_classifier] mode=table table={VOWEL_TABLE} w2={w2} (gauss_disabled: {_GAUSS_LOAD_ERROR})", flush=True)
        _PRINTED = True

    if f1 is None or f2 is None:
        return 0
    if not math.isfinite(f1) or not math.isfinite(f2):
        return 0
    if f1 <= 0 or f2 <= 0:
        return 0

    # Convert to mel to reduce speaker / recording variance impact.
    f1m = _hz_to_mel(float(f1))
    f2m = _hz_to_mel(float(f2))
    if not math.isfinite(f1m) or not math.isfinite(f2m):
        return 0

    # ===== Preferred path: learned gauss (maha) =====
    if _GAUSS is not None:
        vowels = _GAUSS.get("vowels", {})

        # --- two-stage group split (front vs back) ---
        cand_ids, amb_group = _two_stage_candidates(float(f2))
        cand_set = set(int(x) for x in cand_ids)

        # 1) compute raw d2 for candidates
        d2_map: Dict[int, float] = {}
        tau_map: Dict[int, Any] = {}
        for vid_str, g in vowels.items():
            try:
                vid = int(vid_str)
            except Exception:
                continue
            if vid not in cand_set:
                continue
            mu = g.get("mu")
            inv = g.get("inv")
            if not isinstance(mu, (list, tuple)) or len(mu) != 2:
                continue
            if not isinstance(inv, (list, tuple)) or len(inv) != 2:
                continue
            d2 = _maha2(f1m, f2m, float(mu[0]), float(mu[1]), inv)
            d2_map[vid] = float(d2)
            tau_map[vid] = g.get("tau")

        if not d2_map:
            return 0

        # 2) pick best by raw d2 (for uncertainty logic)
        best_id = min(d2_map.items(), key=lambda kv: kv[1])[0]
        best_d2 = float(d2_map[best_id])
        best_tau = tau_map.get(best_id)

        # 3) detect ambiguity (only then apply prior)
        #    - group ambiguity: F2 near split threshold
        #    - score ambiguity: 2nd-best close to best
        amb_score = False
        if len(d2_map) >= 2:
            d2_sorted = sorted(d2_map.values())
            d2_best = float(d2_sorted[0])
            d2_2nd = float(d2_sorted[1])
            denom = max(d2_best, 1e-9)
            
            ratio_ok = ((d2_2nd / denom) < float(_AMBIG_RATIO_THR))
            if _AMBIG_D2_MARGIN and float(_AMBIG_D2_MARGIN) > 0.0:
                # B方向：ratioだけでambにせず、絶対差も小さい時だけamb扱い
                delta = float(d2_2nd - d2_best)
                delta_ok = (delta <= float(_AMBIG_D2_MARGIN))
                amb_score = bool(ratio_ok and delta_ok)
            else:
                amb_score = bool(ratio_ok)

        is_ambiguous = bool(amb_group or amb_score)

        if is_ambiguous:
            _STAT_AMBIG += 1

        if is_ambiguous and _PRIOR_SRC and _PRIOR_BETA != 0.0 and _PRIOR:
            # === Group-prior (3-way) only on ambiguity ===
            # We avoid collapsing e(4) just because its individual prior is small.
            gprior = _group_prior_from_prior(_PRIOR)

            # For each group, use the best (min) raw d2 inside the group,
            # then apply score = d2_best_in_group - beta*log(P(group)).
            # Choose the best group, then choose raw-d2-best within that group.
            best_group: Optional[str] = None
            best_group_score: float = float("inf")
            for gname in ("a", "front", "back"):
                g_vid, g_d2 = _min_d2_in_group(d2_map, gname)
                if g_vid is None:
                    continue
                pg = float(gprior.get(gname, 0.0))
                score = float(g_d2)
                if pg > 0.0:
                    score = score - (_PRIOR_BETA * math.log(pg))
                if score < best_group_score:
                    best_group_score = score
                    best_group = gname

            if best_group is not None:
                # within-group: raw d2 winner (no intra-group prior)
                g_vid, g_d2 = _min_d2_in_group(d2_map, best_group)
                if g_vid is not None:
                    best_id = int(g_vid)
                    best_d2 = float(g_d2)
                    best_tau = tau_map.get(best_id)

        # ===== Hold-last on uncertain =====
        # Use raw d2 (best_d2) for uncertainty logic to keep tau meaningful.
        ratio, tau_floor = _get_uncertain_params(best_tau)
        try:
            tau_best = float(best_tau) if best_tau is not None else tau_floor
        except Exception:
            tau_best = tau_floor
        tau_thr = max(tau_floor, tau_best) * ratio

        chosen = best_id
        # if uncertain and we have a previous vowel, keep it
        if best_d2 > tau_thr and 1 <= _LAST_VOWEL_ID <= 5:
            chosen = _LAST_VOWEL_ID

        # update last vowel only when returning a vowel (non-zero)
        if 1 <= chosen <= 5:
            _LAST_VOWEL_ID = chosen
        return chosen

    # ===== Fallback path: old table (kept for safety) =====
    best_id: int = 0
    best_d: float = float("inf")
    for vid, (_, mu_f1, mu_f2) in VOWEL_TABLE.items():
        var_f1, var_f2 = VOWEL_COV.get(vid, (1.0, 1.0))
        # NOTE: table is in Hz; keep the original behavior here (Hz-space diagonal distance).
        d = (((float(f1) - mu_f1) ** 2) / var_f1) + (((float(f2) - mu_f2) ** 2) / var_f2)
        if d < best_d:
            best_d = d
            best_id = vid

    # In fallback(table) mode, we also keep last vowel for stability.
    if 1 <= best_id <= 5:
        _LAST_VOWEL_ID = best_id
    return best_id