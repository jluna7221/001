# -*- coding: utf-8 -*-
"""
gen_negative_samples.py  (v10.0 - Forest Mask + neg_type + Bandit with Rich Reward + FN-aware quotas)

本版在 v9.0 的基础上，主要做了三件事：:contentReference[oaicite:1]{index=1}

1）仍然保持采样骨架不变：
    - 森林掩膜（LC 1-10）过滤；
    - hard / ultra-safe / fp_hotspot 的 neg_type 结构：
          -1 : 正样本（仅在 samples_all 中占位）
           0 : 普通 / hard 背景（ever_fire=True 的 BG）
           1 : ultra-safe 背景（ever_fire=False 的 BG）
           2 : fp_hotspot 背景（来自 fullgrid Top5% 假阳性区域）

2）上层：Bandit 控制器 + 更细致的 Reward（与 xgb_train_optuna 对应）：
    - 模板库 BANDIT_TEMPLATES：
          A ~ F：多种 (hard_ratio, bg_fp_ratio, bg_ultra_ratio,
                      focus_type, neg_budget_factor) 组合，
                  覆盖「偏 Hard / 偏 FN / 偏 FP / 偏 ultra-safe / 均衡」等情况。
    - Reward 仍然来自 feedback_it{k}.json 中的 metrics：
          R1, R5, AP, μ_pos, μ_neg, cov_pos_ge_high, neg_high_rate
      详见 _compute_reward_from_feedback()。

3）动作真正用上 FN 环境信号：
    - 从 feedback_it{k}.json 读取 slot_stats_env（year, bin, month, miss_rate,
      contamination, n_fn, n_fp_top5, difficulty 等）并在采样侧按 bin×month 聚合；
    - 在 Bandit 模板中增加：
          * focus_type: "balanced" / "fn_heavy" / "fp_heavy" /
                        "safe_calib" / "hard_focus"
          * neg_budget_factor: 控制本轮总负样本预算（相当于全局倍率）
    - 在 _distribute_quotas() 里，根据 focus_type 选择不同的配额权重设计：
          * fn_heavy: 高 miss_rate、高 n_fn 的槽位得到更多 Hard/BG 配额；
          * fp_heavy: 高 contamination、高 n_fp_top5 的槽位得到更多 BG 配额；
          * safe_calib: 偏向“低 miss_rate & 低 contamination”的 slot，用于稳校准；
          * hard_focus/balanced: 更依赖 difficulty 与 n_pos_in_slot 的综合。

这样，上一轮 fullgrid 环境中关于 FN/FP/难度的统计不再只是“评估指标”，
而是直接进入采样动作，用于控制“去哪采样”“采多少”“采什么类型”的负样本。
"""

from pathlib import Path
import json
import datetime as dt
import re
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

# 引入静态特征工具以读取 TIF
import build_static_features as static_mod

BASE = Path(r"G:\fire\outputs")
FIRMS_MASTER = BASE / "firms_master.parquet"
NEG_MASTER_FMT = BASE / "neg_master_it{iter}.parquet"
FEEDBACK_FMT = BASE / "feedback_it{iter}.json"
BANDIT_STATE_PATH = BASE / "bandit_state.json"

DATE_MIN = dt.date(2001, 1, 1)
DATE_MAX = dt.date(2020, 12, 31)
RANDOM_STATE = 2025

# ================= 配置参数 =================
TARGET_POS_NEG_RATIO = 5.0  # 全局正负比例（基础值），实际会乘以 neg_budget_factor

MAX_SAMPLES_CAP = 600000  # 硬性总量上限

# BG 内部目标比例（作为模板的 basic building block）
BASE_ULTRA_RATIO = 0.4
BASE_FP_RATIO = 0.3

# BG 中 ultra-safe 的最低占比，避免超安全样本完全消失
MIN_ULTRA_RATIO = 0.2

# Bandit 相关：模板库（动作空间）
# 每个模板包含：
#   - hard_ratio       : 在所有负样本中 Hard 的比例
#   - bg_fp_ratio      : BG 内部用于 FP-hotspot 的比例
#   - bg_ultra_ratio   : BG 内部用于 ultra-safe 的比例
#   - focus_type       : 控制配额如何根据 miss_rate / contamination / n_fn 调整
#   - neg_budget_factor: 控制本轮负样本总预算（相对 TARGET_POS_NEG_RATIO 的倍率）
BANDIT_TEMPLATES: Dict[str, Dict] = {
    # A：相对均衡，轻微偏 Hard
    "A": {
        "hard_ratio": 0.55,
        "bg_fp_ratio": 0.30,
        "bg_ultra_ratio": 0.30,
        "focus_type": "balanced",
        "neg_budget_factor": 1.0,
    },
    # B：偏 FN，Hard 比例高，整体多采一点
    "B": {
        "hard_ratio": 0.70,
        "bg_fp_ratio": 0.20,
        "bg_ultra_ratio": 0.25,
        "focus_type": "fn_heavy",
        "neg_budget_factor": 1.2,
    },
    # C：偏 FP，重挖假阳性热点
    "C": {
        "hard_ratio": 0.45,
        "bg_fp_ratio": 0.45,
        "bg_ultra_ratio": 0.25,
        "focus_type": "fp_heavy",
        "neg_budget_factor": 1.0,
    },
    # D：偏 ultra-safe，用于稳概率刻度
    "D": {
        "hard_ratio": 0.50,
        "bg_fp_ratio": 0.20,
        "bg_ultra_ratio": 0.40,
        "focus_type": "safe_calib",
        "neg_budget_factor": 0.8,
    },
    # E：极偏 Hard，适合“猛攻难槽位”
    "E": {
        "hard_ratio": 0.80,
        "bg_fp_ratio": 0.10,
        "bg_ultra_ratio": 0.20,
        "focus_type": "hard_focus",
        "neg_budget_factor": 1.1,
    },
    # F：BG 丰富，兼顾 FP 挖矿和 ultra-safe
    "F": {
        "hard_ratio": 0.40,
        "bg_fp_ratio": 0.40,
        "bg_ultra_ratio": 0.40,
        "focus_type": "balanced",
        "neg_budget_factor": 1.0,
    },
}

# Bandit Reward 相关（与 xgb_train_optuna v10.0 对应）:contentReference[oaicite:2]{index=2}
AP_TARGET = 0.60          # 希望 RL-val AP 不低于此阈值
NEG_HIGH_MAX = 0.05       # 负例中 p≥high_thr 的“可接受上限”，超过就惩罚
LAMBDA_AP = 0.6           # AP 低于阈值的惩罚强度
LAMBDA_GAP = 0.3          # (μ_pos - μ_neg) 的奖励权重
LAMBDA_NEG = 0.7          # neg_high_rate 超标的惩罚强度
UCB_C = 1.0               # UCB 中的探索系数
HIGH_PROB_THR = 0.75      # 和 xgb_train_optuna 中的 HIGH_PROB_THR 保持一致

# ===========================================

# ----------------- LC 辅助函数 -----------------


def _build_year2path_generic(dir_path: Path) -> dict:
    mapping = {}
    if not dir_path.exists():
        return mapping
    for p in dir_path.glob("*.tif"):
        m = re.search(r"(\d{4})", p.stem)
        if m:
            mapping[int(m.group(1))] = p
    return mapping


def _nearest_year_path(year: int, mapping: dict) -> Optional[Path]:
    if not mapping:
        return None
    years = sorted(mapping.keys())
    best = min(years, key=lambda y: abs(y - year))
    return mapping[best]


def _filter_by_forest_lc(df: pd.DataFrame,
                         is_grid: bool = False,
                         ref_year: int = 2018) -> pd.DataFrame:
    """
    核心过滤函数：仅保留 LC 值在 1~10 (森林) 范围内的点。
    - df: 必须包含 'lat', 'lon'。如果是样本数据还需包含 'year' 或 'date_key'。
    - is_grid: 如果是 grid，统一使用 ref_year 进行过滤。
    """
    print(f"[LC Filter] 开始过滤非森林区域 (保留 LC 1-10)... 输入行数: {len(df):,}")

    static_mod._init_lc_year2path()
    mapping = static_mod._LC_YEAR2PATH
    if not mapping:
        print("[LC Filter] 警告：未找到 LC TIF 文件，跳过过滤。")
        return df

    lons = df["lon"].to_numpy()
    lats = df["lat"].to_numpy()

    if is_grid:
        tif_path = _nearest_year_path(ref_year, mapping)
        if tif_path is None:
            return df
        vals = static_mod.sample_tif_points(tif_path, lons, lats)
    else:
        if "year" not in df.columns:
            temp_dt = pd.to_datetime(df["date_key"], errors="coerce")
            years = temp_dt.dt.year.fillna(ref_year).astype(int).to_numpy()
        else:
            years = df["year"].to_numpy().astype(int)

        vals = static_mod.sample_yearly_tif_points(years, lons, lats, mapping)

    vals = np.nan_to_num(vals, nan=-9999)
    mask = (vals >= 1) & (vals <= 10)

    filtered_df = df[mask].copy()
    print(f"[LC Filter] 过滤完成。剩余: {len(filtered_df):,} "
          f"(剔除率: {1 - len(filtered_df) / max(len(df), 1):.1%})")
    return filtered_df


# ----------------- FIRMS 及 GRID 构建 -----------------


def _latlon_to_bin(lat: np.ndarray,
                   lon: np.ndarray,
                   grid_deg: float = 0.1) -> np.ndarray:
    i = np.floor((lat + 90.0) / grid_deg).astype(int)
    j = np.floor((lon + 180.0) / grid_deg).astype(int)
    return np.array([f"{ii}_{jj}" for ii, jj in zip(i, j)], dtype=object)


def _load_firms_master(grid_deg: float = 0.1) -> pd.DataFrame:
    """
    供训练 & fullgrid 评估使用的 FIRMS 总表读取函数。
    """
    if not FIRMS_MASTER.exists():
        raise FileNotFoundError(f"未找到 FIRMS 总表：{FIRMS_MASTER}")
    df = pd.read_parquet(FIRMS_MASTER)
    df["date_key"] = pd.to_datetime(df["date_key"], errors="coerce")
    df = df.dropna(subset=["date_key", "lon", "lat"]).reset_index(drop=True)
    df = df[(df["date_key"].dt.date >= DATE_MIN) &
            (df["date_key"].dt.date <= DATE_MAX)]
    df["year"] = df["date_key"].dt.year.astype(int)
    df["month"] = df["date_key"].dt.month.astype(int)
    df["bin"] = _latlon_to_bin(
        df["lat"].to_numpy(), df["lon"].to_numpy(), grid_deg=grid_deg
    )
    return df


def _build_grid_df_from_firms(firms: pd.DataFrame,
                              grid_deg: float = 0.1) -> pd.DataFrame:
    """
    仅根据 FIRMS 空间范围构建全省森林格网中心点。
    """
    lat_min, lat_max = firms["lat"].min(), firms["lat"].max()
    lon_min, lon_max = firms["lon"].min(), firms["lon"].max()
    lat_edges = np.arange(lat_min - grid_deg, lat_max + grid_deg, grid_deg)
    lon_edges = np.arange(lon_min - grid_deg, lon_max + grid_deg, grid_deg)
    centers = [
        (la + grid_deg / 2, lo + grid_deg / 2)
        for la in lat_edges for lo in lon_edges
    ]
    grid_df = pd.DataFrame(centers, columns=["lat_center", "lon_center"])
    grid_df = grid_df.rename(columns={"lat_center": "lat",
                                      "lon_center": "lon"})

    grid_df["bin"] = _latlon_to_bin(
        grid_df["lat"].to_numpy(),
        grid_df["lon"].to_numpy(),
        grid_deg=grid_deg,
    )
    grid_df = grid_df.drop_duplicates(subset=["bin"]).reset_index(drop=True)
    return grid_df


# ----------------- Bandit 工具 -----------------


def _load_bandit_state() -> Dict:
    """
    读取 / 初始化 bandit_state.json。

    对每个模板，除了 n_plays / sum_reward 以外，强制刷新：
        - hard_ratio / bg_fp_ratio / bg_ultra_ratio
        - focus_type / neg_budget_factor

    这样即使你在代码里调整了模板的“静态参数”，旧的 state 也会自动对齐新配置。
    """
    if BANDIT_STATE_PATH.exists():
        try:
            with open(BANDIT_STATE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            state = {}
    else:
        state = {}

    templates_state = state.get("templates", {})

    for tid, cfg in BANDIT_TEMPLATES.items():
        t = templates_state.get(tid, {})
        # 同步静态配置
        t["hard_ratio"] = float(cfg["hard_ratio"])
        t["bg_fp_ratio"] = float(cfg["bg_fp_ratio"])
        t["bg_ultra_ratio"] = float(cfg["bg_ultra_ratio"])
        t["focus_type"] = cfg.get("focus_type", "balanced")
        t["neg_budget_factor"] = float(cfg.get("neg_budget_factor", 1.0))
        # 动态统计
        t.setdefault("n_plays", 0)
        t.setdefault("sum_reward", 0.0)
        templates_state[tid] = t

    state["templates"] = templates_state
    state.setdefault("last_iter", None)
    state.setdefault("last_template_id", None)
    state.setdefault("total_plays", 0)
    return state


def _save_bandit_state(state: Dict):
    BANDIT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BANDIT_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _compute_reward_from_feedback(feedback_path: Path) -> Optional[float]:
    """
    从 feedback_it{k}.json 中提取上一轮指标并计算 Reward。

    依赖 xgb_train_optuna 中 metrics 字段，形如：
        "metrics": {
            "val_ap": ...,
            "fullgrid_recall_top1": ...,
            "fullgrid_recall_top5": ...,
            "mu_pos": ...,
            "mu_neg": ...,
            "cov_pos_ge_high": ...,
            "neg_high_rate": ...
        }

    Reward 设计：
        pos_term = 0.5*R5 + 0.2*R1 + 0.3*cov_pos_high
        gap_term = max(0, μ_pos - μ_neg)

        penalty_ap  = max(0, AP_target - AP)
        penalty_neg = max(0, neg_high_rate - NEG_HIGH_MAX)

        Reward = pos_term + λ_gap * gap_term
                 - λ_ap * penalty_ap
                 - λ_neg * penalty_neg
    """
    if not feedback_path.exists():
        return None
    try:
        with open(feedback_path, "r", encoding="utf-8") as f:
            fb = json.load(f)
        metrics = fb.get("metrics", {}) or {}
        R1 = float(metrics.get("fullgrid_recall_top1", 0.0))
        R5 = float(metrics.get("fullgrid_recall_top5", 0.0))
        AP = float(metrics.get("val_ap", 0.0))

        mu_pos = float(metrics.get("mu_pos", 0.0))
        mu_neg = float(metrics.get("mu_neg", 0.0))
        cov_pos = float(metrics.get("cov_pos_ge_high",
                                    metrics.get("fire_highbin_ratio", 0.0)))
        neg_high_rate = float(metrics.get("neg_high_rate", 0.0))

        gap = max(0.0, mu_pos - mu_neg)

        # 额外稳定性与误差项
        topk = fb.get("fullgrid_topk_metrics", {}) or {}
        fp_rate_high = float(topk.get("overall_fp_rate_high", 0.0))
        spatial_r5 = float(metrics.get("spatial_holdout_recall_top5", 0.0))

        pos_term = 0.45 * R5 + 0.20 * R1 + 0.25 * cov_pos + 0.10 * spatial_r5
        gap_term = gap

        penalty_ap = max(0.0, AP_TARGET - AP)
        penalty_neg = max(0.0, neg_high_rate - NEG_HIGH_MAX)
        penalty_fp = 0.5 * fp_rate_high

        stab_std_r5 = float(metrics.get("stability_std_recall_top5", 0.0))
        stab_std_ece = float(metrics.get("stability_std_ece", 0.0))
        stab_penalty = 0.2 * stab_std_r5 + 0.1 * stab_std_ece

        reward = (
            pos_term
            + LAMBDA_GAP * gap_term
            - LAMBDA_AP * penalty_ap
            - LAMBDA_NEG * penalty_neg
            - penalty_fp
            - stab_penalty
        )

        print(
            "[Bandit] 解析上一轮反馈:\n"
            f"         R1={R1:.4f}, R5={R5:.4f}, AP={AP:.4f}\n"
            f"         μ_pos={mu_pos:.4f}, μ_neg={mu_neg:.4f}, "
            f"cov_pos_ge_high={cov_pos:.4f}, neg_high_rate={neg_high_rate:.4f}\n"
            f"         -> Reward={reward:.4f}"
        )
        return float(reward)
    except Exception as e:
        print(f"[Bandit] 从 {feedback_path} 解析 Reward 失败：{e}")
        return None


def _bandit_choose_template(
    iter_idx: int,
    prev_feedback_path: Path,
) -> Tuple[str, float, float, float, str, float]:
    """
    Bandit 主逻辑：
        - 读取 / 初始化 bandit_state.json；
        - 若 state 中记录的 last_iter == iter_idx-1，
          则用 prev_feedback_path 更新上一轮模板的 reward；
        - 使用 UCB1 选择本轮模板；
        - 更新 state.last_iter / last_template_id，并保存。

    返回：
        (chosen_tid, hard_ratio, bg_fp_ratio, bg_ultra_ratio,
         focus_type, neg_budget_factor)
    """
    state = _load_bandit_state()
    templates_state = state["templates"]

    # 1）首先用上一轮反馈更新上一轮模板的 reward
    last_iter = state.get("last_iter", None)
    last_tid = state.get("last_template_id", None)
    if last_iter is not None and last_tid is not None and last_iter == iter_idx - 1:
        reward = _compute_reward_from_feedback(prev_feedback_path)
        if reward is not None and last_tid in templates_state:
            t = templates_state[last_tid]
            t["n_plays"] = int(t.get("n_plays", 0)) + 1
            t["sum_reward"] = float(t.get("sum_reward", 0.0) + reward)
            print(f"[Bandit] 模板 {last_tid} 更新：n={t['n_plays']}, "
                  f"sum_reward={t['sum_reward']:.4f}")
            state["total_plays"] = int(state.get("total_plays", 0)) + 1

    # 2）选择本轮模板：优先尝试“未玩过”的模板
    untried = [tid for tid, t in templates_state.items() if t["n_plays"] == 0]
    if untried:
        chosen_tid = sorted(untried)[0]  # 简单起见，按 ID 顺序
        print(f"[Bandit] 选择尚未尝试过的模板: {chosen_tid}")
    else:
        # UCB1：mean_reward + c * sqrt(2 ln T / n_plays)
        T = max(1, int(state.get("total_plays", 1)))
        best_val = -1e9
        chosen_tid = None
        for tid, t in templates_state.items():
            n = max(1, int(t["n_plays"]))
            mean_r = float(t["sum_reward"] / n)
            bonus = UCB_C * np.sqrt(2.0 * np.log(T + 1) / n)
            ucb_val = mean_r + bonus
            print(f"[Bandit] 模板 {tid}: mean={mean_r:.4f}, bonus={bonus:.4f}, "
                  f"UCB={ucb_val:.4f}")
            if ucb_val > best_val:
                best_val = ucb_val
                chosen_tid = tid
        print(f"[Bandit] UCB 选择模板: {chosen_tid}")

    # 3）回写 state
    state["last_iter"] = int(iter_idx)
    state["last_template_id"] = chosen_tid
    _save_bandit_state(state)

    cfg = templates_state[chosen_tid]
    hard_ratio = float(cfg["hard_ratio"])
    bg_fp_ratio = float(cfg["bg_fp_ratio"])
    bg_ultra_ratio = float(cfg["bg_ultra_ratio"])
    focus_type = cfg.get("focus_type", "balanced")
    neg_budget_factor = float(cfg.get("neg_budget_factor", 1.0))

    # 轻微防御：保证 ultra 有下限、bg_fp+bg_ultra 不超过 0.9
    bg_ultra_ratio = max(bg_ultra_ratio, MIN_ULTRA_RATIO)
    if bg_fp_ratio + bg_ultra_ratio > 0.9:
        bg_ultra_ratio = max(MIN_ULTRA_RATIO, 0.9 - bg_fp_ratio)

    # 连续微调（基于上一轮反馈）
    try:
        with open(prev_feedback_path, "r", encoding="utf-8") as f:
            fb = json.load(f)
        m = fb.get("metrics", {}) or {}
        topk = fb.get("fullgrid_topk_metrics", {}) or {}
        r5 = float(m.get("fullgrid_recall_top5", 0.0))
        fp_rate_high = float(topk.get("overall_fp_rate_high", 0.0))
        neg_high = float(m.get("neg_high_rate", 0.0))
        stab_std_r5 = float(m.get("stability_std_recall_top5", 0.0))
        stab_std_ece = float(m.get("stability_std_ece", 0.0))
        if r5 < 0.30:
            hard_ratio = min(0.7, hard_ratio + 0.05)
        if fp_rate_high > 0.20 or neg_high > 0.10:
            bg_ultra_ratio = min(0.6, bg_ultra_ratio + 0.05)
            bg_fp_ratio = max(0.05, bg_fp_ratio - 0.05)
        if stab_std_r5 > 0.10 or stab_std_ece > 0.05:
            # 稳定性较差，适度增加 ultra-safe 与总预算，降低 fp
            bg_ultra_ratio = min(0.7, bg_ultra_ratio + 0.05)
            bg_fp_ratio = max(0.05, bg_fp_ratio - 0.05)
            neg_budget_factor = min(1.3, neg_budget_factor + 0.05)
        bg_fp_ratio = float(max(0.0, min(0.9, bg_fp_ratio)))
        bg_ultra_ratio = float(max(MIN_ULTRA_RATIO, min(0.9, bg_ultra_ratio)))
        hard_ratio = float(max(0.0, min(0.9, hard_ratio)))
    except Exception:
        pass

    print(
        f"[Bandit] iter={iter_idx} 采用模板 {chosen_tid}: "
        f"hard_ratio={hard_ratio:.2f}, "
        f"bg_fp_ratio={bg_fp_ratio:.2f}, "
        f"bg_ultra_ratio={bg_ultra_ratio:.2f}, "
        f"focus_type={focus_type}, "
        f"neg_budget_factor={neg_budget_factor:.2f}"
    )
    return chosen_tid, hard_ratio, bg_fp_ratio, bg_ultra_ratio, focus_type, neg_budget_factor


# ----------------- 从 feedback 读取 RL 信号（slot / FP/FN-hotspot） -----------------


def _load_feedback_slot_info(feedback_path: Path) -> pd.DataFrame:
    """
    从 feedback_it{k}.json 读取 slot 级环境信息 slot_stats_env，
    并在采样侧按 (bin, month) 聚合。

    主要字段：
        - difficulty    : 围绕 Top5% 定义的综合难度
        - miss_rate     : FN 占比
        - contamination : Top5% 内 FP 占比
        - n_fn          : 假阴性数量
        - n_fp_top5     : Top5% 内的假阳性数量
        - n_pos         : 该槽位内真实火点数量

    返回列：
        ['bin', 'month', 'difficulty', 'miss_rate',
         'contamination', 'n_fn', 'n_fp_top5', 'n_pos']
    """
    try:
        with open(feedback_path, "r", encoding="utf-8") as f:
            fb = json.load(f)
        env_slots = fb.get("slot_stats_env", [])
        slot_raw = pd.DataFrame(env_slots)
        if slot_raw.empty:
            return pd.DataFrame(
                columns=[
                    "bin",
                    "month",
                    "difficulty",
                    "miss_rate",
                    "contamination",
                    "n_fn",
                    "n_fp_top5",
                    "n_pos",
                ]
            )

        # 保底列
        for col in ["year", "bin", "month"]:
            if col not in slot_raw.columns:
                slot_raw[col] = 0
        for col in ["difficulty", "miss_rate", "contamination"]:
            if col not in slot_raw.columns:
                slot_raw[col] = 0.0
        for col in ["n_fn", "n_fp_top5", "n_pos"]:
            if col not in slot_raw.columns:
                slot_raw[col] = 0

        slot_raw["bin"] = slot_raw["bin"].astype(str)
        slot_raw["month"] = slot_raw["month"].astype(int)
        slot_raw["difficulty"] = (
            slot_raw["difficulty"].astype(float).fillna(0.1).clip(0.0, 1.0)
        )
        slot_raw["miss_rate"] = (
            slot_raw["miss_rate"].astype(float).fillna(0.0).clip(0.0, 1.0)
        )
        slot_raw["contamination"] = (
            slot_raw["contamination"].astype(float).fillna(0.0).clip(0.0, 1.0)
        )
        slot_raw["n_fn"] = slot_raw["n_fn"].astype(int).clip(lower=0)
        slot_raw["n_fp_top5"] = slot_raw["n_fp_top5"].astype(int).clip(lower=0)
        slot_raw["n_pos"] = slot_raw["n_pos"].astype(int).clip(lower=0)

        # 采样侧只关心 bin×month 的总体情况：按 (bin, month) 聚合
        slot = (
            slot_raw.groupby(["bin", "month"], as_index=False)
            .agg(
                difficulty=("difficulty", "mean"),
                miss_rate=("miss_rate", "mean"),
                contamination=("contamination", "mean"),
                n_fn=("n_fn", "sum"),
                n_fp_top5=("n_fp_top5", "sum"),
                n_pos=("n_pos", "sum"),
            )
            .reset_index(drop=True)
        )
        return slot[
            [
                "bin",
                "month",
                "difficulty",
                "miss_rate",
                "contamination",
                "n_fn",
                "n_fp_top5",
                "n_pos",
            ]
        ]
    except Exception as e:
        print(f"[slot] 从 {feedback_path} 读取 slot_stats_env 失败：{e}")
        return pd.DataFrame(
            columns=[
                "bin",
                "month",
                "difficulty",
                "miss_rate",
                "contamination",
                "n_fn",
                "n_fp_top5",
                "n_pos",
            ]
        )


def _load_fp_hotspots(feedback_path: Path) -> pd.DataFrame:
    """
    从 feedback_it{k}.json 读取 fullgrid FP-hotspot 信息：
        [{"year": y, "month": m, "bin": "...", "n_fp_top5": k, ...}, ...]
    """
    try:
        with open(feedback_path, "r", encoding="utf-8") as f:
            fb = json.load(f)
        fps = fb.get("fp_hotspots", [])
        df = pd.DataFrame(fps)
        if df.empty:
            return pd.DataFrame(columns=["year", "month", "bin", "n_fp_top5"])
        df["year"] = df["year"].astype(int)
        df["month"] = df["month"].astype(int)
        df["bin"] = df["bin"].astype(str)
        if "n_fp_top5" not in df.columns:
            df["n_fp_top5"] = 1
        df["n_fp_top5"] = df["n_fp_top5"].astype(int)
        return df[["year", "month", "bin", "n_fp_top5"]]
    except Exception:
        return pd.DataFrame(columns=["year", "month", "bin", "n_fp_top5"])


def _load_fn_hotspots(feedback_path: Path) -> pd.DataFrame:
    """
    预留接口：从 feedback_it{k}.json 读取 FN-hotspot 信息（如果存在）：
        [{"year": y, "month": m, "bin": "...", "n_fn": k,
          "miss_rate": ..., "difficulty": ...}, ...]

    当前版本 xgb_train_optuna 里暂未显式写入 fn_hotspots，
    因此本函数通常返回空表；一旦你在训练侧增加该字段，
    这里即可无缝接入。
    """
    try:
        with open(feedback_path, "r", encoding="utf-8") as f:
            fb = json.load(f)
        fns = fb.get("fn_hotspots", [])
        df = pd.DataFrame(fns)
        if df.empty:
            return pd.DataFrame(columns=["year", "month", "bin", "n_fn", "miss_rate", "difficulty"])
        df["year"] = df["year"].astype(int)
        df["month"] = df["month"].astype(int)
        df["bin"] = df["bin"].astype(str)
        if "n_fn" not in df.columns:
            df["n_fn"] = 1
        if "miss_rate" not in df.columns:
            df["miss_rate"] = 1.0
        if "difficulty" not in df.columns:
            df["difficulty"] = 1.0
        df["n_fn"] = df["n_fn"].astype(int)
        df["miss_rate"] = df["miss_rate"].astype(float)
        df["difficulty"] = df["difficulty"].astype(float)
        return df[["year", "month", "bin", "n_fn", "miss_rate", "difficulty"]]
    except Exception:
        return pd.DataFrame(columns=["year", "month", "bin", "n_fn", "miss_rate", "difficulty"])


# ----------------- 配额分配 -----------------


def _distribute_quotas(
    pos_df: pd.DataFrame,
    slot_info: pd.DataFrame,
    hard_ratio: float,
    focus_type: str,
    neg_budget_factor: float,
) -> pd.DataFrame:
    """
    根据正样本分布 + 槽位难度/错判情况，按 bin×month 分配 Hard / BG 配额。

    - 基础总量：
        n_neg_total = min( len(pos_df) * TARGET_POS_NEG_RATIO * neg_budget_factor,
                           MAX_SAMPLES_CAP )

    - Hard / BG 拆分：
        Hard = hard_ratio * n_neg_total
        BG   = n_neg_total - Hard

    - 槽位权重（weight_hard / weight_bg）取决于 focus_type：
        * "balanced"  ：使用 difficulty + miss_rate 作为 Hard 权重主驱动；
        * "fn_heavy"  ：进一步放大高 miss_rate、高 n_fn 的槽位；
        * "fp_heavy"  ：更关注 contamination / n_fp_top5，高 FP 槽位 BG 配额更多；
        * "safe_calib"：Hard 适度，BG 更偏向低 miss_rate & 低 contamination 的槽位；
        * "hard_focus"：主要依赖 difficulty（类似旧版），偏向“难槽位”。

    返回表包含：
        ['bin', 'month', 'n_pos_in_slot', 'hard_quota', 'bg_quota', ...]
    """
    n_pos = len(pos_df)
    if n_pos == 0:
        raise ValueError("pos_df 为空，无法分配负样本配额。")

    # 总负样本预算：基础比例 * 模板系数
    n_neg_total = int(n_pos * TARGET_POS_NEG_RATIO * max(neg_budget_factor, 0.1))
    n_neg_total = min(n_neg_total, MAX_SAMPLES_CAP)

    n_hard_target = int(n_neg_total * hard_ratio)
    n_bg_target = n_neg_total - n_hard_target

    print(
        f"[Quota] 正样本={n_pos:,} | 目标负样本={n_neg_total:,} "
        f"(Hard={n_hard_target:,}, BG={n_bg_target:,}) "
        f"| focus_type={focus_type}, neg_budget_factor={neg_budget_factor:.2f}"
    )

    # 按 bin×month 看正样本分布
    slots = pos_df.groupby(["bin", "month"]).size().reset_index(
        name="n_pos_in_slot"
    )
    slots["bin"] = slots["bin"].astype(str)
    slots["month"] = slots["month"].astype(int)

    # 合并反馈中的 slot 环境信息
    if not slot_info.empty:
        slot_info = slot_info.copy()
        slot_info["bin"] = slot_info["bin"].astype(str)
        slot_info["month"] = slot_info["month"].astype(int)
        slots = slots.merge(slot_info, on=["bin", "month"], how="left")
    else:
        # 保底列
        slots["difficulty"] = 0.1
        slots["miss_rate"] = 0.0
        slots["contamination"] = 0.0
        slots["n_fn"] = 0
        slots["n_fp_top5"] = 0

    # 填充缺失 & 归一化
    slots["difficulty"] = (
        slots.get("difficulty", pd.Series(0.1, index=slots.index))
        .astype(float)
        .fillna(0.1)
        .clip(0.0, 1.0)
    )
    slots["miss_rate"] = (
        slots.get("miss_rate", pd.Series(0.0, index=slots.index))
        .astype(float)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    slots["contamination"] = (
        slots.get("contamination", pd.Series(0.0, index=slots.index))
        .astype(float)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    # 注意：这里一定要先 fillna 再 astype(int)，否则会出现 IntCastingNaNError
    slots["n_fn"] = (
        slots.get("n_fn", pd.Series(0, index=slots.index))
        .fillna(0)
        .astype(int)
        .clip(lower=0)
    )
    slots["n_fp_top5"] = (
        slots.get("n_fp_top5", pd.Series(0, index=slots.index))
        .fillna(0)
        .astype(int)
        .clip(lower=0)
    )

    # 填充缺失 & 归一化
    slots["difficulty"] = (
        slots.get("difficulty", pd.Series(0.1, index=slots.index))
        .astype(float)
        .fillna(0.1)
        .clip(0.0, 1.0)
    )
    slots["miss_rate"] = (
        slots.get("miss_rate", pd.Series(0.0, index=slots.index))
        .astype(float)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    slots["contamination"] = (
        slots.get("contamination", pd.Series(0.0, index=slots.index))
        .astype(float)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    # 注意：这里一定要先 fillna 再 astype(int)，否则会出现 IntCastingNaNError
    slots["n_fn"] = (
        slots.get("n_fn", pd.Series(0, index=slots.index))
        .fillna(0)
        .astype(int)
        .clip(lower=0)
    )
    slots["n_fp_top5"] = (
        slots.get("n_fp_top5", pd.Series(0, index=slots.index))
        .fillna(0)
        .astype(int)
        .clip(lower=0)
    )


    max_n_fn = float(slots["n_fn"].max()) if len(slots) > 0 else 0.0
    max_n_fp = float(slots["n_fp_top5"].max()) if len(slots) > 0 else 0.0
    slots["fn_norm"] = (
        slots["n_fn"] / max(max_n_fn, 1.0)
        if max_n_fn > 0
        else 0.0
    )
    slots["fp_norm"] = (
        slots["n_fp_top5"] / max(max_n_fp, 1.0)
        if max_n_fp > 0
        else 0.0
    )

    base = slots["n_pos_in_slot"].astype(float)

    ft = (focus_type or "balanced").lower()

    if ft == "fn_heavy":
        # 更强地放大 FN 重灾区
        slots["weight_hard"] = base * (
            1.0 + 4.0 * slots["miss_rate"] + 3.0 * slots["fn_norm"]
        )
        slots["weight_bg"] = base * (
            1.0 + 1.5 * slots["miss_rate"] + 1.0 * slots["fn_norm"]
        )
    elif ft == "fp_heavy":
        # 高 contamination / 高 n_fp_top5 槽位，BG 配额更大，用来压顽固 FP
        slots["weight_hard"] = base * (
            1.0 + 1.5 * slots["contamination"] + 0.5 * slots["fp_norm"]
        )
        slots["weight_bg"] = base * (
            1.0 + 3.0 * slots["contamination"] + 2.0 * slots["fp_norm"]
        )
    elif ft == "safe_calib":
        # Hard 适度，BG 偏向“容易”的安全区，帮助校准
        inv_miss = 1.0 - slots["miss_rate"]
        inv_cont = 1.0 - slots["contamination"]
        slots["weight_hard"] = base * (1.0 + 0.5 * slots["difficulty"])
        slots["weight_bg"] = base * (1.0 + inv_miss + inv_cont)
    elif ft == "hard_focus":
        # 更集中在 difficulty 高的槽位
        slots["weight_hard"] = base * (1.0 + 5.0 * (slots["difficulty"] ** 2))
        slots["weight_bg"] = base
    else:
        # balanced：既看 difficulty，也看 miss_rate / contamination
        slots["weight_hard"] = base * (
            1.0
            + 3.0 * (slots["difficulty"] ** 2)
            + 1.5 * slots["miss_rate"]
        )
        slots["weight_bg"] = base * (
            1.0 + 0.5 * slots["miss_rate"] + 0.5 * slots["contamination"]
        )

    sum_w_hard = float(slots["weight_hard"].sum())
    sum_w_bg = float(slots["weight_bg"].sum())

    if sum_w_hard <= 0:
        sum_w_hard = 1.0
    if sum_w_bg <= 0:
        sum_w_bg = 1.0

    slots["hard_quota"] = (
        slots["weight_hard"] / sum_w_hard * n_hard_target
    ).round().astype(int)
    slots["bg_quota"] = (
        slots["weight_bg"] / sum_w_bg * n_bg_target
    ).round().astype(int)

    # BG 至少 1，防止完全缺槽位
    slots["bg_quota"] = slots["bg_quota"].clip(lower=1)

    cap_bg = max(1, int(round(n_bg_target * 0.02)))
    cap_hard = max(1, int(round(n_hard_target * 0.02)))
    over_bg = slots["bg_quota"] - slots["bg_quota"].clip(upper=cap_bg)
    over_h = slots["hard_quota"] - slots["hard_quota"].clip(upper=cap_hard)
    slots["bg_quota"] = slots["bg_quota"].clip(upper=cap_bg)
    slots["hard_quota"] = slots["hard_quota"].clip(upper=cap_hard)
    rem_bg = int(max(0, over_bg.sum()))
    rem_h = int(max(0, over_h.sum()))
    if rem_bg > 0:
        wbg = slots["weight_bg"].to_numpy(dtype=float)
        wbg = wbg / max(wbg.sum(), 1e-9)
        add = np.random.default_rng(RANDOM_STATE).choice(len(slots), size=rem_bg, replace=True, p=wbg)
        for i in add:
            slots.at[slots.index[i], "bg_quota"] += 1
    if rem_h > 0:
        wh = slots["weight_hard"].to_numpy(dtype=float)
        wh = wh / max(wh.sum(), 1e-9)
        add = np.random.default_rng(RANDOM_STATE).choice(len(slots), size=rem_h, replace=True, p=wh)
        for i in add:
            slots.at[slots.index[i], "hard_quota"] += 1

    return slots


# ----------------- 负样本采样 -----------------


def _sample_negatives(
    firms: pd.DataFrame,
    grid_df: pd.DataFrame,
    dates,
    quotas: pd.DataFrame,
    fp_hotspots: pd.DataFrame,
    hard_ratio: float,
    bg_fp_ratio: float,
    bg_ultra_ratio: float,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """
    采样逻辑：

    1）Hard Negative（近火）：
        - 同 bin，|Δt|<=3 天但当天没烧；
        - is_hard=True, neg_type=0。

    2）BG Negative：
        - 由 Bandit 选择的 (hard_ratio, bg_fp_ratio, bg_ultra_ratio) 决定 BG 内部结构：
            * ultra-safe BG (neg_type=1) : 目标比例 bg_ultra_ratio
            * fp_hotspot BG (neg_type=2) : 目标比例 bg_fp_ratio
            * normal BG (neg_type=0)     : 其余比例
        - ultra-safe: ever_fire=False 的 bin；
        - fp_hotspot: 来自上一轮 fullgrid Top5% 假阳性区域的 bin；
        - normal: ever_fire=True 且非 fp_hotspot 的 bin。

    fp_hotspots 表格式：
        year, month, bin, n_fp_top5
    """
    quota_map = {
        (str(r["bin"]), int(r["month"])): (int(r["hard_quota"]),
                                           int(r["bg_quota"]))
        for _, r in quotas.iterrows()
    }

    firms["date_dt"] = pd.to_datetime(firms["date_key"], errors="coerce")
    firms["year"] = firms["date_dt"].dt.year.astype(int)

    pos_by = firms.groupby(["year", "bin", "month"])
    neg_rows = []

    print(f"[Sampling] 开始遍历 {len(pos_by)} 个时空槽位...")

    has_ever_fire_col = "ever_fire" in grid_df.columns

    # 可观测性掩码（若存在 eval_env）
    obs_map = {}
    try:
        from eval_fullgrid_env import get_eval_env_path
        env_path = get_eval_env_path()
        if env_path.exists():
            env_df = pd.read_parquet(env_path)
            env_df["date_key"] = pd.to_datetime(env_df["date_key"], errors="coerce").dt.strftime("%Y-%m-%d")
            obs_map = env_df.groupby("bin")["date_key"].apply(set).to_dict()
            print(f"[Sampling] 载入可观测掩码：bins={len(obs_map)}")
    except Exception as e:
        print(f"[Sampling] 可观测掩码载入失败（不影响继续）：{e}")

    if has_ever_fire_col:
        ultra_bins_all = grid_df[grid_df["ever_fire"] == False]["bin"].astype(
            str
        ).unique()
        normal_bins_all = grid_df[grid_df["ever_fire"] == True]["bin"].astype(
            str
        ).unique()
    else:
        ultra_bins_all = np.array([], dtype=str)
        normal_bins_all = grid_df["bin"].astype(str).unique()

    # 将 fp_hotspots 整理成 year-month 维度的字典，便于快速查找
    fp_bins_by_ym = {}
    if not fp_hotspots.empty:
        for (yy, mm), g in fp_hotspots.groupby(["year", "month"]):
            fp_bins_by_ym[(int(yy), int(mm))] = g["bin"].astype(str).unique()

    for (y, b, m), g_pos in pos_by:
        b = str(b)
        q_h, q_b = quota_map.get((b, int(m)), (0, 1))
        if q_h == 0 and q_b == 0:
            continue

        fire_dates = set(g_pos["date_dt"].dt.date.tolist())
        month_dates = [d for d in dates if (d.year == y and d.month == m)]
        if not month_dates:
            continue

        # ---- 1. Hard Negatives：同 bin，|Δt|<=3 天但当天没烧 ----
        if q_h > 0:
            hard_candidates = []
            for d in month_dates:
                if d in fire_dates:
                    continue
                if any(abs((d - fd).days) <= 3 for fd in fire_dates):
                    # 若提供了可观测掩码，必须在可观测日采样
                    if b in obs_map:
                        if d.strftime("%Y-%m-%d") in obs_map[b]:
                            hard_candidates.append(d)
                    else:
                        hard_candidates.append(d)

            if hard_candidates:
                count = min(len(hard_candidates), q_h)
                picked_dates = rng.choice(
                    hard_candidates, size=count, replace=False
                )

                bin_info = grid_df[grid_df["bin"] == b]
                if not bin_info.empty:
                    lat_c, lon_c = (bin_info["lat"].iloc[0],
                                    bin_info["lon"].iloc[0])
                    for d in picked_dates:
                        neg_rows.append(
                            {
                                "sample_id": f"neg_h_{y}_{b}_{m}_{d}",
                                "date_key": d.strftime("%Y-%m-%d"),
                                "lon": lon_c,
                                "lat": lat_c,
                                "is_hard": True,
                                "neg_type": 0,  # hard negative
                                "hardness_score": 0.8
                            }
                        )

        # ---- 2. BG Negatives：ultra-safe / fp_hotspot / normal 的混合 ----
        if q_b > 0 and month_dates:
            valid_dates = [d for d in month_dates if d not in fire_dates]
            if not valid_dates:
                continue

            # 本 slot 的 FP-hotspot bin 列表（同年同月）
            fp_bins_slot = fp_bins_by_ym.get((int(y), int(m)), np.array([], dtype=str))
            fp_bins_slot = fp_bins_slot[fp_bins_slot != b]

            ultra_bins = ultra_bins_all[ultra_bins_all != b]
            normal_bins = normal_bins_all[normal_bins_all != b]
            # normal_bins 中排除 fp_bins_slot
            if len(fp_bins_slot) > 0:
                normal_bins = np.setdiff1d(normal_bins, fp_bins_slot)

            # 若没有 ever_fire 信息，则所有都当 normal
            if not has_ever_fire_col:
                ultra_bins = np.array([], dtype=str)
                normal_bins = grid_df[grid_df["bin"] != b]["bin"].astype(
                    str
                ).to_numpy()

            # 目标个数
            q_fp = int(round(q_b * bg_fp_ratio))
            q_ultra = int(round(q_b * bg_ultra_ratio))
            q_fp = min(max(q_fp, 0), q_b)
            q_ultra = min(max(q_ultra, 0), max(q_b - q_fp, 0))
            q_normal = max(q_b - q_fp - q_ultra, 0)

            # 若某类 bin 不足，则将目标数回流到 normal
            if len(fp_bins_slot) == 0 or q_fp == 0:
                q_normal += q_fp
                q_fp = 0

            if len(ultra_bins) == 0 or q_ultra == 0:
                q_normal += q_ultra
                q_ultra = 0

            picked_bins_fp = []
            picked_bins_ultra = []
            picked_bins_normal = []

            if q_fp > 0 and len(fp_bins_slot) > 0:
                picked_bins_fp = rng.choice(
                    fp_bins_slot, size=q_fp, replace=True
                ).tolist()

            if q_ultra > 0 and len(ultra_bins) > 0:
                picked_bins_ultra = rng.choice(
                    ultra_bins, size=q_ultra, replace=True
                ).tolist()

            if q_normal > 0:
                if len(normal_bins) == 0 and len(ultra_bins) > 0:
                    picked_bins_normal = rng.choice(
                        ultra_bins, size=q_normal, replace=True
                    ).tolist()
                elif len(normal_bins) > 0:
                    picked_bins_normal = rng.choice(
                        normal_bins, size=q_normal, replace=True
                    ).tolist()

            # 生成 fp_hotspot BG 样本
            for ob in picked_bins_fp:
                # 仅采样可观测日期
                valid_obs_dates = valid_dates
                if ob in obs_map:
                    valid_obs_dates = [dd for dd in valid_dates if dd.strftime("%Y-%m-%d") in obs_map[ob]]
                if not valid_obs_dates:
                    continue
                d = rng.choice(valid_obs_dates)
                bg_info = grid_df[grid_df["bin"] == ob]
                if bg_info.empty:
                    continue
                neg_rows.append(
                    {
                        "sample_id": f"neg_fp_{y}_{ob}_{m}_{d}",
                        "date_key": d.strftime("%Y-%m-%d"),
                        "lon": bg_info["lon"].iloc[0],
                        "lat": bg_info["lat"].iloc[0],
                        "is_hard": False,
                        "neg_type": 2,  # fp_hotspot BG
                        "hardness_score": 0.6
                    }
                )

            # 生成 ultra-safe BG 样本
            for ob in picked_bins_ultra:
                valid_obs_dates = valid_dates
                if ob in obs_map:
                    valid_obs_dates = [dd for dd in valid_dates if dd.strftime("%Y-%m-%d") in obs_map[ob]]
                if not valid_obs_dates:
                    continue
                d = rng.choice(valid_obs_dates)
                bg_info = grid_df[grid_df["bin"] == ob]
                if bg_info.empty:
                    continue
                neg_rows.append(
                    {
                        "sample_id": f"neg_ub_{y}_{ob}_{m}_{d}",
                        "date_key": d.strftime("%Y-%m-%d"),
                        "lon": bg_info["lon"].iloc[0],
                        "lat": bg_info["lat"].iloc[0],
                        "is_hard": False,
                        "neg_type": 1,  # ultra-safe BG
                        "hardness_score": 0.1
                    }
                )

            # 生成普通 BG 样本
            for ob in picked_bins_normal:
                valid_obs_dates = valid_dates
                if ob in obs_map:
                    valid_obs_dates = [dd for dd in valid_dates if dd.strftime("%Y-%m-%d") in obs_map[ob]]
                if not valid_obs_dates:
                    continue
                d = rng.choice(valid_obs_dates)
                bg_info = grid_df[grid_df["bin"] == ob]
                if bg_info.empty:
                    continue
                neg_rows.append(
                    {
                        "sample_id": f"neg_b_{y}_{ob}_{m}_{d}",
                        "date_key": d.strftime("%Y-%m-%d"),
                        "lon": bg_info["lon"].iloc[0],
                        "lat": bg_info["lat"].iloc[0],
                        "is_hard": False,
                        "neg_type": 0,  # normal BG
                        "hardness_score": 0.3
                    }
                )

    neg_df = pd.DataFrame(neg_rows)
    n_total = len(neg_df)
    if n_total == 0:
        print("[Sampling] 注意：本轮未采样到任何负样本！")
        return neg_df

    n_hard = int((neg_df["is_hard"] == True).sum())
    n_bg = int((neg_df["is_hard"] == False).sum())
    n_ultra = int(((neg_df.get("neg_type", 0) == 1) &
                   (neg_df["is_hard"] == False)).sum())
    n_fp = int(((neg_df.get("neg_type", 0) == 2) &
                (neg_df["is_hard"] == False)).sum())
    print(
        f"[Sampling] 完成。生成负样本: {n_total:,} "
        f"(Hard={n_hard:,}, BG={n_bg:,}, "
        f"Ultra-safe BG={n_ultra:,}, FP-hotspot BG={n_fp:,})"
    )
    return neg_df


# ----------------- 主入口 -----------------


def gen_negative_samples_for_iter(iter_idx: int,
                                  feedback_path: Path,
                                  grid_deg: float = 0.1) -> Path:
    """
    为第 iter_idx 轮生成负样本（设计用于 iter_idx >= 2）。

    使用上一轮 feedback_it{iter_idx-1}.json 作为 Bandit reward 来源，
    并在若干采样策略模板之间进行自适应选择。

    这一步会真正把 fullgrid 环境中的：
        - slot_stats_env（miss_rate / contamination / n_fn 等）
        - fp_hotspots
    映射成：
        - 各 bin×month 槽位的 Hard/BG 配额；
        - 本轮总负样本量 + neg_type 结构。
    """
    if iter_idx <= 1:
        raise ValueError("设计用于 iter_idx >= 2。")

    print(f"\n[neg] ========== 生成第 {iter_idx} 轮负样本 "
          f"(v10.0 Forest Mask + neg_type + Bandit FN-aware) ==========")

    # 1. 加载数据并过滤正样本（仅森林内火点）
    print("[neg] 加载 FIRMS 正样本...")
    firms = _load_firms_master(grid_deg=grid_deg)
    firms = _filter_by_forest_lc(firms, is_grid=False)

    # 2. 构建并过滤背景格网（仅森林格网）
    print("[neg] 构建背景格网...")
    grid_df = _build_grid_df_from_firms(firms, grid_deg=grid_deg)
    grid_df = _filter_by_forest_lc(grid_df, is_grid=True, ref_year=2018)

    if grid_df.empty:
        raise ValueError("严重错误：过滤后没有剩余的背景格网，请检查 LC 数据是否正确。")

    # 标记 ever_fire（用于 ultra-safe BG 判定）
    fire_bins = set(firms["bin"].astype(str).unique())
    grid_df["bin"] = grid_df["bin"].astype(str)
    grid_df["ever_fire"] = grid_df["bin"].isin(fire_bins)

    # 槽位难度信息 + FP/FN-hotspots 信息
    slot_info = _load_feedback_slot_info(feedback_path)
    fp_hotspots = _load_fp_hotspots(feedback_path)
    fn_hotspots = _load_fn_hotspots(feedback_path)  # 当前通常为空，仅预留
    if not fp_hotspots.empty:
        print(f"[neg] 读取 FP-hotspots 槽位数: {len(fp_hotspots):,}")
    if not fn_hotspots.empty:
        print(f"[neg] 读取 FN-hotspots 槽位数: {len(fn_hotspots):,}")
    dates = pd.date_range(DATE_MIN, DATE_MAX, freq="D").date

    # 3. Bandit 决策 Hard / BG 比例 & BG 内部结构 & FN/FP 聚焦模式
    print("[neg] 使用 Bandit 控制器选择采样策略模板 ...")
    prev_fb = feedback_path  # 这里的 feedback_path 是上一轮 feedback_it{iter_idx-1}.json
    (
        chosen_tid,
        hard_ratio,
        bg_fp_ratio,
        bg_ultra_ratio,
        focus_type,
        neg_budget_factor,
    ) = _bandit_choose_template(iter_idx, prev_fb)

    # 根据 focus_type / neg_budget_factor 重新计算槽位配额
    quotas = _distribute_quotas(
        firms,
        slot_info,
        hard_ratio=hard_ratio,
        focus_type=focus_type,
        neg_budget_factor=neg_budget_factor,
    )

    # 4. 执行采样
    rng = np.random.RandomState(RANDOM_STATE + iter_idx)
    neg_samples = _sample_negatives(
        firms,
        grid_df,
        dates,
        quotas,
        fp_hotspots=fp_hotspots,
        hard_ratio=hard_ratio,
        bg_fp_ratio=bg_fp_ratio,
        bg_ultra_ratio=bg_ultra_ratio,
        rng=rng,
    )

    # 5. 输出
    out_path = NEG_MASTER_FMT.with_name(NEG_MASTER_FMT.name.format(
        iter=iter_idx
    ))
    neg_samples.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"[neg] 采用模板 {chosen_tid} (focus_type={focus_type}) 输出文件：{out_path}")
    print(f"[neg] ========== 完成 ==========\n")
    return out_path


if __name__ == "__main__":
    # 简单测试：假设已有 feedback_it1.json，则生成第 2 轮负样本
    fb = FEEDBACK_FMT.with_name(FEEDBACK_FMT.name.format(iter=1))
    if fb.exists():
        gen_negative_samples_for_iter(iter_idx=2,
                                      feedback_path=fb,
                                      grid_deg=0.1)
    else:
        print(f"找不到 {fb}，无法测试。")
