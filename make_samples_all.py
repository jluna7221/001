# -*- coding: utf-8 -*-
"""
make_samples_all.py (v6.0 - 森林掩膜版 + neg_type 支持)
"""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

import build_static_features as static_mod  # 引入特征工具进行过滤

BASE = Path(r"E:\fire")
OUT_DIR = BASE / "outputs"
FIRMS_MASTER = OUT_DIR / "firms_master.parquet"
NEG_MASTER_FMT = OUT_DIR / "neg_master_it{iter}.parquet"
NEG_MASTER_INIT = OUT_DIR / "neg_master.parquet"
SAMPLES_ALL = OUT_DIR / "samples_all.parquet"

DATE_MIN = dt.date(2001, 1, 1)
DATE_MAX = dt.date(2020, 12, 31)


def _filter_by_forest(df):
    """仅保留 LC 1-10 的点"""
    print(f"    [Filter] 过滤前: {len(df):,}")
    static_mod._init_lc_year2path()
    mapping = static_mod._LC_YEAR2PATH
    if not mapping:
        return df

    # 提取年份
    if "year" not in df.columns:
        ys = pd.to_datetime(df["date_key"]).dt.year.fillna(2018).astype(int).values
    else:
        ys = df["year"].values

    lons, lats = df["lon"].values, df["lat"].values
    lc_vals = static_mod.sample_yearly_tif_points(ys, lons, lats, mapping)
    lc_vals = np.nan_to_num(lc_vals, nan=-9999)

    mask = (lc_vals >= 1) & (lc_vals <= 10)
    filtered = df[mask].copy()
    print(f"    [Filter] 过滤后(LC 1-10): {len(filtered):,} (剔除 {len(df) - len(filtered):,})")
    return filtered


def make_samples_all_for_iter(iter_idx: int, pos_to_neg_ratio: float = None) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载并过滤正样本
    print("[Sample] 加载 FIRMS 正样本...")
    pos = pd.read_parquet(FIRMS_MASTER)
    pos["label"] = 1
    pos["is_hard"] = False
    # 正样本没有 neg_type，用 -1 占位
    pos["neg_type"] = -1
    pos["date_key"] = pd.to_datetime(pos["date_key"]).dt.strftime("%Y-%m-%d")
    if "is_observed" not in pos.columns:
        pos["is_observed"] = 1
    pos = _filter_by_forest(pos)  # 关键过滤

    # 2. 加载并过滤负样本
    print(f"[Sample] 加载 Iter {iter_idx} 负样本...")
    p_neg = NEG_MASTER_FMT.with_name(NEG_MASTER_FMT.name.format(iter=iter_idx))
    if not p_neg.exists():
        if iter_idx == 1 and NEG_MASTER_INIT.exists():
            p_neg = NEG_MASTER_INIT
        else:
            raise FileNotFoundError(f"Missing negatives: {p_neg}")

    neg = pd.read_parquet(p_neg)
    neg["label"] = 0
    if "is_hard" not in neg.columns:
        neg["is_hard"] = False

    # [New] 构造 / 标准化 neg_type
    # 约定：
    #   - hard 或普通负样本：neg_type = 0
    #   - ultra-safe 背景：neg_type = 1
    if "neg_type" not in neg.columns:
        neg["is_hard"] = neg["is_hard"].fillna(False)
        neg["neg_type"] = np.where(neg["is_hard"].astype(bool), 0, 1).astype(np.int8)
        print(
            "[Sample] 负样本中未发现 neg_type 列，按照 is_hard=True->0, False->1 进行默认赋值。"
        )
    else:
        neg["neg_type"] = neg["neg_type"].fillna(1).astype(np.int8)

    neg["date_key"] = pd.to_datetime(neg["date_key"]).dt.strftime("%Y-%m-%d")
    if "is_observed" not in neg.columns:
        neg["is_observed"] = 1
    neg = _filter_by_forest(neg)  # 关键过滤

    # 3. 采样平衡 (Hard 优先)
    target_ratio = pos_to_neg_ratio if pos_to_neg_ratio else 10.0
    n_pos = len(pos)
    n_target = int(n_pos * target_ratio)

    hard = neg[neg["is_hard"] == True]
    easy = neg[neg["is_hard"] == False]

    print(f"[Sample] 平衡策略: Hard({len(hard):,}) + Easy({len(easy):,}) -> Target({n_target:,})")

    if len(hard) >= n_target:
        neg_sel = hard
    else:
        n_need = n_target - len(hard)
        easy_sel = easy.sample(n=min(n_need, len(easy)), random_state=2025)
        neg_sel = pd.concat([hard, easy_sel], ignore_index=True)

    # 4. 合并输出
    all_df = pd.concat([pos, neg_sel], ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["sample_id", "date_key"])
    all_df.to_parquet(SAMPLES_ALL, index=False)
    print(f"[OK] 样本构建完成: {len(all_df):,} (1:{(len(all_df) - n_pos) / n_pos:.1f}) -> {SAMPLES_ALL}")
    return SAMPLES_ALL


if __name__ == "__main__":
    make_samples_all_for_iter(1)
1
