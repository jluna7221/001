# -*- coding: utf-8 -*-
"""
xgb_train_optuna.py  (v11.0 - Base logloss + TopK Ranking Head + FN-hotspots + Rich RL Signals)

核心结构：

1）底层：F/H 双子 logloss 模型
    - 负责学习“哪里像火 / 哪里绝对安全”，保证全局 AUC / AP / 概率刻度；
    - 继续使用 neg_type / is_hard / FN 加权逻辑；
    - 额外引入 fullgrid FN-hotspots 驱动的火点加权（真正“业务上漏报”的地方）。

2）中层：TopK 头部重排器（Ranking Head）
    - 在 fullgrid env（默认 2017-2018）上：
        * 用 F/H 组合得到 base 分数 p_base；
        * 按 year×month 分组，每个月取 base 分数 Top20% 作为候选集；
        * 在候选集上训练 XGBoost 排序模型：
              objective='rank:ndcg', eval_metric='ndcg@10'
          group = year*100 + month；
        * 对 fullgrid 全体格点跑一遍 rank-head 得到 score_rank；
        * 再对 score_rank 训练 1D Logistic 校准器：
              p_final = sigmoid(a * score_rank + b)
          作为“最终系统输出概率”。

    - fullgrid 四件套（N_total/N_valid、火点概率四分区、TopK Recall、分位曲线），
      全部基于 p_final 计算。
    - RL 奖励里的 R@Top1/5/10、slot difficulty、FP-hotspot / FN-hotspot 也都基于这套打分。

3）上层：为 Bandit / RL 提供更丰富的反馈指标
    - 在 fullgrid 把“正负样本的概率刻度”拆成若干量化指标：
        * µ_pos：真实火点的平均预测值
        * µ_neg：背景格点的平均预测值
        * Q_pos_0.5 / Q_pos_0.8：真实火点预测值中位数 / 80% 分位
        * cov_pos_ge_high(θ)：真实火点中 p≥θ 的比例（θ = 0.75）
        * neg_high_rate(θ)：负例中 p≥θ 的比例
    - 再额外输出：
        * fn_hotspots：fullgrid 上“被打得太低的真实火点聚集区”（FN-hotspots）
        * bucket_stats：按 (season × lc_group × elev_band) 粗粒度分桶的刻度统计
    - 全部写入 fullgrid_eval_it{iter}.json / feedback_it{iter}.json / iter_metrics.csv，
      供 gen_negative_samples.py 里的 Bandit 组合成 Reward + 聚焦策略。
"""

from pathlib import Path
import json
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import matplotlib.pyplot as plt

try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None

from build_static_features import build_feature_matrix, attach_bin_month_year
from eval_fullgrid_env import get_eval_env_path, ensure_eval_fullgrid_env
from gen_negative_samples import _load_firms_master  # 用于统计 N_total（全年 FIRMS 火点数）

# ----------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------

BASE_OUT = Path(r"G:\fire\outputs")

USE_OPTUNA = True
N_TRIALS = 50
RANDOM_STATE = 2025

USE_RANK_OBJECTIVE = False

# 默认参数（logloss / rank 都在此基础上微调）
DEFAULT_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "gamma": 0.1,
    "min_child_weight": 10,
    "max_delta_step": 1.0,
}

# 后续轮次的增量训练轮数
ADDITIONAL_ROUNDS = 500

# fullgrid Topα 指标
TOPK_ALPHAS = [0.01, 0.05, 0.10]  # 1%, 5%, 10%
RECALL_CURVE_ALPHAS = np.linspace(0.01, 1.0, 100)  # 覆盖曲线 α 网格

# slot 难度计算权重（围绕 Top5% 定义）
DIFF_ALPHA = 0.7  # miss_rate 的权重
DIFF_BETA = 0.3   # contamination 的权重

# 综合得分 S_it 的权重（fullgrid Topα 为第一公民）
# S_it = w1 * R@1% + w2 * R@5% + w3 * R@10% + λ * AP_RLVAL
SCORE_W_R1 = 0.5   # w1，对 R@1% 的权重
SCORE_W_R5 = 0.3   # w2，对 R@5% 的权重（核心指标）
SCORE_W_R10 = 0.1  # w3，对 R@10% 的权重
SCORE_W_AP = 0.1   # λ，对 RL-val AP 的权重

# FN 假阴性挖矿相关：权重放大系数
HARD_POS_PERCENTILE = 0.8        # 分位 <80% 视为“难救火点”（基于训练样本上的旧模型预测）
HARD_POS_WEIGHT_FACTOR = 2.0     # 对这类火点，sample_weight 乘以该系数

# fullgrid 级 FN-hotspots 判定阈值 & 权重
FN_HOTSPOT_THR = 0.50            # 在 fullgrid 上，p_final < 0.5 的火点视为“漏报区域的候选”
FN_HOTSPOT_WEIGHT_FACTOR = 2.0   # 这些 FN-hotspot 所在 bin×month 的火点，额外升权

# Ranking Head 超参
RANK_CAND_ALPHA = 0.20       # 每个 year-month 内，base Top20% 作为候选集
RANK_MAX_SAMPLES = 300_000   # 排序头最大训练样本数（超过则下采样）

# 定义“高风险概率阈值”——用于 RL 的正/负刻度统计
HIGH_PROB_THR = 0.75

# --------------------- 防泄漏工具 ---------------------

_LEAK_REGEX = (
    "label",
    "slot",
    "pred",
    "topk",
    "n_pos",
    "n_neg",
    "difficulty",
    "is_in_topk",
)


def _drop_leak_by_name(cols: List[str]) -> List[str]:
    """根据列名关键词粗暴剔除明显泄漏列。"""
    keep = []
    for c in cols:
        name = c.lower()
        if any(k in name for k in _LEAK_REGEX):
            continue
        keep.append(c)
    return keep


def drop_leaky_features_auc(
    X: pd.DataFrame, y: np.ndarray, threshold: float = 0.999
) -> List[str]:
    """
    单列 AUC 过滤：对每个特征单独训练一个浅层 XGB，
    若单列 AUC >= threshold，则认为存在强泄漏，直接剔除。
    """
    leaky = []
    for col in X.columns:
        try:
            dtrain = xgb.DMatrix(X[[col]].to_numpy(), label=y)
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": 1,
                "eta": 0.5,
            }
            bst = xgb.train(params, dtrain, num_boost_round=20, verbose_eval=False)
            p = bst.predict(dtrain)
            auc = roc_auc_score(y, p)
        except Exception:
            continue

        if auc >= threshold:
            leaky.append(col)
            print(f"[leak] {col} 单列 AUC={auc:.5f} ≥ {threshold} -> 剔除")
    return leaky


# --------------------- 数据切分 ---------------------


def split_train_val_test(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按年份切分：<=2016 -> train，2017~2018 -> RL-val，>=2019 -> final test
    若样本量太小则回退到 8:1:1 随机切分（按时间顺序）。
    """
    d = df.copy()
    if "year" not in d.columns:
        d["date_key"] = pd.to_datetime(d["date_key"])
        d["year"] = d["date_key"].dt.year.astype(int)

    tr = d[d["year"] <= 2016].copy()
    rl_val = d[(d["year"] >= 2017) & (d["year"] <= 2018)].copy()
    final_test = d[d["year"] >= 2019].copy()

    if len(tr) >= 1000 and len(rl_val) >= 1000 and len(final_test) >= 1000:
        print(
            f"[split] train={len(tr):,}, rl_val={len(rl_val):,}, "
            f"final_test={len(final_test):,}"
        )
        return tr, rl_val, final_test

    # 回退方案：样本较少时使用 8:1:1 按时间顺序切分
    d = d.sort_values("year").reset_index(drop=True)
    n = len(d)
    tr = d.iloc[: int(n * 0.8)].copy()
    rl_val = d.iloc[int(n * 0.8): int(n * 0.9)].copy()
    final_test = d.iloc[int(n * 0.9):].copy()
    print("[split] 使用 8:1:1 切分（样本或年份不足）")
    return tr, rl_val, final_test


# --------------------- 训练辅助 ---------------------


def _train_xgb(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    group_tr: Optional[np.ndarray],
    group_va: Optional[np.ndarray],
    base_params: dict,
    use_optuna: bool,
):
    """
    针对一个特征子集（flamm / human）训练单支 XGB 模型，
    可选使用 Optuna 进行超参搜索。

    若 USE_RANK_OBJECTIVE=True，则使用 rank:ndcg 目标 + year×month 分组；
    否则使用 binary:logistic + AUC 作为 eval_metric。
    """
    # ---- 构建 DMatrix ----
    if USE_RANK_OBJECTIVE and (group_tr is not None) and (group_va is not None):
        # 排序目标：需按 group_key 排序并设置 group
        def _make_rank_dmatrix(X, y, w, gkey):
            order = np.argsort(gkey)
            Xs = X.iloc[order].to_numpy()
            ys = y[order]
            ws = w[order]
            gkey_sorted = gkey[order]
            _, counts = np.unique(gkey_sorted, return_counts=True)
            dmat = xgb.DMatrix(Xs, label=ys, weight=ws)
            dmat.set_group(counts.tolist())
            return dmat

        dtrain = _make_rank_dmatrix(X_tr, y_tr, w_tr, group_tr)
        dval = _make_rank_dmatrix(X_va, y_va,
                                  np.ones_like(y_va, dtype=np.float32),
                                  group_va)
    else:
        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        dval = xgb.DMatrix(X_va, label=y_va)

    # ---- 设置参数 ----
    params = base_params.copy()
    pos_rate = float(np.mean(y_tr))
    pos_rate = min(max(pos_rate, 1e-6), 1.0 - 1e-6)
    params.setdefault("scale_pos_weight", (1.0 - pos_rate) / pos_rate)
    params.setdefault("base_score", pos_rate)

    if USE_RANK_OBJECTIVE:
        params["objective"] = "rank:ndcg"
        params["eval_metric"] = "ndcg@10"
    else:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "auc"

    # ---- Optuna （可选）----
    if use_optuna and (optuna is not None):
        print("[optuna] tune ...")

        def objective(trial):
            tp = params.copy()
            tp["max_depth"] = trial.suggest_int("max_depth", 3, 7)
            tp["learning_rate"] = trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            )
            tp["min_child_weight"] = trial.suggest_int("min_child_weight", 5, 100)
            tp["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
            tp["colsample_bytree"] = trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            )
            tp["reg_lambda"] = trial.suggest_float("reg_lambda", 0.1, 10.0)
            tp["reg_alpha"] = trial.suggest_float("reg_alpha", 0.0, 5.0)

            bst = xgb.train(
                tp,
                dtrain,
                num_boost_round=1000,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            pv = bst.predict(dval)
            # 即使用 rank 目标，外层我们仍用 AUC 来评估好坏
            return roc_auc_score(y_va, pv)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=N_TRIALS)
        params.update(study.best_params)
        print("[optuna] best:", study.best_params)

    # ---- 最终训练 ----
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    pred_va = bst.predict(dval)
    auc_val = roc_auc_score(y_va, pred_va)
    ap_val = average_precision_score(y_va, pred_va)
    return bst, auc_val, ap_val, params


def _groups_from_columns(cols: List[str]) -> Tuple[List[str], List[str]]:
    """
    按名称规则把特征划分为 flamm / human 两组。
    """
    flamm_keys = (
        "Mate",
        "Mite",
        "Ate",
        "Ate_3d_mean",
        "Arh",
        "Arh_3d_mean",
        "Aws",
        "Aws_3d_mean",
        "Suh",
        "Pre",
        "Pre_3d_sum",
        "Pre_7d_sum",
        "Pre_30d_sum",
        "Raindate",
        "ndvi",
        "dem",
        "slope",
        "aspect",
        "aspect_sin",
        "aspect_cos",
        "lc_raw",
    )
    human_keys = (
        "pop_density",
        "gdp_density",
        "dist_road",
        "dist_village",
        "dist_river",
        "wd_",
        "is_weekend",
        "is_holiday",
        "is_fire_season",
        "is_summer_vacation",
        "is_tourism_peak",
        "month",
        "dayofyear",
    )
    f_cols = []
    h_cols = []
    for c in cols:
        if any(c.startswith(k) for k in flamm_keys):
            f_cols.append(c)
        elif any(c.startswith(k) for k in human_keys):
            h_cols.append(c)

    if not f_cols:
        f_cols = [c for c in cols if c not in h_cols][:5]
    if not h_cols:
        h_cols = [c for c in cols if c not in f_cols][:5]
    return f_cols, h_cols


def _pick_w_by_auc(
    y: np.ndarray, pF: np.ndarray, pH: np.ndarray
) -> Tuple[float, float, float]:
    """
    在 [0,1] 上网格搜索 w，选择 AP 优先、AUC 次要 的最优组合权重。
    """
    grid = np.linspace(0.0, 1.0, 21)
    best = (0.5, -1.0, -1.0)  # (w, auc, ap)
    for w in grid:
        pf = w * pF + (1.0 - w) * pH
        try:
            auc = roc_auc_score(y, pf)
            ap = average_precision_score(y, pf)
        except Exception:
            continue
        if (ap > best[2]) or (np.isclose(ap, best[2]) and auc > best[1]):
            best = (float(w), float(auc), float(ap))
    return best


# --------------------- fullgrid 评估 & 概率校准 ---------------------


def _fit_logistic_calibrator(
    pred: np.ndarray, label: np.ndarray, max_samples: int = 200000, train_mask: Optional[np.ndarray] = None
) -> Optional[Dict]:
    """
    在 fullgrid 上对最终得分进行一维 Logistic 校准：
        s_raw -> p_cal = sigmoid(a * s_raw + b)
    """
    pred = np.asarray(pred, dtype=np.float32)
    label = np.asarray(label, dtype=int)
    if train_mask is not None:
        train_mask = np.asarray(train_mask, dtype=bool)
        if train_mask.shape[0] == pred.shape[0]:
            pred = pred[train_mask]
            label = label[train_mask]

    mask = np.isfinite(pred) & np.isin(label, [0, 1])
    pred = pred[mask]
    label = label[mask]

    n_pos = int((label == 1).sum())
    n_neg = int((label == 0).sum())
    if n_pos < 100 or n_neg < 100:
        print(f"[calib] 正负样本不足（pos={n_pos}, neg={n_neg}），跳过校准器训练。")
        return None

    if len(pred) > max_samples:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(pred), size=max_samples, replace=False)
        pred = pred[idx]
        label = label[idx]
        print(f"[calib] 下采样到 {max_samples} 个样本进行校准。")

    try:
        lr = LogisticRegression(solver="lbfgs", max_iter=1000, n_jobs=-1)
        lr.fit(pred.reshape(-1, 1), label)
    except Exception as e:  # pragma: no cover
        print(f"[calib] Logistic 回归拟合失败：{e}，跳过校准器。")
        return None

    coef = float(lr.coef_.ravel()[0])
    intercept = float(lr.intercept_[0])

    p_raw = pred
    p_cal = 1.0 / (1.0 + np.exp(-(coef * p_raw + intercept)))
    try:
        brier_raw = brier_score_loss(label, p_raw)
        brier_cal = brier_score_loss(label, p_cal)
        print(f"[calib] Brier(raw)={brier_raw:.5f} -> Brier(calib)={brier_cal:.5f}")
    except Exception:
        pass

    return {
        "type": "logistic_1d",
        "coef": coef,
        "intercept": intercept,
        "input": "rank_head",
        "n_train": int(len(label))
    }

def _compute_calibration_metrics(pred: np.ndarray, label: np.ndarray, n_bins: int = 20) -> Dict:
    pred = np.asarray(pred, dtype=np.float32)
    label = np.asarray(label, dtype=int)
    mask = np.isfinite(pred) & np.isin(label, [0, 1])
    pred = pred[mask]
    label = label[mask]
    if len(pred) == 0:
        return {"ece": float("nan"), "bins": []}
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (pred >= lo) & (pred < hi) if i < n_bins - 1 else (pred >= lo) & (pred <= hi)
        if not np.any(m):
            rows.append({"bin": [float(lo), float(hi)], "n": 0, "p_mean": 0.0, "pos_frac": 0.0})
            continue
        p_mean = float(np.mean(pred[m]))
        pos_frac = float(np.mean(label[m]))
        ece += (np.sum(m) / len(pred)) * abs(p_mean - pos_frac)
        rows.append({"bin": [float(lo), float(hi)], "n": int(np.sum(m)), "p_mean": p_mean, "pos_frac": pos_frac})
    return {"ece": float(ece), "bins": rows}


def _compute_fullgrid_report(
    df_env: pd.DataFrame, pred: np.ndarray, grid_deg: float = 0.1
) -> Dict:
    """
    统一 fullgrid 评估报告：
        - years: 评估年份列表
        - N_total_fires: 评估年份内 FIRMS 火点总数
        - N_valid_fires: fullgrid env 中 label=1 的有效火点数
        - fire_prob_bins: 火点概率四区间统计 + 高风险覆盖比例
        - pos_neg_stats: 正/负样本的刻度统计（µ_pos / µ_neg 等）
        - topk_global: fullgrid Recall@Topα（α=1%,5%,10%）+ lift
        - recall_curve: α∈[1%,100%] 的 Recall@Topα 曲线
    """
    df = df_env.copy()
    df["pred"] = pred.astype(float)

    years = sorted(int(y) for y in df["year"].unique())
    year_min, year_max = years[0], years[-1]

    firms_all = _load_firms_master(grid_deg=grid_deg)
    firms_all["year"] = pd.to_datetime(
        firms_all["date_key"], errors="coerce"
    ).dt.year.astype(int)
    mask_year = (firms_all["year"] >= year_min) & (firms_all["year"] <= year_max)
    n_total = int(mask_year.sum())

    labels = df["label"].to_numpy().astype(int)
    n_valid = int((labels == 1).sum())

    print(
        f"[fullgrid] 评估年份 {year_min}-{year_max}："
        f"N_total(FIRMS)={n_total:,}, N_valid(in env)={n_valid:,}"
    )

    # 火点概率四区间
    bins = [0.0, 0.25, 0.5, 0.75, 1.0000001]
    bin_labels = ["[0,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,1.0]"]

    fire_df = df[labels == 1].copy()
    fire_probs = fire_df["pred"].to_numpy(dtype=np.float32)

    # 正负样本刻度统计（给 RL 用）
    pos_neg_stats = {
        "mu_pos": 0.0,
        "mu_neg": 0.0,
        "q_pos_0.5": 0.0,
        "q_pos_0.8": 0.0,
        "cov_pos_ge_high": 0.0,
        "neg_high_rate": 0.0,
        "high_thr": float(HIGH_PROB_THR),
    }

    if len(fire_probs) > 0:
        counts, _ = np.histogram(fire_probs, bins=bins)
        counts = counts.astype(int)
        percents = (counts / counts.sum() * 100.0).round(2).tolist()

        mu_pos = float(fire_probs.mean())
        q_pos_0_5 = float(np.quantile(fire_probs, 0.5))
        q_pos_0_8 = float(np.quantile(fire_probs, 0.8))
        cov_pos = float((fire_probs >= HIGH_PROB_THR).mean())

        pos_neg_stats["mu_pos"] = mu_pos
        pos_neg_stats["q_pos_0.5"] = q_pos_0_5
        pos_neg_stats["q_pos_0.8"] = q_pos_0_8
        pos_neg_stats["cov_pos_ge_high"] = cov_pos
    else:
        counts = np.zeros(len(bins) - 1, dtype=int)
        percents = [0.0] * (len(bins) - 1)

    if counts.sum() > 0:
        high_bin_ratio = float(counts[-1] / counts.sum())
    else:
        high_bin_ratio = 0.0

    fire_prob_bins = {
        "bin_edges": bins,
        "bin_labels": bin_labels,
        "counts": counts.tolist(),
        "percentages": percents,
        "high_bin_ratio": high_bin_ratio,
    }

    # 负例刻度
    neg_df = df[labels == 0].copy()
    neg_probs = neg_df["pred"].to_numpy(dtype=np.float32)
    if len(neg_probs) > 0:
        mu_neg = float(neg_probs.mean())
        neg_high_rate = float((neg_probs >= HIGH_PROB_THR).mean())
        pos_neg_stats["mu_neg"] = mu_neg
        pos_neg_stats["neg_high_rate"] = neg_high_rate

    preds_all = df["pred"].to_numpy(dtype=np.float32)
    labels_all = labels
    n_cells = len(preds_all)
    pos_total = int((labels_all == 1).sum())

    if n_cells == 0 or pos_total == 0:
        topk = {
            "alphas": TOPK_ALPHAS,
            "K": [0] * len(TOPK_ALPHAS),
            "recall": [0.0] * len(TOPK_ALPHAS),
            "lift": [0.0] * len(TOPK_ALPHAS),
        }
        recall_curve = {
            "alphas": RECALL_CURVE_ALPHAS.tolist(),
            "recall": [0.0] * len(RECALL_CURVE_ALPHAS),
            "pos_total": pos_total,
            "n_cells": n_cells,
        }
        return {
            "years": years,
            "grid_deg": grid_deg,
            "N_total_fires": n_total,
            "N_valid_fires": n_valid,
            "fire_prob_bins": fire_prob_bins,
            "pos_neg_stats": pos_neg_stats,
            "topk_global": topk,
            "recall_curve": recall_curve,
        }

    order = np.argsort(preds_all)[::-1]
    sorted_labels = labels_all[order]
    cum_pos = np.cumsum(sorted_labels)
    pos_total = int(cum_pos[-1])

    # Topα（1%，5%，10%）
    topK_list = []
    recall_list = []
    lift_list = []
    for alpha in TOPK_ALPHAS:
        K = max(1, int(round(alpha * n_cells)))
        if K > n_cells:
            K = n_cells
        tp_top = int(cum_pos[K - 1])
        recall = float(tp_top / pos_total) if pos_total > 0 else 0.0
        lift = float(recall / alpha) if alpha > 0 else 0.0
        topK_list.append(K)
        recall_list.append(recall)
        lift_list.append(lift)

    topk = {
        "alphas": TOPK_ALPHAS,
        "K": topK_list,
        "recall": recall_list,
        "lift": lift_list,
    }

    # 覆盖曲线
    curve_recall = []
    for alpha in RECALL_CURVE_ALPHAS:
        K = max(1, int(round(alpha * n_cells)))
        if K > n_cells:
            K = n_cells
        tp_top = int(cum_pos[K - 1])
        r = float(tp_top / pos_total) if pos_total > 0 else 0.0
        curve_recall.append(r)

    recall_curve = {
        "alphas": RECALL_CURVE_ALPHAS.tolist(),
        "recall": curve_recall,
        "pos_total": pos_total,
        "n_cells": n_cells,
    }

    return {
        "years": years,
        "grid_deg": grid_deg,
        "N_total_fires": n_total,
        "N_valid_fires": n_valid,
        "fire_prob_bins": fire_prob_bins,
        "pos_neg_stats": pos_neg_stats,
        "topk_global": topk,
        "recall_curve": recall_curve,
    }


def _build_slot_env_top5(df_pred: pd.DataFrame, thr_top5: float) -> pd.DataFrame:
    """
    使用全局 Top5% 阈值 thr_top5 计算每个 slot = (year, bin, month) 的难度：
        - miss_rate_s     = 该 slot 火点未进入 Top5% 的比例；
        - contamination_s = Top5% 内高风险格点中假阳性的比例；
        - difficulty_s    = DIFF_ALPHA * miss_rate_s + DIFF_BETA * contamination_s。
    """
    d2 = df_pred.copy()
    d2["is_top5"] = (d2["pred"] >= thr_top5).astype(int)

    rows = []
    for (y, b, m), g in d2.groupby(["year", "bin", "month"]):
        labels = g["label"].to_numpy()
        is_top5 = g["is_top5"].to_numpy()

        n_pos = int((labels == 1).sum())
        n_tp_top5 = int(((labels == 1) & (is_top5 == 1)).sum())
        n_fp_top5 = int(((labels == 0) & (is_top5 == 1)).sum())
        n_fn = n_pos - n_tp_top5
        n_top5 = n_tp_top5 + n_fp_top5

        if n_pos > 0:
            miss_rate = n_fn / float(n_pos)
        else:
            miss_rate = 0.0

        if n_top5 > 0:
            contamination = n_fp_top5 / float(n_top5)
        else:
            contamination = 0.0

        difficulty = DIFF_ALPHA * miss_rate + DIFF_BETA * contamination

        rows.append(
            {
                "year": int(y),
                "bin": str(b),
                "month": int(m),
                "n_pos": int(n_pos),
                "n_tp_top5": int(n_tp_top5),
                "n_fp_top5": int(n_fp_top5),
                "n_fn": int(n_fn),
                "n_top5": int(n_top5),
                "miss_rate": float(miss_rate),
                "contamination": float(contamination),
                "difficulty": float(difficulty),
            }
        )

    if not rows:
        return pd.DataFrame(columns=[
            "year",
            "bin",
            "month",
            "n_pos",
            "n_tp_top5",
            "n_fp_top5",
            "n_fn",
            "n_top5",
            "miss_rate",
            "contamination",
            "difficulty",
        ])

    return pd.DataFrame(rows)


def _extract_fp_hotspots(df_pred: pd.DataFrame, thr_top5: float) -> pd.DataFrame:
    """
    从 fullgrid 预测结果中提取 FP-hotspot：
        - pred >= thr_top5
        - label == 0
        - 所在 bin 在评估年份内“整年无火”（ever_fire_env=False）
    输出列：year, month, bin, n_fp_top5
    """
    d = df_pred.copy()
    d["is_top5"] = (d["pred"] >= thr_top5).astype(int)

    # bin 是否在评估年份内发生过火
    bin_fire = d.groupby("bin")["label"].max().rename("ever_fire_env")
    d = d.merge(bin_fire, on="bin", how="left")

    mask_fp = (d["is_top5"] == 1) & (d["label"] == 0) & (d["ever_fire_env"] == 0)
    fp = d[mask_fp].copy()
    if fp.empty:
        return pd.DataFrame(columns=["year", "month", "bin", "n_fp_top5"])

    agg = (
        fp.groupby(["year", "month", "bin"])
        .size()
        .reset_index(name="n_fp_top5")
        .sort_values("n_fp_top5", ascending=False)
        .reset_index(drop=True)
    )
    return agg[["year", "month", "bin", "n_fp_top5"]]


def _extract_fn_hotspots(
    df_pred: pd.DataFrame, thr_fn: float = FN_HOTSPOT_THR
) -> pd.DataFrame:
    """
    从 fullgrid 预测结果中提取 FN-hotspots：
        - label == 1（真实火点）
        - pred < thr_fn（被打得偏低）

    输出列：year, month, bin, n_fn, mu_pred
    """
    d = df_pred.copy()
    d["label"] = d["label"].astype(int)
    d["pred"] = d["pred"].astype(float)

    mask_fn = (d["label"] == 1) & (d["pred"] < float(thr_fn))
    fn = d[mask_fn].copy()
    if fn.empty:
        return pd.DataFrame(columns=["year", "month", "bin", "n_fn", "mu_pred"])

    agg = (
        fn.groupby(["year", "month", "bin"])["pred"]
        .agg(["count", "mean"])
        .reset_index()
    )
    agg = agg.rename(columns={"count": "n_fn", "mean": "mu_pred"})
    agg = agg.sort_values("n_fn", ascending=False).reset_index(drop=True)
    return agg[["year", "month", "bin", "n_fn", "mu_pred"]]


def _build_bucket_stats(
    df_env: pd.DataFrame, p_final: np.ndarray, thr_top5: float
) -> pd.DataFrame:
    """
    构造用于 RL 的粗粒度 bucket 统计：
        bucket = (season, lc_group, elev_band)

    对每个 bucket 统计：
        - n_pos, n_neg
        - mu_pos, cov_pos_ge_high
        - mu_neg, neg_high_rate
        - miss_rate_top5 : 在该 bucket 内，被“全局 Top5%” 区域覆盖不到的火点比例
    """
    df = df_env.copy()
    n = len(df)
    if n == 0:
        return pd.DataFrame(
            columns=[
                "season",
                "lc_group",
                "elev_band",
                "n_pos",
                "n_neg",
                "mu_pos",
                "cov_pos_ge_high",
                "mu_neg",
                "neg_high_rate",
                "miss_rate_top5",
            ]
        )

    if "label" in df.columns:
        label = df["label"].to_numpy().astype(int)
    else:
        label = np.zeros(n, dtype=int)

    # 基础字段
    year = df.get("year", pd.Series([0] * n)).to_numpy()
    month = df.get("month", pd.Series([1] * n)).to_numpy().astype(int)
    pred = np.asarray(p_final, dtype=float)

    # season
    season_list = []
    for m in month:
        if m in (3, 4, 5):
            season_list.append("spring")
        elif m in (6, 7, 8):
            season_list.append("summer")
        elif m in (9, 10, 11):
            season_list.append("autumn")
        else:
            season_list.append("winter")

    # lc_group
    if "lc_raw" in df.columns:
        lc_group = df["lc_raw"].fillna(-1).astype(int).to_numpy()
    else:
        lc_group = np.full(n, -1, dtype=int)

    # elev_band
    if "dem" in df.columns:
        dem = df["dem"].to_numpy()
        elev_band = []
        for v in dem:
            try:
                x = float(v)
            except Exception:
                x = np.nan
            if not np.isfinite(x):
                elev_band.append("unknown")
            elif x < 200:
                elev_band.append("low")
            elif x < 800:
                elev_band.append("mid")
            else:
                elev_band.append("high")
    else:
        elev_band = ["unknown"] * n

    is_top5 = (pred >= float(thr_top5)).astype(int)

    df_stats = pd.DataFrame(
        {
            "year": year,
            "month": month,
            "label": label,
            "pred": pred,
            "season": season_list,
            "lc_group": lc_group,
            "elev_band": elev_band,
            "is_top5": is_top5,
        }
    )

    rows = []
    for (season_key, lc_g, e_band), g in df_stats.groupby(
        ["season", "lc_group", "elev_band"]
    ):
        labels_g = g["label"].to_numpy()
        preds_g = g["pred"].to_numpy(dtype=np.float32)
        is_top5_g = g["is_top5"].to_numpy()

        n_pos = int((labels_g == 1).sum())
        n_neg = int((labels_g == 0).sum())

        if n_pos > 0:
            preds_pos = preds_g[labels_g == 1]
            mu_pos = float(preds_pos.mean())
            cov_pos_ge_high = float((preds_pos >= HIGH_PROB_THR).mean())
            n_pos_in_top5 = int(((labels_g == 1) & (is_top5_g == 1)).sum())
            miss_rate_top5 = float(
                (n_pos - n_pos_in_top5) / float(n_pos)
            )
        else:
            mu_pos = 0.0
            cov_pos_ge_high = 0.0
            miss_rate_top5 = 0.0

        if n_neg > 0:
            preds_neg = preds_g[labels_g == 0]
            mu_neg = float(preds_neg.mean())
            neg_high_rate = float((preds_neg >= HIGH_PROB_THR).mean())
        else:
            mu_neg = 0.0
            neg_high_rate = 0.0

        rows.append(
            {
                "season": str(season_key),
                "lc_group": int(lc_g),
                "elev_band": str(e_band),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "mu_pos": mu_pos,
                "cov_pos_ge_high": cov_pos_ge_high,
                "mu_neg": mu_neg,
                "neg_high_rate": neg_high_rate,
                "miss_rate_top5": miss_rate_top5,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "season",
                "lc_group",
                "elev_band",
                "n_pos",
                "n_neg",
                "mu_pos",
                "cov_pos_ge_high",
                "mu_neg",
                "neg_high_rate",
                "miss_rate_top5",
            ]
        )
    return pd.DataFrame(rows)


# --------------------- Ranking Head 训练 ---------------------


def _train_ranking_head(
    df_env: pd.DataFrame,
    X_env_df: pd.DataFrame,
    y_env: np.ndarray,
    base_pred: np.ndarray,
    train_year: Optional[int] = None,
) -> Tuple[xgb.Booster, List[str], np.ndarray]:
    """
    在 fullgrid 上训练“头部重排器”（rank:ndcg）：
        - group = year*100 + month；
        - 每个 group 内，用 base_pred Top20% 作为候选；
        - 在候选集上训练 rank 模型；
        - 对 fullgrid 全体格点输出 score_rank。

    返回：
        bst_rank    : 训练好的排序模型
        rank_feats  : 使用的特征列名
        score_rank  : 对 fullgrid 全体格点的排序分数（与 df_env 行对齐）
    """
    print("[rank-head] 构建候选集并训练 Ranking Head ...")

    # 用 build_feature_matrix 的列作为基础特征，再附加一列 base_pred
    rank_feats = list(X_env_df.columns) + ["p_base"]
    X_rank_all = X_env_df.copy()
    X_rank_all["p_base"] = base_pred.astype(np.float32)

    df_tmp = df_env[["year", "month"]].copy()
    df_tmp["p_base"] = base_pred.astype(float)
    df_tmp["label"] = y_env.astype(int)

    n_all = len(df_tmp)
    cand_mask = np.zeros(n_all, dtype=bool)
    train_mask = np.ones(n_all, dtype=bool)
    if train_year is not None:
        train_mask = (df_tmp["year"].to_numpy() == int(train_year))

    # 按 year-month 选 Top20% 候选
    for (yy, mm), g in df_tmp.groupby(["year", "month"]):
        idx = g.index.values
        scores = g["p_base"].to_numpy()
        if len(scores) < 5:
            continue
        k = max(1, int(round(len(scores) * RANK_CAND_ALPHA)))
        order = np.argsort(scores)[::-1]
        cand_idx = idx[order[:k]]
        cand_mask[cand_idx] = True
        mid_k = max(1, int(round(len(scores) * 0.40)))
        mid_idx = idx[order[k:mid_k]]
        if len(mid_idx) > 0:
            cand_mask[mid_idx] = True

    n_cand = int(cand_mask.sum())
    print(f"[rank-head] 候选集样本数 = {n_cand:,} / {n_all:,} "
          f"(global ≈ {n_cand / max(n_all,1):.1%})")

    if n_cand < 1000:
        print("[rank-head] 候选样本太少，跳过 Ranking Head，直接使用 base_pred。")
        # 构造一个“恒等”模型的占位符
        dummy_bst = xgb.Booster()
        score_rank = base_pred.astype(float)
        return dummy_bst, rank_feats, score_rank

    X_cand = X_rank_all.loc[cand_mask].copy()
    y_cand = y_env[cand_mask]
    group_cand = (
        df_env.loc[cand_mask, "year"].to_numpy() * 100
        + df_env.loc[cand_mask, "month"].to_numpy()
    ).astype(int)
    w_cand = np.ones(len(y_cand), dtype=np.float32)
    base_cand = df_tmp.loc[cand_mask, "p_base"].to_numpy()
    thr_top20 = np.quantile(base_cand, 1.0 - RANK_CAND_ALPHA)
    top_mask = base_cand >= thr_top20
    w_cand[~top_mask] = 0.3

    # 如有必要，对候选集进行下采样，防止训练过慢
    if n_cand > RANK_MAX_SAMPLES:
        rng = np.random.default_rng(RANDOM_STATE)
        idx_sub = rng.choice(n_cand, size=RANK_MAX_SAMPLES, replace=False)
        X_cand = X_cand.iloc[idx_sub].reset_index(drop=True)
        y_cand = y_cand[idx_sub]
        group_cand = group_cand[idx_sub]
        w_cand = w_cand[idx_sub]
        print(f"[rank-head] 候选集下采样到 {RANK_MAX_SAMPLES:,} 行用于训练。")

    # 排序目标：按 group 排序并设置 group sizes
    order = np.argsort(group_cand)
    Xs = X_cand.iloc[order].to_numpy()
    ys = y_cand[order]
    g_sorted = group_cand[order]
    w_sorted = w_cand[order]
    _, counts = np.unique(g_sorted, return_counts=True)

    dtrain = xgb.DMatrix(Xs, label=ys)
    dtrain.set_group(counts.tolist())

    params_rank = DEFAULT_PARAMS.copy()
    params_rank["objective"] = "rank:ndcg"
    params_rank["eval_metric"] = "ndcg@10"
    params_rank.setdefault("learning_rate", 0.05)
    params_rank.setdefault("max_depth", 5)
    params_rank["seed"] = RANDOM_STATE

    bst_rank = xgb.train(
        params_rank,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    # 对 fullgrid 全体格点打 rank 分数
    d_all = xgb.DMatrix(
        X_rank_all[rank_feats].to_numpy(),
        feature_names=list(rank_feats),
    )
    score_rank = bst_rank.predict(d_all).astype(float)
    print("[rank-head] Ranking Head 训练完成。")
    return bst_rank, rank_feats, score_rank


def _eval_fullgrid_combo(
    bstF,
    bstH,
    f_cols: List[str],
    h_cols: List[str],
    w: float,
    grid_deg: float = 0.1,
    years: Optional[Tuple[int, int]] = None,
):
    """
    针对 fullgrid 评估环境：
        1）跑 F/H 双子模型 + 线性组合得到 base 分数 p_base；
        2）基于 base 的 Top20% 训练 Ranking Head，并得到 score_rank；
        3）对 score_rank 训练一维 Logistic 校准器 p_final；
        4）基于 p_final 计算：
              - fullgrid 四件套（N_total/N_valid + 火点概率分布 + R@Topα + 分位曲线）
              - 全局 Top5% 阈值 thr_top5 及 slot_env（供 RL 使用）
              - FP-hotspots / FN-hotspots 列表；
              - bucket_stats：season×lc_group×elev_band 粗粒度的刻度统计
        5）返回：
              fullgrid_topk_metrics（mode='topk'）
              slot_env
              fullgrid_report
              calibrator
              fp_hotspots
              fn_hotspots
              bucket_stats
              bst_rank
              rank_feats
    """
    env_path = get_eval_env_path(grid_deg=grid_deg, years=years if years is not None else (2017, 2018))
    if not env_path.exists():
        raise FileNotFoundError(f"[fullgrid] env not found: {env_path}")

    df_env = pd.read_parquet(env_path)
    df_env = attach_bin_month_year(df_env, grid_deg=grid_deg)

    # 确保特征列齐全
    tmp = df_env.copy()
    X_env, y_env, feat_cols = build_feature_matrix(tmp)
    X_env_df = pd.DataFrame(X_env, columns=feat_cols, index=tmp.index)

    # 底层 F/H base 分数
    def _ensure_features(X: pd.DataFrame, cols_need: List[str]) -> pd.DataFrame:
        X2 = X.copy()
        for c in cols_need:
            if c not in X2.columns:
                X2[c] = 0.0
        return X2[cols_need]

    XF_env = _ensure_features(X_env_df, f_cols)
    XH_env = _ensure_features(X_env_df, h_cols)

    pF = bstF.predict(
        xgb.DMatrix(XF_env.values, feature_names=list(XF_env.columns))
    )
    pH = bstH.predict(
        xgb.DMatrix(XH_env.values, feature_names=list(XH_env.columns))
    )
    p_base = (w * pF + (1.0 - w) * pH).astype(float)

    # 训练 Ranking Head，并得到 score_rank
    bst_rank, rank_feats, score_rank = _train_ranking_head(
        df_env, X_env_df, y_env, p_base
    )

    # 基于 score_rank 训练 Logistic 校准器
    years_all = sorted(int(y) for y in df_env["year"].unique())
    train_year = years_all[0] if len(years_all) > 0 else None
    calibrator = _fit_logistic_calibrator(score_rank, y_env, train_mask=(df_env["year"].to_numpy() == train_year) if train_year is not None else None)
    if calibrator is not None:
        a = calibrator["coef"]
        b = calibrator["intercept"]
        p_final = 1.0 / (1.0 + np.exp(-(a * score_rank + b)))
    else:
        # 极端情况下的兜底：线性归一化到 [0,1]
        s = score_rank.astype(float)
        s_min, s_max = float(np.min(s)), float(np.max(s))
        if s_max > s_min:
            p_final = (s - s_min) / (s_max - s_min)
        else:
            p_final = np.full_like(s, 0.5, dtype=float)
        print("[calib] 未成功训练校准器，使用线性 rescale 兜底。")

    # 统一 DataFrame d，用 p_final 做后续 TopK / FP / FN 分析
    d = df_env[["year", "month", "bin"]].copy()
    d["label"] = df_env["label"].astype(int)
    d["pred"] = p_final.astype(float)

    # --- 全局 Top5% 阈值 thr_top5 ---
    preds_all = d["pred"].to_numpy(dtype=np.float32)
    n_cells = len(preds_all)
    if n_cells > 0:
        K5 = max(1, int(round(0.05 * n_cells)))
        order = np.argsort(preds_all)
        thr_top5 = float(preds_all[order[-K5]])
    else:
        K5 = 0
        thr_top5 = 1.0

    # --- 每月 high-risk 区（Top5%）召回 & FP-rate ---
    month_rows = []
    for (yy, mm), g in d.groupby(["year", "month"]):
        labels_m = g["label"].to_numpy()
        n_pos_m = int((labels_m == 1).sum())
        if n_pos_m == 0:
            continue

        mask_high = g["pred"] >= thr_top5
        g_high = g[mask_high]
        tp = int((g_high["label"] == 1).sum())
        fp = int((g_high["label"] == 0).sum())
        K = int(len(g_high))

        recall = float(tp / max(n_pos_m, 1))
        fp_rate = float(fp / max(K, 1)) if K > 0 else 0.0

        month_rows.append(
            {
                "year": int(yy),
                "month": int(mm),
                "n_pos_month": n_pos_m,
                "n_high": K,
                "tp_high": tp,
                "fp_high": fp,
                "recall_high": recall,
                "fp_rate_high": fp_rate,
            }
        )

    df_month = pd.DataFrame(month_rows)
    if len(df_month) > 0:
        overall_recall = float(
            df_month["tp_high"].sum() / max(df_month["n_pos_month"].sum(), 1)
        )
        overall_fp_rate = float(
            df_month["fp_high"].sum() / max(df_month["n_high"].sum(), 1)
        ) if df_month["n_high"].sum() > 0 else 0.0
    else:
        overall_recall, overall_fp_rate = 0.0, 0.0

    fullgrid_topk_metrics = {
        "mode": "topk",
        "alpha": 0.05,
        "high_risk_thr": thr_top5,
        "K_top5": K5,
        "overall_recall_high": overall_recall,
        "overall_fp_rate_high": overall_fp_rate,
    }

    # 2）slot_env (year, bin, month) 难度 = miss_rate & contamination
    slot_env = _build_slot_env_top5(d, thr_top5)

    # 3）fullgrid 四件套（基于 p_final）
    fullgrid_report = _compute_fullgrid_report(df_env, p_final, grid_deg=grid_deg)
    calib_metrics = _compute_calibration_metrics(p_final, y_env, n_bins=20)
    try:
        xs = [r["p_mean"] for r in calib_metrics["bins"]]
        ys = [r["pos_frac"] for r in calib_metrics["bins"]]
        plt.figure(figsize=(5, 4))
        plt.plot([0, 1], [0, 1], "k--", label="ideal")
        plt.plot(xs, ys, "o-", label="model")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed frequency")
        ece_val = calib_metrics.get("ece", float("nan"))
        plt.title(f"Reliability (ECE={ece_val:.4f})")
        out_png = BASE_OUT / (f"calibration_{years[0]}_{years[1]}.png" if years is not None else "calibration.png")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
    except Exception:
        pass

    # 4）FP-hotspots 列表（Top5% 内、且 ever_fire_env=False 的假阳性区）
    fp_hotspots = _extract_fp_hotspots(d, thr_top5)

    # 5）FN-hotspots 列表（真实火点且 p_final < FN_HOTSPOT_THR 的区域）
    fn_hotspots = _extract_fn_hotspots(d, thr_fn=FN_HOTSPOT_THR)

    # 6）bucket_stats：season × lc_group × elev_band 粗粒度桶的刻度统计
    bucket_stats = _build_bucket_stats(df_env, p_final, thr_top5)

    # 7）空间阻塞（简版）：按 bin 的哈希分块取一个 holdout 折报告 Top5% 召回
    spatial_holdout = {}
    try:
        bins_list = df_env["bin"].astype(str).unique().tolist()
        sel_bins = [b for b in bins_list if (hash(b) % 5) == 0]
        m_hold = d["bin"].isin(sel_bins)
        preds_h = d.loc[m_hold, "pred"].to_numpy(dtype=np.float32)
        labels_h = d.loc[m_hold, "label"].to_numpy(dtype=int)
        n_cells_h = len(preds_h)
        if n_cells_h > 0:
            K5h = max(1, int(round(0.05 * n_cells_h)))
            order_h = np.argsort(preds_h)
            thr5h = float(preds_h[order_h[-K5h]])
            tp_h = int(((preds_h >= thr5h) & (labels_h == 1)).sum())
            n_pos_h = int((labels_h == 1).sum())
            spatial_holdout = {
                "bins": len(sel_bins),
                "n_cells": n_cells_h,
                "recall_top5": float(tp_h / max(n_pos_h, 1)) if n_pos_h > 0 else 0.0
            }
    except Exception:
        spatial_holdout = {}

    return (
        fullgrid_topk_metrics,
        slot_env,
        fullgrid_report,
        calibrator,
        fp_hotspots,
        fn_hotspots,
        bucket_stats,
        bst_rank,
        rank_feats,
        calib_metrics,
        spatial_holdout,
    )


# --------------------- 迭代指标汇总 ---------------------


def _update_iter_metrics(
    iter_idx: int,
    val_ap: float,
    report: Dict,
    score_S: float,
    out_dir: Path = BASE_OUT,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "iter_metrics.csv"

    topk = report.get("topk_global", {})
    alphas = topk.get("alphas", [])
    recalls = topk.get("recall", [])

    recall_top1 = 0.0
    recall_top5 = 0.0
    recall_top10 = 0.0
    for a, r in zip(alphas, recalls):
        if np.isclose(a, 0.01):
            recall_top1 = float(r)
        elif np.isclose(a, 0.05):
            recall_top5 = float(r)
        elif np.isclose(a, 0.10):
            recall_top10 = float(r)

    fire_bins = report.get("fire_prob_bins", {})
    high_ratio = float(fire_bins.get("high_bin_ratio", 0.0))

    pos_neg_stats = report.get("pos_neg_stats", {})
    mu_pos = float(pos_neg_stats.get("mu_pos", 0.0))
    mu_neg = float(pos_neg_stats.get("mu_neg", 0.0))
    q_pos_0_5 = float(pos_neg_stats.get("q_pos_0.5", 0.0))
    q_pos_0_8 = float(pos_neg_stats.get("q_pos_0.8", 0.0))
    cov_pos_ge_high = float(pos_neg_stats.get("cov_pos_ge_high", 0.0))
    neg_high_rate = float(pos_neg_stats.get("neg_high_rate", 0.0))

    row = {
        "iter": int(iter_idx),
        "val_ap": float(val_ap),
        "score_S": float(score_S),
        "fullgrid_recall_top1": float(recall_top1),
        "fullgrid_recall_top5": float(recall_top5),
        "fullgrid_recall_top10": float(recall_top10),
        "fire_highbin_ratio": float(high_ratio),
        "mu_pos": mu_pos,
        "mu_neg": mu_neg,
        "q_pos_0.5": q_pos_0_5,
        "q_pos_0.8": q_pos_0_8,
        "cov_pos_ge_high": cov_pos_ge_high,
        "neg_high_rate": neg_high_rate,
    }

    try:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = df[df["iter"] != iter_idx]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])

        df = df.sort_values("iter").reset_index(drop=True)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[iter-metrics] 已更新: {csv_path}")
    except Exception as e:
        print(f"[iter-metrics] 写入 {csv_path} 失败 ({e})，本轮跳过 iter_metrics 更新。")
        return

    # 画迭代曲线图（仍然以 Recall@Top5% + high_bin_ratio 为主）
    try:
        if len(df) >= 1:
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(
                df["iter"],
                df["fullgrid_recall_top5"],
                marker="o",
                label="Recall@Top5% (fullgrid)",
            )
            ax1.set_xlabel("Iter")
            ax1.set_ylabel("Recall@Top5%", color="C0")
            ax1.tick_params(axis="y", labelcolor="C0")

            ax2 = ax1.twinx()
            ax2.plot(
                df["iter"],
                df["fire_highbin_ratio"],
                marker="s",
                linestyle="--",
                color="C1",
                label="P(fire in [0.75,1])",
            )
            ax2.set_ylabel("High-risk bin ratio", color="C1")
            ax2.tick_params(axis="y", labelcolor="C1")

            fig.suptitle("Fullgrid Metrics vs Iter (Rank-head + FN-hotspots)")
            fig.tight_layout()
            out_png = out_dir / "iter_curves.png"
            plt.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"[iter-metrics] 迭代曲线图已保存到: {out_png}")
    except Exception as e:  # pragma: no cover
        print(f"[iter-metrics] 绘制迭代曲线失败（不影响训练）: {e}")


# --------------------- 训练主流程 ---------------------


def train_xgb_model(
    df_all: pd.DataFrame, iter_idx: int, grid_deg: float = 0.1
):
    """
    主训练函数：
        - 构建特征矩阵，做防泄漏处理；
        - 划分 train / RL-val / final-test；
        - 训练 flamm / human 双子模型 + 组合权重 w；
        - 利用 fullgrid 环境训练 Ranking Head + 概率校准器；
        - 在 fullgrid 上做统一评估（四件套基于最终概率 p_final）；
        - 写出：
            * xgb_model_flamm_it{iter}.json
            * xgb_model_human_it{iter}.json
            * xgb_model_combo_it{iter}.json
            * xgb_model_rank_it{iter}.json
            * xgb_model_rank_meta_it{iter}.json
            * xgb_calibrator_it{iter}.json
            * feedback_it{iter}.json
            * fullgrid_eval_it{iter}.json
            * iter_metrics.csv（累积）
    """
    BASE_OUT.mkdir(parents=True, exist_ok=True)

    # 1）特征矩阵 & 防泄漏
    X_all, y_all, cols0 = build_feature_matrix(df_all)
    cols = _drop_leak_by_name(list(X_all.columns))
    X_all = X_all[cols]

    leaky_cols = drop_leaky_features_auc(X_all, y_all, threshold=0.999)
    if leaky_cols:
        X_all = X_all.drop(columns=leaky_cols)
        cols = list(X_all.columns)

    f_cols, h_cols = _groups_from_columns(cols)

    df_all2 = df_all.copy()
    df_all2["label"] = y_all

    # 2）基础 sample_weight（负样本结构：neg_type / is_hard）
    sample_w = pd.Series(1.0, index=df_all2.index, dtype=np.float32)

    if "neg_type" in df_all2.columns:
        neg_type = df_all2["neg_type"].fillna(0).astype(int)
        # ultra-safe 负样本：label=0 & neg_type=1
        mask_safe_neg = (df_all2["label"] == 0) & (neg_type == 1)
        n_safe = int(mask_safe_neg.sum())
        if n_safe > 0:
            sample_w.loc[mask_safe_neg] = 0.3
            print(f"[weight] ultra-safe negatives: {n_safe:,} -> weight=0.3")

        # fp_hotspot 负样本：label=0 & neg_type=2
        mask_fp_neg = (df_all2["label"] == 0) & (neg_type == 2)
        n_fp = int(mask_fp_neg.sum())
        if n_fp > 0:
            sample_w.loc[mask_fp_neg] = 1.5
            print(f"[weight] fp-hotspot negatives: {n_fp:,} -> weight=1.5")

    if "is_hard" in df_all2.columns:
        is_hard = df_all2["is_hard"].fillna(False).astype(bool)
        mask_near_fire = (df_all2["label"] == 0) & is_hard
        n_near = int(mask_near_fire.sum())
        if n_near > 0:
            # near-fire hard 负样本较低，但仍具信息量
            sample_w.loc[mask_near_fire] = 0.5
            print(f"[weight] near-fire hard negatives: {n_near:,} -> weight=0.5")

    # 3）FN 假阴性挖矿
    # 3.1 利用上一轮模型在“训练样本世界”里找“低分火点”
    if iter_idx > 1:
        prev_F = BASE_OUT / f"xgb_model_flamm_it{iter_idx - 1}.json"
        prev_H = BASE_OUT / f"xgb_model_human_it{iter_idx - 1}.json"
        try:
            if prev_F.exists() and prev_H.exists():
                bstF_prev = xgb.Booster(model_file=str(prev_F))
                bstH_prev = xgb.Booster(model_file=str(prev_H))

                X_all_df = pd.DataFrame(
                    X_all, columns=cols, index=df_all2.index
                )
                XF_all = X_all_df.reindex(columns=f_cols, fill_value=0.0)
                XH_all = X_all_df.reindex(columns=h_cols, fill_value=0.0)
                pF_all = bstF_prev.predict(
                    xgb.DMatrix(XF_all.values, feature_names=list(XF_all.columns))
                )
                pH_all = bstH_prev.predict(
                    xgb.DMatrix(XH_all.values, feature_names=list(XH_all.columns))
                )

                combo_info_prev_path = BASE_OUT / f"xgb_model_combo_it{iter_idx - 1}.json"
                if combo_info_prev_path.exists():
                    with open(combo_info_prev_path, "r", encoding="utf-8") as f:
                        combo_prev = json.load(f)
                    w_prev = float(combo_prev.get("w", 0.5))
                else:
                    w_prev = 0.5

                p_all = (w_prev * pF_all + (1.0 - w_prev) * pH_all).astype(float)

                # 在所有“正样本”中找分位 < HARD_POS_PERCENTILE 的“难救火点”
                mask_pos = (df_all2["label"].values == 1)
                if mask_pos.any():
                    pos_scores = p_all[mask_pos]
                    thr_fn = float(
                        np.quantile(
                            pos_scores,
                            HARD_POS_PERCENTILE,
                            interpolation="linear",
                        )
                    )
                    hard_pos_mask = mask_pos & (p_all <= thr_fn)
                    n_hard_pos = int(hard_pos_mask.sum())
                    if n_hard_pos > 0:
                        sample_w.loc[hard_pos_mask] *= HARD_POS_WEIGHT_FACTOR
                        print(
                            f"[weight-FN] hard positives: {n_hard_pos:,} "
                            f"(<= {HARD_POS_PERCENTILE:.2f} 分位) "
                            f"-> weight *= {HARD_POS_WEIGHT_FACTOR}"
                        )
        except Exception as e:
            print(f"[weight-FN] 利用上一轮模型做 FN 挖矿时发生异常：{e}，本轮跳过该步。")

    # 3.2 利用上一轮 fullgrid FN-hotspots 进一步对“业务上漏报”的火点升权
    if iter_idx > 1:
        prev_eval_path = BASE_OUT / f"fullgrid_eval_it{iter_idx - 1}.json"
        if prev_eval_path.exists():
            try:
                with open(prev_eval_path, "r", encoding="utf-8") as f:
                    eval_prev = json.load(f)
                fn_list = eval_prev.get("fn_hotspots", [])
                if fn_list:
                    df_fn = pd.DataFrame(fn_list)
                    if not df_fn.empty:
                        for col in ("year", "month"):
                            if col in df_fn.columns:
                                df_fn[col] = df_fn[col].astype(int)
                        df_fn["bin"] = df_fn["bin"].astype(str)

                        key_fn = set(
                            zip(
                                df_fn["year"].tolist(),
                                df_fn["bin"].tolist(),
                                df_fn["month"].tolist(),
                            )
                        )

                        df_all_local = df_all2.copy()
                        # 确保 year/month/bin 存在
                        if ("year" not in df_all_local.columns) or ("month" not in df_all_local.columns) or ("bin" not in df_all_local.columns):
                            df_all_local = attach_bin_month_year(
                                df_all_local, grid_deg=grid_deg
                            )
                        df_all_local["bin"] = df_all_local["bin"].astype(str)
                        df_all_local["year"] = df_all_local["year"].astype(int)
                        df_all_local["month"] = df_all_local["month"].astype(int)

                        keys_samples = list(
                            zip(
                                df_all_local["year"].tolist(),
                                df_all_local["bin"].tolist(),
                                df_all_local["month"].tolist(),
                            )
                        )
                        mask_key = np.array([k in key_fn for k in keys_samples])
                        mask_fn_pos = (
                            (df_all_local["label"].values == 1) & mask_key
                        )

                        n_fn_pos = int(mask_fn_pos.sum())
                        if n_fn_pos > 0:
                            sample_w.loc[df_all_local.index[mask_fn_pos]] *= FN_HOTSPOT_WEIGHT_FACTOR
                            print(
                                f"[weight-FN] fullgrid FN-hotspots positives: {n_fn_pos:,} "
                                f"-> weight *= {FN_HOTSPOT_WEIGHT_FACTOR}"
                            )
            except Exception as e:
                print(f"[weight-FN] 从上一轮 fullgrid_eval 提取 FN-hotspots 失败：{e}，本轮跳过该步。")

    # 4）划分 train / RL-val / final-test
    tr_df, va_df, te_df = split_train_val_test(df_all2)

    X_all_df2 = pd.DataFrame(X_all, columns=cols, index=df_all2.index)

    X_tr, y_tr = X_all_df2.loc[tr_df.index], y_all[tr_df.index]
    X_va, y_va = X_all_df2.loc[va_df.index], y_all[va_df.index]
    X_te, y_te = X_all_df2.loc[te_df.index], y_all[te_df.index]

    w_tr = sample_w.loc[tr_df.index].to_numpy(dtype=np.float32)
    if "is_observed" in df_all2.columns:
        obs_tr = df_all2.loc[tr_df.index, "is_observed"].fillna(1).astype(int).to_numpy()
        neg_mask_tr = (y_tr == 0) & (obs_tr == 0)
        w_tr[neg_mask_tr] = w_tr[neg_mask_tr] * 0.3
    if "hardness_score" in df_all2.columns:
        hs = df_all2.loc[tr_df.index, "hardness_score"].fillna(0.0).astype(float).to_numpy()
        neg_mask_h = (y_tr == 0)
        w_tr[neg_mask_h] = w_tr[neg_mask_h] * (0.5 + 0.5 * np.clip(hs[neg_mask_h], 0.0, 1.0))

    # 排序目标的分组 key（year×month）
    group_tr = None
    group_va = None
    if USE_RANK_OBJECTIVE:
        for df_sub in (tr_df, va_df):
            if "year" not in df_sub.columns or "month" not in df_sub.columns:
                df_sub["date_key"] = pd.to_datetime(df_sub["date_key"])
                df_sub["year"] = df_sub["date_key"].dt.year.astype(int)
                df_sub["month"] = df_sub["date_key"].dt.month.astype(int)
        group_tr = tr_df["year"].to_numpy() * 100 + tr_df["month"].to_numpy()
        group_va = va_df["year"].to_numpy() * 100 + va_df["month"].to_numpy()

    def _ensure_features2(X: pd.DataFrame, cols_need: List[str]) -> pd.DataFrame:
        X2 = X.copy()
        for c in cols_need:
            if c not in X2.columns:
                X2[c] = 0.0
        return X2[cols_need]

    XF_tr = _ensure_features2(X_tr, f_cols)
    XH_tr = _ensure_features2(X_tr, h_cols)
    XF_va = _ensure_features2(X_va, f_cols)
    XH_va = _ensure_features2(X_va, h_cols)
    XF_te = _ensure_features2(X_te, f_cols)
    XH_te = _ensure_features2(X_te, h_cols)

    prev_F = BASE_OUT / f"xgb_model_flamm_it{iter_idx - 1}.json"
    prev_H = BASE_OUT / f"xgb_model_human_it{iter_idx - 1}.json"

    # 5）训练双子模型
    if iter_idx == 1 or (not prev_F.exists()):
        bstF, aucF, apF, pF = _train_xgb(
            XF_tr,
            y_tr,
            w_tr,
            XF_va,
            y_va,
            group_tr=group_tr,
            group_va=group_va,
            base_params=DEFAULT_PARAMS,
            use_optuna=(USE_OPTUNA and iter_idx == 1),
        )
        bstH, aucH, apH, pH = _train_xgb(
            XH_tr,
            y_tr,
            w_tr,
            XH_va,
            y_va,
            group_tr=group_tr,
            group_va=group_va,
            base_params=DEFAULT_PARAMS,
            use_optuna=(USE_OPTUNA and iter_idx == 1),
        )
    else:
        print("[combo] 加载上一轮模型进行增量微调...")
        dtrF = xgb.DMatrix(XF_tr, label=y_tr, weight=w_tr)
        dvaF = xgb.DMatrix(XF_va, label=y_va)
        dtrH = xgb.DMatrix(XH_tr, label=y_tr, weight=w_tr)
        dvaH = xgb.DMatrix(XH_va, label=y_va)
        bstF_prev = xgb.Booster(model_file=str(prev_F))
        bstH_prev = xgb.Booster(model_file=str(prev_H))

        ft_params = DEFAULT_PARAMS.copy()
        ft_params["learning_rate"] = 0.01
        if USE_RANK_OBJECTIVE:
            ft_params["objective"] = "rank:ndcg"
            ft_params["eval_metric"] = "ndcg@10"
        else:
            ft_params["objective"] = "binary:logistic"
            ft_params["eval_metric"] = "auc"

        bstF = xgb.train(
            ft_params,
            dtrF,
            num_boost_round=ADDITIONAL_ROUNDS,
            evals=[(dtrF, "train"), (dvaF, "val")],
            early_stopping_rounds=50,
            verbose_eval=100,
            xgb_model=bstF_prev,
        )
        bstH = xgb.train(
            ft_params,
            dtrH,
            num_boost_round=ADDITIONAL_ROUNDS,
            evals=[(dtrH, "train"), (dvaH, "val")],
            early_stopping_rounds=50,
            verbose_eval=100,
            xgb_model=bstH_prev,
        )

    # 6）在 RL-val 上选择组合权重 w
    pF_va = bstF.predict(xgb.DMatrix(XF_va))
    pH_va = bstH.predict(xgb.DMatrix(XH_va))
    w_star, auc_star, ap_star = _pick_w_by_auc(y_va, pF_va, pH_va)
    print(
        f"[combo] best w={w_star:.2f} -> AUC={auc_star:.5f}, AP={ap_star:.5f}"
    )

    # 7）保存 F/H 模型 & 组合配置
    model_F = BASE_OUT / f"xgb_model_flamm_it{iter_idx}.json"
    model_H = BASE_OUT / f"xgb_model_human_it{iter_idx}.json"
    bstF.save_model(model_F)
    bstH.save_model(model_H)

    combo_info = {
        "w": float(w_star),
        "flamm_cols": list(f_cols),
        "human_cols": list(h_cols),
    }
    with open(
        BASE_OUT / f"xgb_model_combo_it{iter_idx}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(combo_info, f, ensure_ascii=False, indent=2)

    # 8）fullgrid 评估 + Ranking Head + 概率校准 + FP/FN-hotspot + bucket_stats
    eval_windows = []
    for yrs in [(2015, 2016), (2017, 2018), (2019, 2020)]:
        try:
            ensure_eval_fullgrid_env(grid_deg=grid_deg, years=yrs)
            (
                fullgrid_topk_metrics,
                slot_env,
                fullgrid_report,
                calibrator,
                fp_hotspots,
                fn_hotspots,
                bucket_stats,
                bst_rank,
                rank_feats,
                calib_metrics,
                spatial_holdout,
            ) = _eval_fullgrid_combo(
                bstF, bstH, f_cols, h_cols, w_star, grid_deg=grid_deg, years=yrs
            )
            # 抽取核心窗口指标
            topk = fullgrid_report.get("topk_global", {})
            alphas = topk.get("alphas", [])
            recalls = topk.get("recall", [])
            r1 = r5 = r10 = 0.0
            for a, r in zip(alphas, recalls):
                if np.isclose(a, 0.01): r1 = float(r)
                elif np.isclose(a, 0.05): r5 = float(r)
                elif np.isclose(a, 0.10): r10 = float(r)
            fp_rate_high = float(fullgrid_topk_metrics.get("overall_fp_rate_high", 0.0))
            fire_bins = fullgrid_report.get("fire_prob_bins", {})
            high_ratio = float(fire_bins.get("high_bin_ratio", 0.0))
            pos_neg_stats = fullgrid_report.get("pos_neg_stats", {})
            mu_pos = float(pos_neg_stats.get("mu_pos", 0.0))
            mu_neg = float(pos_neg_stats.get("mu_neg", 0.0))
            neg_high_rate = float(pos_neg_stats.get("neg_high_rate", 0.0))
            eval_windows.append({
                "years": list(yrs),
                "recall_top1": r1,
                "recall_top5": r5,
                "recall_top10": r10,
                "fp_rate_high": fp_rate_high,
                "high_bin_ratio": high_ratio,
                "mu_pos": mu_pos,
                "mu_neg": mu_neg,
                "neg_high_rate": neg_high_rate,
                "ece": float(calib_metrics.get("ece", float("nan"))),
                "spatial_holdout_recall_top5": float(spatial_holdout.get("recall_top5", 0.0)),
            })
            # 仅保留最后一次构建的对象用于写文件
            last_fullgrid_topk_metrics = fullgrid_topk_metrics
            last_slot_env = slot_env
            last_fullgrid_report = fullgrid_report
            last_calibrator = calibrator
            last_fp_hotspots = fp_hotspots
            last_fn_hotspots = fn_hotspots
            last_bucket_stats = bucket_stats
            last_bst_rank = bst_rank
            last_rank_feats = rank_feats
            last_calib_metrics = calib_metrics
            last_spatial_holdout = spatial_holdout
        except Exception as e:
            print(f"[fullgrid-eval] 窗口 {yrs} 评估失败：{e}，跳过该窗口。")
            continue

    # 使用主窗口（2017-2018）报告综合指标
    fullgrid_report = last_fullgrid_report
    fullgrid_topk_metrics = last_fullgrid_topk_metrics
    slot_env = last_slot_env
    calibrator = last_calibrator
    fp_hotspots = last_fp_hotspots
    fn_hotspots = last_fn_hotspots
    bucket_stats = last_bucket_stats
    bst_rank = last_bst_rank
    rank_feats = last_rank_feats
    calib_metrics = last_calib_metrics
    spatial_holdout = last_spatial_holdout
    topk = fullgrid_report.get("topk_global", {})
    alphas = topk.get("alphas", [])
    recalls = topk.get("recall", [])

    recall_top1 = recall_top5 = recall_top10 = 0.0
    for a, r in zip(alphas, recalls):
        if np.isclose(a, 0.01):
            recall_top1 = float(r)
        elif np.isclose(a, 0.05):
            recall_top5 = float(r)
        elif np.isclose(a, 0.10):
            recall_top10 = float(r)

    fire_bins = fullgrid_report.get("fire_prob_bins", {})
    high_ratio = float(fire_bins.get("high_bin_ratio", 0.0))

    pos_neg_stats = fullgrid_report.get("pos_neg_stats", {})
    mu_pos = float(pos_neg_stats.get("mu_pos", 0.0))
    mu_neg = float(pos_neg_stats.get("mu_neg", 0.0))
    q_pos_0_5 = float(pos_neg_stats.get("q_pos_0.5", 0.0))
    q_pos_0_8 = float(pos_neg_stats.get("q_pos_0.8", 0.0))
    cov_pos_ge_high = float(pos_neg_stats.get("cov_pos_ge_high", 0.0))
    neg_high_rate = float(pos_neg_stats.get("neg_high_rate", 0.0))

    # 9）综合得分 S_it（以 R@TopK + AP 为主）
    score_S = (
        SCORE_W_R1 * float(recall_top1)
        + SCORE_W_R5 * float(recall_top5)
        + SCORE_W_R10 * float(recall_top10)
        + SCORE_W_AP * float(ap_star)
    )
    print(
        f"[score] iter={iter_idx}, "
        f"AP_rlval={ap_star:.4f}, "
        f"R@1_fg={recall_top1:.4f}, "
        f"R@5_fg={recall_top5:.4f}, "
        f"R@10_fg={recall_top10:.4f} "
        f"-> S_it={score_S:.4f}"
    )

    # 10）fullgrid_eval_it{iter}.json
    # 计算稳定性指标
    def _std_of(key: str) -> float:
        vals = [float(w.get(key, 0.0)) for w in eval_windows if np.isfinite(w.get(key, 0.0))]
        return float(np.std(vals)) if len(vals) >= 2 else 0.0
    stability = {
        "std_recall_top5": _std_of("recall_top5"),
        "std_neg_high_rate": _std_of("neg_high_rate"),
        "std_ece": _std_of("ece"),
    }

    fullgrid_eval = {
        "iter": int(iter_idx),
        "grid_deg": float(grid_deg),
        "window_years": [2017, 2018],
        "metrics": {
            "val_auc": float(auc_star),
            "val_ap": float(ap_star),
            "score_S": float(score_S),
            "fullgrid_recall_top1": float(recall_top1),
            "fullgrid_recall_top5": float(recall_top5),
            "fullgrid_recall_top10": float(recall_top10),
            "fire_highbin_ratio": float(high_ratio),
            "mu_pos": mu_pos,
            "mu_neg": mu_neg,
            "q_pos_0.5": q_pos_0_5,
            "q_pos_0.8": q_pos_0_8,
            "cov_pos_ge_high": cov_pos_ge_high,
            "neg_high_rate": neg_high_rate,
            "N_total_fires": int(fullgrid_report.get("N_total_fires", 0)),
            "N_valid_fires": int(fullgrid_report.get("N_valid_fires", 0)),
            "ece": float(calib_metrics.get("ece", float("nan"))),
            "spatial_holdout_recall_top5": float(spatial_holdout.get("recall_top5", 0.0)),
            "stability_std_recall_top5": float(stability["std_recall_top5"]),
            "stability_std_neg_high_rate": float(stability["std_neg_high_rate"]),
            "stability_std_ece": float(stability["std_ece"]),
        },
        "fullgrid_report": fullgrid_report,
        "fullgrid_topk_metrics": fullgrid_topk_metrics,
        "fp_hotspots": fp_hotspots.to_dict(orient="records"),
        "fn_hotspots": fn_hotspots.to_dict(orient="records"),
        "bucket_stats": bucket_stats.to_dict(orient="records"),
        "calibration_bins": calib_metrics.get("bins", []),
        "spatial_holdout": spatial_holdout,
        "eval_windows": eval_windows,
    }
    eval_path = BASE_OUT / f"fullgrid_eval_it{iter_idx}.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(fullgrid_eval, f, ensure_ascii=False, indent=2)
    print(f"[fullgrid-eval] 写入：{eval_path}")

    # 11）概率校准器
    if calibrator is not None:
        calib_path = BASE_OUT / f"xgb_calibrator_it{iter_idx}.json"
        with open(calib_path, "w", encoding="utf-8") as f:
            json.dump(calibrator, f, ensure_ascii=False, indent=2)
        print(f"[calib] 概率校准器已写入：{calib_path}")
    else:
        print("[calib] 本轮未生成校准器（可能是正负样本不足）。")

    # 12）Ranking Head 模型 & meta 信息
    rank_model_path = BASE_OUT / f"xgb_model_rank_it{iter_idx}.json"
    bst_rank.save_model(rank_model_path)
    rank_meta = {
        "features": list(rank_feats),
        "cand_alpha": float(RANK_CAND_ALPHA),
        "train_year": int(train_year) if train_year is not None else None
    }
    rank_meta_path = BASE_OUT / f"xgb_model_rank_meta_it{iter_idx}.json"
    with open(rank_meta_path, "w", encoding="utf-8") as f:
        json.dump(rank_meta, f, ensure_ascii=False, indent=2)
    print(f"[rank-head] 模型已写入：{rank_model_path}")
    print(f"[rank-head] meta 已写入：{rank_meta_path}")

    # 13）feedback_it{iter}.json（供 Bandit 采样器使用）
    feedback = {
        "iter": int(iter_idx),
        "window_years": [2017, 2018],
        "metrics": {
            "val_auc": float(auc_star),
            "val_ap": float(ap_star),
            "score_S": float(score_S),
            "fullgrid_recall_top1": float(recall_top1),
            "fullgrid_recall_top5": float(recall_top5),
            "fullgrid_recall_top10": float(recall_top10),
            "fire_highbin_ratio": float(high_ratio),
            "mu_pos": mu_pos,
            "mu_neg": mu_neg,
            "q_pos_0.5": q_pos_0_5,
            "q_pos_0.8": q_pos_0_8,
            "cov_pos_ge_high": cov_pos_ge_high,
            "neg_high_rate": neg_high_rate,
            "ece": float(calib_metrics.get("ece", float("nan"))),
            "spatial_holdout_recall_top5": float(spatial_holdout.get("recall_top5", 0.0)),
            "stability_std_recall_top5": float(stability["std_recall_top5"]),
            "stability_std_neg_high_rate": float(stability["std_neg_high_rate"]),
            "stability_std_ece": float(stability["std_ece"]),
        },
        "slot_stats_env": slot_env.to_dict(orient="records"),
        "fullgrid_topk_metrics": fullgrid_topk_metrics,
        "combo": {"w": float(w_star)},
        "fp_hotspots": fp_hotspots.to_dict(orient="records"),
        "fn_hotspots": fn_hotspots.to_dict(orient="records"),
        "bucket_stats": bucket_stats.to_dict(orient="records"),
        "calibration_bins": calib_metrics.get("bins", []),
        "spatial_holdout": spatial_holdout,
        "eval_windows": eval_windows,
    }
    fb_path = BASE_OUT / f"feedback_it{iter_idx}.json"
    with open(fb_path, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)
    print(f"[feedback] 写入：{fb_path}")

    # 14）更新迭代曲线
    _update_iter_metrics(iter_idx, ap_star, fullgrid_report, score_S, out_dir=BASE_OUT)

    return fb_path, model_F
