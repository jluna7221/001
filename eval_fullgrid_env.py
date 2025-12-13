# -*- coding: utf-8 -*-
"""
eval_fullgrid_env.py (v5.2 - 业务规则修正版)

修订内容：
1. [关键修正] _infer_forest_classes_from_firms: 不再自动推断。
   根据用户业务规则，强制指定 LC 值 1-10 均为森林类别，确保评估覆盖全省所有林地。
2. 保持之前的 n_bins 广播修复和 NaN 处理。
"""

from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta
from typing import Tuple, List, Optional
import re
import numpy as np
import pandas as pd

# 工具链导入
from gen_negative_samples import _load_firms_master
import extract_met_features_cdmet as met_mod
import add_festival_features as holi_mod
import build_static_features as static_mod

try:
    import rasterio
except Exception:
    rasterio = None

BASE = Path(r"G:\fire")
OUT_DIR = BASE / "outputs"

GRID_DEG_DEFAULT: float = 0.1
EVAL_YEARS_DEFAULT: Tuple[int, int] = (2017, 2018)


def get_eval_paths(years: Tuple[int, int] = EVAL_YEARS_DEFAULT, grid_deg: float = GRID_DEG_DEFAULT):
    y0, y1 = years
    g_str = str(grid_deg).replace(".", "p")
    suf = f"{y0}_{y1}_g{g_str}"
    samples = OUT_DIR / f"eval_fullgrid_samples_{suf}.parquet"
    met = OUT_DIR / f"eval_met_features_{suf}.parquet"
    holi = OUT_DIR / f"eval_holiday_features_{suf}.parquet"
    static_p = OUT_DIR / f"eval_static_features_{suf}.parquet"
    env = OUT_DIR / f"eval_env_fullgrid_{suf}.parquet"
    return samples, met, holi, static_p, env


def get_eval_env_path(grid_deg: float = GRID_DEG_DEFAULT, years: Tuple[int, int] = EVAL_YEARS_DEFAULT) -> Path:
    *_, env = get_eval_paths(years=years, grid_deg=grid_deg)
    return env


def _build_year2path_generic(dir_path: Path) -> dict:
    mapping = {}
    if not dir_path.exists(): return mapping
    for p in dir_path.glob("*.tif"):
        m = re.search(r"(\d{4})", p.stem)
        if m: mapping[int(m.group(1))] = p
    return mapping


def _nearest_year_path(year: int, mapping: dict) -> Optional[Path]:
    if not mapping: return None
    years = sorted(mapping.keys())
    best = min(years, key=lambda y: abs(y - year))
    return mapping[best]


def _infer_forest_classes_from_firms(lc_path: Path) -> List[int]:
    """
    [Modified] 业务规则修正：
    用户明确指出 LC 数据中 1-10 均代表森林范围。
    不再根据火点分布自动推断，直接返回 1~10，防止漏掉火点较少但仍属林地的区域。
    """
    forest_classes = list(range(1, 11))  # [1, 2, ..., 10]
    print(f"[LC] 根据业务规则，强制指定森林类别集合：{forest_classes}")
    return forest_classes


def _build_grid_df_from_lc(grid_deg: float, years: Tuple[int, int]) -> Optional[pd.DataFrame]:
    lc_dir = static_mod.LC_DIR
    mapping = _build_year2path_generic(lc_dir)
    p2017 = _nearest_year_path(2017, mapping)
    p2018 = _nearest_year_path(2018, mapping)
    p_ref = p2018 or p2017
    if (rasterio is None) or (p_ref is None) or (not p_ref.exists()):
        return None

    with rasterio.open(str(p_ref)) as ds:
        west, south, east, north = ds.bounds

    lat_edges = np.arange(south, north, grid_deg)
    lon_edges = np.arange(west, east, grid_deg)
    centers = [(la + grid_deg / 2, lo + grid_deg / 2) for la in lat_edges for lo in lon_edges]
    grid_df = pd.DataFrame(centers, columns=["lat_center", "lon_center"])

    i = np.floor((grid_df["lat_center"].to_numpy() + 90.0) / grid_deg).astype(int)
    j = np.floor((grid_df["lon_center"].to_numpy() + 180.0) / grid_deg).astype(int)
    grid_df["bin"] = [f"{ii}_{jj}" for ii, jj in zip(i, j)]
    grid_df = grid_df.drop_duplicates(subset=["bin"]).reset_index(drop=True)

    # 获取森林类别 (1-10)
    forest_classes = _infer_forest_classes_from_firms(p_ref)

    for_year = []
    for y, p in [(2017, p2017), (2018, p2018)]:
        if p is None: continue
        vals = static_mod.sample_tif_points(p, grid_df["lon_center"].to_numpy(), grid_df["lat_center"].to_numpy())

        # 处理 NaN 并筛选
        vals = np.nan_to_num(vals, nan=-9999)
        keep = np.isin(vals.astype(np.int64), np.array(forest_classes, dtype=np.int64))

        tmp = grid_df.loc[keep].copy()
        for_year.append(tmp)

    if not for_year: return grid_df
    kept = pd.concat(for_year, ignore_index=True).drop_duplicates(subset=["bin"])
    return kept[["lat_center", "lon_center", "bin"]].reset_index(drop=True)


def build_eval_fullgrid_samples(grid_deg: float = GRID_DEG_DEFAULT,
                                years: Tuple[int, int] = EVAL_YEARS_DEFAULT) -> Path:
    samples_path, *_ = get_eval_paths(years=years, grid_deg=grid_deg)
    if samples_path.exists(): return samples_path

    y0, y1 = years
    grid_df = _build_grid_df_from_lc(grid_deg=grid_deg, years=years)
    if grid_df is None:
        firms_all = _load_firms_master()
        firms = firms_all[(firms_all["date_key"].dt.year >= y0) & (firms_all["date_key"].dt.year <= y1)].copy()
        from gen_negative_samples import _build_grid_df_from_firms
        grid_df = _build_grid_df_from_firms(firms, grid_deg=grid_deg)

    bins = grid_df["bin"].to_numpy()
    lat_centers = grid_df["lat_center"].to_numpy()
    lon_centers = grid_df["lon_center"].to_numpy()

    start_date = date(y0, 1, 1)
    end_date = date(y1, 12, 31)
    days = (end_date - start_date).days + 1
    date_list = [start_date + timedelta(days=i) for i in range(days)]

    # 修复广播错误：n_bins 必须是整数
    n_bins = len(bins)

    date_arr = np.repeat(np.array(date_list, dtype="datetime64[D]"), n_bins)
    bin_arr = np.tile(bins, days)
    lat_arr = np.tile(lat_centers, days)
    lon_arr = np.tile(lon_centers, days)

    df = pd.DataFrame({"date_key": date_arr, "bin": bin_arr, "lat": lat_arr, "lon": lon_arr})
    df["date_dt"] = pd.to_datetime(df["date_key"])
    df["sample_id"] = df["date_dt"].dt.strftime("%Y%m%d").radd("eval_") + "_" + df["bin"].astype(str)

    firms_all = _load_firms_master()
    firms = firms_all[(firms_all["date_key"].dt.year >= y0) & (firms_all["date_key"].dt.year <= y1)][
        ["date_key", "bin"]].copy()
    firms["date_dt"] = pd.to_datetime(firms["date_key"]).dt.normalize()
    firms = firms.drop_duplicates(subset=["date_dt", "bin"])
    firms["label"] = 1

    df["date_dt"] = df["date_dt"].dt.normalize()
    df = df.merge(firms[["date_dt", "bin", "label"]], on=["date_dt", "bin"], how="left")
    df["label"] = df["label"].fillna(0).astype(np.int8)
    df["date_key"] = df["date_dt"].dt.strftime("%Y-%m-%d")
    df = df.drop(columns=["date_dt"])
    df["is_observed"] = 1

    samples_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(samples_path, index=False)
    return samples_path


def ensure_eval_fullgrid_env(grid_deg: float = GRID_DEG_DEFAULT, years: Tuple[int, int] = EVAL_YEARS_DEFAULT) -> Path:
    samples_path, met_path, holi_path, static_path, env_path = get_eval_paths(years=years, grid_deg=grid_deg)
    if env_path.exists(): return env_path

    samples_path = build_eval_fullgrid_samples(grid_deg=grid_deg, years=years)

    orig_samples_path = met_mod.SAMPLES_PATH
    orig_out_path = met_mod.OUT_PATH
    try:
        met_mod.SAMPLES_PATH = samples_path
        met_mod.OUT_PATH = met_path
        met_mod.extract_met_features()
    finally:
        met_mod.SAMPLES_PATH = orig_samples_path
        met_mod.OUT_PATH = orig_out_path

    orig_samples_holi, orig_out_holi = holi_mod.SAMPLES, holi_mod.OUT
    try:
        holi_mod.SAMPLES = samples_path
        holi_mod.OUT = holi_path
        holi_mod.add_festival_features()
    finally:
        holi_mod.SAMPLES, holi_mod.OUT = orig_samples_holi, orig_out_holi

    orig_samples_static, orig_met_path, orig_holi_path, orig_static_out = static_mod.SAMPLES, static_mod.MET_PATH, static_mod.HOLI_PATH, static_mod.STATIC_OUT
    try:
        static_mod.SAMPLES = samples_path
        static_mod.MET_PATH = met_path
        static_mod.HOLI_PATH = holi_path
        static_mod.STATIC_OUT = static_path
        static_mod.build_static_features()
        df_env = static_mod.load_and_join()
        df_env = static_mod.attach_bin_month_year(df_env, grid_deg=grid_deg)
        env_path.parent.mkdir(parents=True, exist_ok=True)
        df_env.to_parquet(env_path, index=False)
    finally:
        static_mod.SAMPLES, static_mod.MET_PATH, static_mod.HOLI_PATH, static_mod.STATIC_OUT = orig_samples_static, orig_met_path, orig_holi_path, orig_static_out

    return env_path


if __name__ == "__main__":
    ensure_eval_fullgrid_env(grid_deg=GRID_DEG_DEFAULT, years=EVAL_YEARS_DEFAULT)
