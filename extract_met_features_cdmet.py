# -*- coding: utf-8 -*-
"""
extract_met_features_cdmet.py

从 Zarr 抽取 CDMet 日值到样本点（“网格卷积 + 一次插值”加速版）

特征设计：
- 当天量：
    Mate  : 日最高气温 (tmax)
    Mite  : 日最低气温 (tmin)
    Ate   : 当天平均气温 ( (tmax + tmin) / 2 )
    Arh   : 当天相对湿度
    Aws   : 当天风速
    Suh   : 当天日照 / 短波辐射（视你在 prep_meteorology_cdmet 中定义）
    Pre   : 当天降水量

- 时间窗口（全部在“格点场”上先做卷积，再对样本做一次插值）：
    Pre_3d_sum   : 过去 3 天累积降水（含当天）
    Pre_7d_sum   : 过去 7 天累积降水（含当天）
    Pre_30d_sum  : 过去 30 天累积降水（含当天）
    Ate_3d_mean  : 过去 3 天平均气温
    Arh_3d_mean  : 过去 3 天平均相对湿度
    Aws_3d_mean  : 过去 3 天平均风速

- Raindate：
    Raindate     : 向前最多 30 天，距离最近一次“有雨日”的天数
                   0 表示当天有雨，1 表示前一天有雨，>=30 表示 30 天内都无明显降水

实现思路（方案 3）：
1. 时间维度按 CDMet 的 time 轴从头到尾顺序扫一次；
2. 对每一天 t，先在“格点场”上用滑动窗口更新：
      - pre 的 3/7/30 日累积、Raindate
      - ate / rhu / win 的 3 日均值
3. 再看看这一天是否有样本（按 date_key 分组）：
      - 如果有，就把这一时刻的各个格点场（当天量 + 窗口量）在经纬度上做一次最近邻插值，
        得到所有样本点的气象特征。
4. 全过程不依赖 dask，只用 xarray + numpy，内存中只保存少量“最近 30 天”的二维格点窗口，
   既避免了之前的“逐点多次插值”极其耗时，也避免一次性把整套 20 年 3D 气象场全展开到内存。

注意：
- 代码假设 prep_meteorology_cdmet.py 生成的 Zarr 目录结构为：
    G:\\fire\\staging\\met_zarr\\maxtmp.zarr
    G:\\fire\\staging\\met_zarr\\mintmp.zarr
    G:\\fire\\staging\\met_zarr\\pre.zarr
    G:\\fire\\staging\\met_zarr\\win.zarr
    G:\\fire\\staging\\met_zarr\\rhu.zarr
    G:\\fire\\staging\\met_zarr\\sst.zarr
  且各变量的 lat / lon / time 坐标完全一致（prep 脚本已经保证了）。
"""

from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr


# ========= 固定路径 =========
BASE = Path(r"G:\fire")
SAMPLES_PATH = BASE / r"outputs\samples_all.parquet"
ZARR_DIR = BASE / r"staging\met_zarr"               # 形如 maxtmp.zarr / pre.zarr / ...
OUT_PATH = BASE / r"outputs\met_features.parquet"

# 时间维名称（与 prep_meteorology_cdmet 保持一致）
DIM_TIME = "time"
DIM_LAT = "lat"
DIM_LON = "lon"

# 逻辑时间窗（与你整体项目保持一致）
DATE_MIN = datetime(2001, 1, 1).date()
DATE_MAX = datetime(2020, 12, 31).date()

# Raindate 相关
RAINDATE_THRESHOLD = 0.1  # mm，认为“有雨”的最小降水量
RAINDATE_MAX_DAYS = 30    # 最多回看 30 天


# ========= 基础工具 =========
def _load_zarr_var(name: str) -> xr.DataArray:
    """
    读取 Zarr 变量；Zarr 目录命名为 {name}.zarr，内部变量名为 {name}（或唯一 data_var）。
    会对 lat / lon / time 做排序，以保证单调递增，方便后续插值。
    """
    z = ZARR_DIR / f"{name}.zarr"
    if not z.exists():
        raise FileNotFoundError(f"气象 Zarr 文件不存在：{z}")

    ds = xr.open_zarr(z)
    if name in ds.data_vars:
        da = ds[name]
    else:
        # 某些写法可能把变量名写成固定名，兜底取唯一 data_var
        data_vars = list(ds.data_vars)
        if len(data_vars) == 1:
            da = ds[data_vars[0]]
        else:
            raise KeyError(f"{z} 内未找到变量 '{name}'，可用变量={data_vars}")

    # 统一排序，保证 lat / lon / time 单调递增
    if DIM_LAT in da.dims:
        da = da.sortby(DIM_LAT)
    if DIM_LON in da.dims:
        da = da.sortby(DIM_LON)
    if DIM_TIME in da.dims:
        da = da.sortby(DIM_TIME)

    return da


def _nearest_index_1d(coord: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    在一维坐标 coord（升序）上，为每个 values 找到最近的坐标索引（最近邻）。
    """
    coord = np.asarray(coord)
    values = np.asarray(values)

    idx = np.searchsorted(coord, values)
    idx = np.clip(idx, 1, len(coord) - 1)

    left = coord[idx - 1]
    right = coord[idx]
    choose_left = (values - left) <= (right - values)
    idx = idx - choose_left.astype(np.int64)
    return idx


def _interp2d_nearest(
    grid: np.ndarray,
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """
    简单的二维最近邻插值：
    - grid: 2D 数组，shape = (ny, nx)，对应纬度 lat_arr、经度 lon_arr
    - lat_arr: 1D 纬度数组，升序
    - lon_arr: 1D 经度数组，升序
    - lats, lons: 要插值的点坐标（同长度）

    返回：每个点对应的 grid 值，一维 float32 数组。
    """
    if grid.ndim != 2:
        raise ValueError(f"grid 必须是 2D 数组，当前 ndim={grid.ndim}")
    if len(lat_arr.shape) != 1 or len(lon_arr.shape) != 1:
        raise ValueError("lat_arr / lon_arr 必须是一维数组")

    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)

    iy = _nearest_index_1d(lat_arr, lats)
    ix = _nearest_index_1d(lon_arr, lons)

    vals = grid[iy, ix]
    return vals.astype(np.float32)


def _read_samples() -> pd.DataFrame:
    """
    读取 samples_all.parquet，并做基础清洗：
    - 保留 sample_id / lon / lat / date_key
    - date_key -> datetime64[ns]
    - 限制在 DATE_MIN ~ DATE_MAX 范围内
    """
    if not SAMPLES_PATH.exists():
        raise FileNotFoundError(f"未找到样本文件：{SAMPLES_PATH}")

    df = pd.read_parquet(SAMPLES_PATH)
    need_cols = ["sample_id", "lon", "lat", "date_key"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"samples_all 缺少必要列：{c}")

    df = df.copy()
    df = df.dropna(subset=["lon", "lat", "date_key"])
    df["date_dt"] = pd.to_datetime(df["date_key"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date_dt"])
    df["date_dt"] = df["date_dt"].dt.date

    # 双保险：限制在 DATE_MIN ~ DATE_MAX
    before_min = min(df["date_dt"])
    before_max = max(df["date_dt"])
    df = df[(df["date_dt"] >= DATE_MIN) & (df["date_dt"] <= DATE_MAX)].reset_index(drop=True)
    after_min = min(df["date_dt"])
    after_max = max(df["date_dt"])

    print(
        f"[samples] rows={len(df):,} "
        f"date range: {before_min}~{before_max} -> clipped to {after_min}~{after_max}"
    )

    return df


def _align_dates_to_met(
    df: pd.DataFrame,
    met_time: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[np.datetime64, int]]:
    """
    将样本日期对齐到气象时间轴上的最近日期。
    - df["date_dt"] 当前是 python date
    - met_time: xarray time 轴（np.datetime64[ns] 数组），已排序
    返回：
    - new_df  : 增加一列 "met_date"（np.datetime64[ns]），用于与气象时间匹配
    - time_arr: met_time（np.datetime64[ns]）
    - time_index_map: dict[met_date] -> 整数索引 t
    """
    # 转成 datetime64[ns]
    time_arr = pd.to_datetime(met_time).values  # np.datetime64[ns]
    t_min = time_arr.min()
    t_max = time_arr.max()

    # 先把 date_dt 转成 datetime64[ns]（当天 00:00）
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date_dt"]).values

    # 超出 met 范围的先裁剪到边界，再对齐到最近日期
    before_min = df["date_dt"].min()
    before_max = df["date_dt"].max()
    df.loc[df["date_dt"] < t_min, "date_dt"] = t_min
    df.loc[df["date_dt"] > t_max, "date_dt"] = t_max
    after_min = df["date_dt"].min()
    after_max = df["date_dt"].max()

    print(
        f"[time clip] met range: {str(t_min)[:10]}~{str(t_max)[:10]} | "
        f"samples: {str(before_min)[:10]}~{str(before_max)[:10]} -> "
        f"{str(after_min)[:10]}~{str(after_max)[:10]}"
    )

    # 对所有“唯一日期”做一次“找最近 met_time”的映射，再回填
    unique_dates = np.unique(df["date_dt"].values)
    mapping: Dict[np.datetime64, np.datetime64] = {}
    time_index_map: Dict[np.datetime64, int] = {}

    for d in unique_dates:
        # 找到 met_time 中与 d 最接近的那一天
        diffs = np.abs(time_arr - d)
        idx = int(diffs.argmin())
        mapped = time_arr[idx]
        mapping[d] = mapped
        time_index_map[mapped] = idx

    df["met_date"] = df["date_dt"].map(mapping)

    return df, time_arr, time_index_map


# ========= 主函数 =========
def extract_met_features() -> Path:
    """
    主入口：生成 met_features.parquet
    """
    # 1) 读取样本
    df = _read_samples()
    if df.empty:
        raise RuntimeError("样本为空，无法提取气象特征。")

    # 2) 打开 Zarr 变量
    print("[open] load meteorology Zarr ...")
    da_tmax = _load_zarr_var("maxtmp")  # 日最高气温（prep 中已转成 degC）
    da_tmin = _load_zarr_var("mintmp")  # 日最低气温
    da_pre  = _load_zarr_var("pre")     # 降水量（mm）
    da_win  = _load_zarr_var("win")     # 风速（m/s）
    da_rhu  = _load_zarr_var("rhu")     # 相对湿度（%）
    da_sst  = _load_zarr_var("sst")     # 日照时长 / 辐射

    # 3) 时间 & 空间坐标
    time_vals = da_pre[DIM_TIME].values  # np.datetime64[ns]
    lat_arr = da_pre[DIM_LAT].values     # 1D
    lon_arr = da_pre[DIM_LON].values     # 1D

    # 4) 将样本日期对齐到 met 时间轴
    df, time_arr, time_index_map = _align_dates_to_met(df, time_vals)

    # 按 met_date 分组，后面按时间轴顺序扫描时，只在“有样本的日子”做插值
    groups: Dict[np.datetime64, pd.DataFrame] = {
        k: v.copy()
        for k, v in df.groupby("met_date")
    }
    print(f"[group] unique sample days on met axis: {len(groups):,}")

    # 5) 准备“滑动窗口”的缓存：只在格点上做累积，不保存整个 3D 数据
    #    预先拿一个格点的 shape
    ny = lat_arr.size
    nx = lon_arr.size

    # pre 的 3/7/30 日累积（格点）
    sum_pre3  = np.zeros((ny, nx), dtype=np.float32)
    sum_pre7  = np.zeros((ny, nx), dtype=np.float32)
    sum_pre30 = np.zeros((ny, nx), dtype=np.float32)
    q_pre3: deque = deque(maxlen=3)
    q_pre7: deque = deque(maxlen=7)
    q_pre30: deque = deque(maxlen=30)

    # ate / rhu / win 的 3 日均值（格点）
    sum_ate3 = np.zeros((ny, nx), dtype=np.float32)
    sum_rhu3 = np.zeros((ny, nx), dtype=np.float32)
    sum_win3 = np.zeros((ny, nx), dtype=np.float32)
    q_ate3: deque = deque(maxlen=3)
    q_rhu3: deque = deque(maxlen=3)
    q_win3: deque = deque(maxlen=3)

    # Raindate：记录“距最近有雨日”的天数（> RAINDATE_MAX_DAYS 表示很久没下雨）
    last_rain = np.full((ny, nx), RAINDATE_MAX_DAYS + 1, dtype=np.int16)

    # 6) 顺着气象时间轴扫描，一旦遇到有样本的日期就做插值
    results = []
    total_days = time_arr.size
    print(f"[loop] scan meteorology time axis, total days = {total_days:,}")

    for t_idx, t_val in enumerate(time_arr):
        # 当前这一天的格点场（注意：全部是 2D 数组）
        pre_grid = da_pre.isel({DIM_TIME: t_idx}).values.astype(np.float32)
        tmax_grid = da_tmax.isel({DIM_TIME: t_idx}).values.astype(np.float32)
        tmin_grid = da_tmin.isel({DIM_TIME: t_idx}).values.astype(np.float32)
        rhu_grid  = da_rhu.isel({DIM_TIME: t_idx}).values.astype(np.float32)
        win_grid  = da_win.isel({DIM_TIME: t_idx}).values.astype(np.float32)
        sst_grid  = da_sst.isel({DIM_TIME: t_idx}).values.astype(np.float32)

        # --- Pre: 3/7/30 日累积（含当天） ---
        # 3 日窗口
        sum_pre3 += pre_grid
        q_pre3.append(pre_grid)
        if len(q_pre3) > 3:
            oldest = q_pre3.popleft()
            sum_pre3 -= oldest

        # 7 日窗口
        sum_pre7 += pre_grid
        q_pre7.append(pre_grid)
        if len(q_pre7) > 7:
            oldest = q_pre7.popleft()
            sum_pre7 -= oldest

        # 30 日窗口
        sum_pre30 += pre_grid
        q_pre30.append(pre_grid)
        if len(q_pre30) > 30:
            oldest = q_pre30.popleft()
            sum_pre30 -= oldest

        # --- Raindate：向前最多 30 天的最近有雨日 ---
        # 先整体 +1（最多 RAINDATE_MAX_DAYS+1），再对“有雨格点”置 0
        last_rain = np.where(last_rain < RAINDATE_MAX_DAYS + 1, last_rain + 1, last_rain)
        rain_flag = pre_grid > RAINDATE_THRESHOLD
        last_rain = np.where(rain_flag, 0, last_rain)
        raindate_grid = np.minimum(last_rain, RAINDATE_MAX_DAYS).astype(np.int16)

        # --- Ate / Arh / Aws 的 3 日均 ---
        ate_today = (tmax_grid + tmin_grid) / 2.0

        # Ate 3 日均
        sum_ate3 += ate_today
        q_ate3.append(ate_today)
        if len(q_ate3) > 3:
            oldest = q_ate3.popleft()
            sum_ate3 -= oldest
        ate3_grid = (sum_ate3 / float(len(q_ate3))).astype(np.float32)

        # Arh 3 日均
        sum_rhu3 += rhu_grid
        q_rhu3.append(rhu_grid)
        if len(q_rhu3) > 3:
            oldest = q_rhu3.popleft()
            sum_rhu3 -= oldest
        rhu3_grid = (sum_rhu3 / float(len(q_rhu3))).astype(np.float32)

        # Aws 3 日均
        sum_win3 += win_grid
        q_win3.append(win_grid)
        if len(q_win3) > 3:
            oldest = q_win3.popleft()
            sum_win3 -= oldest
        win3_grid = (sum_win3 / float(len(q_win3))).astype(np.float32)

        # --- 如果这一天 met_date 上没有样本，就继续下一天 ---
        if t_val not in groups:
            if (t_idx + 1) % 365 == 0 or (t_idx + 1) == total_days:
                print(f"[loop] processed {t_idx + 1:6d}/{total_days:6d} days, "
                      f"collected rows={sum(len(x) for x in results):,}")
            continue

        sub = groups[t_val]
        lons = sub["lon"].values
        lats = sub["lat"].values

        # 做一次二维最近邻插值（所有样本一起）
        Mate  = _interp2d_nearest(tmax_grid, lat_arr, lon_arr, lats, lons)
        Mite  = _interp2d_nearest(tmin_grid, lat_arr, lon_arr, lats, lons)
        Ate   = _interp2d_nearest(ate_today,  lat_arr, lon_arr, lats, lons)
        Ate3m = _interp2d_nearest(ate3_grid,  lat_arr, lon_arr, lats, lons)

        Arh   = _interp2d_nearest(rhu_grid,   lat_arr, lon_arr, lats, lons)
        Arh3m = _interp2d_nearest(rhu3_grid,  lat_arr, lon_arr, lats, lons)

        Aws   = _interp2d_nearest(win_grid,   lat_arr, lon_arr, lats, lons)
        Aws3m = _interp2d_nearest(win3_grid,  lat_arr, lon_arr, lats, lons)

        Suh   = _interp2d_nearest(sst_grid,   lat_arr, lon_arr, lats, lons)

        Pre1  = _interp2d_nearest(pre_grid,      lat_arr, lon_arr, lats, lons)
        Pre3  = _interp2d_nearest(sum_pre3,      lat_arr, lon_arr, lats, lons)
        Pre7  = _interp2d_nearest(sum_pre7,      lat_arr, lon_arr, lats, lons)
        Pre30 = _interp2d_nearest(sum_pre30,     lat_arr, lon_arr, lats, lons)

        RainD = _interp2d_nearest(
            raindate_grid.astype(float),
            lat_arr,
            lon_arr,
            lats,
            lons,
        ).astype(np.int16)

        day_str = pd.to_datetime(t_val).strftime("%Y-%m-%d")

        part = pd.DataFrame({
            "sample_id": sub["sample_id"].values,
            "date_key":  [day_str] * len(sub),
            "Mate": Mate,
            "Mite": Mite,
            "Ate": Ate,
            "Ate_3d_mean": Ate3m,
            "Arh": Arh,
            "Arh_3d_mean": Arh3m,
            "Aws": Aws,
            "Aws_3d_mean": Aws3m,
            "Suh": Suh,
            "Pre": Pre1,
            "Pre_3d_sum": Pre3,
            "Pre_7d_sum": Pre7,
            "Pre_30d_sum": Pre30,
            "Raindate": RainD,
        })

        results.append(part)

        if (t_idx + 1) % 365 == 0 or (t_idx + 1) == total_days:
            print(f"[loop] processed {t_idx + 1:6d}/{total_days:6d} days, "
                  f"collected rows={sum(len(x) for x in results):,}")

    if not results:
        raise RuntimeError("在气象时间轴上没有匹配到任何样本，请检查日期对齐逻辑。")

    met_df = pd.concat(results, ignore_index=True)
    print(f"[merge] met feature rows = {len(met_df):,}")

    # 安全检查：行数是否与样本一致
    if len(met_df) != len(df):
        print(f"[warn] met_features 行数({len(met_df):,}) 与样本行数({len(df):,}) 不一致。"
              f"（可能有日期对齐导致的重复/合并，后续 join 时请注意）")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    met_df.to_parquet(OUT_PATH, index=False)
    print(f"[OK] met_features -> {OUT_PATH}  rows={len(met_df):,}  cols={len(met_df.columns)}")
    return OUT_PATH


if __name__ == "__main__":
    extract_met_features()
