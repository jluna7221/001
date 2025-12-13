# -*- coding: utf-8 -*-
"""
build_static_features.py (v5.6 - 接口修复版)

修订内容：
1. 恢复 _init_lc_year2path 等函数名（作为 _init_paths 的别名），修复 AttributeError。
2. 保持 v5.5 的 NoData 过滤和 NDVI 排序修复逻辑。
"""

from pathlib import Path
import datetime as dt
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

try:
    import rasterio
except Exception:
    rasterio = None

try:
    import xarray as xr
except Exception:
    xr = None

BASE = Path(r"G:\fire\outputs")
SAMPLES = BASE / "samples_all.parquet"
STATIC_OUT = BASE / "static_features.parquet"
MET_PATH = BASE / "met_features.parquet"
HOLI_PATH = BASE / "holiday_features.parquet"

DATE_MIN = dt.date(2001, 1, 1)
DATE_MAX = dt.date(2020, 12, 31)

DEM_TIF = Path(r"G:\DEM\DEM.tif")
SLOPE_TIF = Path(r"G:\DEM\坡度.tif")
ASPECT_TIF = Path(r"G:\DEM\坡向.tif")
DIST_RIVER_TIF = Path(r"G:\广东欧氏距离\距离河流距离.tif")
DIST_ROAD_TIF = Path(r"G:\广东欧氏距离\距离道路距离.tif")
DIST_VILLAGE_TIF = Path(r"G:\广东欧氏距离\距离村庄距离.tif")
NDVI_DIR = Path(r"G:\NDVI")
POP_DIR = Path(r"G:\广东省人口密度")
GDP_DIR = Path(r"G:\GDP")
LC_DIR = Path(r"G:\土地类型\1")

DIM_TIME, DIM_LAT, DIM_LON = "time", "lat", "lon"


def sample_tif_points(tif_path: Path, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    vals = np.full(len(lons), np.nan, dtype=np.float32)
    try:
        if (rasterio is None) or (not tif_path.exists()): return vals
        with rasterio.open(str(tif_path)) as ds:
            nodata = ds.nodata
            band = ds.read(1)
            for i, (lo, la) in enumerate(zip(lons, lats)):
                try:
                    r, c = ds.index(float(lo), float(la))
                    if 0 <= r < ds.height and 0 <= c < ds.width:
                        v = band[r, c]
                        # [Fix] 过滤 NoData
                        if nodata is not None and np.isclose(v, nodata):
                            vals[i] = np.nan
                        elif not np.isfinite(v):
                            vals[i] = np.nan
                        else:
                            vals[i] = np.float32(v)
                except:
                    pass
    except:
        return vals
    return vals


_NDVI_INDEX_BUILT, _NDVI_DS_CACHE = False, {}
_NDVI_TIME_ARR, _NDVI_FILE_INDEX, _NDVI_TIME_INDEX = None, None, None
_NDVI_LAT_ARR, _NDVI_LON_ARR, _NDVI_FILES = None, None, []


def _nearest_index_1d(coord: np.ndarray, values: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(coord, values)
    idx = np.clip(idx, 1, len(coord) - 1)
    left, right = coord[idx - 1], coord[idx]
    choose_left = (values - left) <= (right - values)
    return idx - choose_left.astype(np.int64)


def _interp2d_nearest(grid, lat_arr, lon_arr, lats, lons):
    iy = _nearest_index_1d(lat_arr, np.asarray(lats))
    ix = _nearest_index_1d(lon_arr, np.asarray(lons))
    return grid[iy, ix].astype(np.float32)


def _build_ndvi_index():
    global _NDVI_INDEX_BUILT, _NDVI_TIME_ARR, _NDVI_FILE_INDEX, _NDVI_TIME_INDEX
    global _NDVI_LAT_ARR, _NDVI_LON_ARR, _NDVI_FILES
    if _NDVI_INDEX_BUILT: return
    files = sorted(NDVI_DIR.glob("*.nc4"))
    time_list, file_list, idx_list = [], [], []
    lat_ref, lon_ref = None, None
    for file_idx, p in enumerate(files):
        ds = xr.open_dataset(str(p))
        if "ndvi" not in ds.data_vars: ds.close(); continue
        if lat_ref is None:
            # [Fix] 强制排序坐标
            lat_ref = np.sort(ds[DIM_LAT].values)
            lon_ref = np.sort(ds[DIM_LON].values)
        t_vals = ds[DIM_TIME].values.astype("datetime64[D]")
        for j, t in enumerate(t_vals):
            time_list.append(t)
            file_list.append(file_idx)
            idx_list.append(j)
        ds.close()

    order = np.argsort(np.array(time_list))
    _NDVI_TIME_ARR = np.array(time_list)[order]
    _NDVI_FILE_INDEX = np.array(file_list)[order]
    _NDVI_TIME_INDEX = np.array(idx_list)[order]
    _NDVI_LAT_ARR, _NDVI_LON_ARR, _NDVI_FILES = lat_ref, lon_ref, files
    _NDVI_INDEX_BUILT = True


def _get_ndvi_dataset(file_idx):
    if file_idx in _NDVI_DS_CACHE: return _NDVI_DS_CACHE[file_idx]
    ds = xr.open_dataset(str(_NDVI_FILES[file_idx])).sortby([DIM_LAT, DIM_LON])
    _NDVI_DS_CACHE[file_idx] = ds
    return ds


def sample_ndvi_points(lons, lats, dates):
    if not NDVI_DIR.exists(): return np.full(len(lons), np.nan, dtype=np.float32)
    _build_ndvi_index()
    dates_np = np.array([np.datetime64(pd.to_datetime(d).date()) for d in dates], dtype="datetime64[D]")
    vals = np.full(len(lons), np.nan, dtype=np.float32)
    unique_dates, inv_idx = np.unique(dates_np, return_inverse=True)
    for k, dval in enumerate(unique_dates):
        if dval < _NDVI_TIME_ARR[0] or dval > _NDVI_TIME_ARR[-1]: continue
        idx = np.searchsorted(_NDVI_TIME_ARR, dval)
        idx = idx if abs(dval - _NDVI_TIME_ARR[idx]) < abs(dval - _NDVI_TIME_ARR[idx - 1]) else idx - 1
        f_idx, t_idx = int(_NDVI_FILE_INDEX[idx]), int(_NDVI_TIME_INDEX[idx])
        ds = _get_ndvi_dataset(f_idx)
        ndvi_2d = ds["ndvi"].isel({DIM_TIME: t_idx}).values
        mask = (inv_idx == k)
        vals[mask] = _interp2d_nearest(ndvi_2d, _NDVI_LAT_ARR, _NDVI_LON_ARR, lats[mask], lons[mask])
    return vals


_POP_YEAR2PATH, _GDP_YEAR2PATH, _LC_YEAR2PATH = None, None, None


def _init_paths():
    """初始化所有年度TIF文件的路径映射"""
    global _POP_YEAR2PATH, _GDP_YEAR2PATH, _LC_YEAR2PATH
    build_map = lambda d: {int(re.search(r"(\d{4})", p.stem).group(1)): p for p in d.glob("*.tif") if
                           re.search(r"(\d{4})", p.stem)}
    if _POP_YEAR2PATH is None: _POP_YEAR2PATH = build_map(POP_DIR)
    if _GDP_YEAR2PATH is None: _GDP_YEAR2PATH = build_map(GDP_DIR)
    if _LC_YEAR2PATH is None: _LC_YEAR2PATH = build_map(LC_DIR)


# [Fix] 增加兼容接口，供外部模块调用
def _init_pop_year2path(): _init_paths()


def _init_gdp_year2path(): _init_paths()


def _init_lc_year2path(): _init_paths()


def sample_yearly_tif_points(years, lons, lats, mapping):
    vals = np.full(len(lons), np.nan, dtype=np.float32)
    unique_years = np.unique(years)
    sorted_map_years = sorted(mapping.keys())
    if not sorted_map_years: return vals
    for y in unique_years:
        best_y = min(sorted_map_years, key=lambda x: abs(x - y))
        mask = (years == y)
        vals[mask] = sample_tif_points(mapping[best_y], lons[mask], lats[mask])
    return vals


def build_static_features() -> Path:
    if not SAMPLES.exists(): raise FileNotFoundError(SAMPLES)
    df = pd.read_parquet(SAMPLES)
    df["date_key"] = pd.to_datetime(df["date_key"], errors="coerce")
    df = df.dropna(subset=["lon", "lat", "date_key"]).reset_index(drop=True)
    lons, lats = df["lon"].values, df["lat"].values
    dates, years = df["date_key"].dt.date.values, df["date_key"].dt.year.values

    _init_paths()
    dem = sample_tif_points(DEM_TIF, lons, lats)
    slope = sample_tif_points(SLOPE_TIF, lons, lats)
    aspect = sample_tif_points(ASPECT_TIF, lons, lats)
    rad = np.deg2rad(aspect)

    static = pd.DataFrame({
        "sample_id": df["sample_id"].astype(str).values,
        "date_key": df["date_key"].dt.strftime("%Y-%m-%d").values,
        "dem": dem, "slope": slope, "aspect": aspect,
        "aspect_sin": np.sin(rad), "aspect_cos": np.cos(rad),
        "dist_river": sample_tif_points(DIST_RIVER_TIF, lons, lats),
        "dist_road": sample_tif_points(DIST_ROAD_TIF, lons, lats),
        "dist_village": sample_tif_points(DIST_VILLAGE_TIF, lons, lats),
        "ndvi": sample_ndvi_points(lons, lats, dates),
        "pop_density": sample_yearly_tif_points(years, lons, lats, _POP_YEAR2PATH),
        "gdp_density": sample_yearly_tif_points(years, lons, lats, _GDP_YEAR2PATH),
        "lc_raw": sample_yearly_tif_points(years, lons, lats, _LC_YEAR2PATH)
    })
    STATIC_OUT.parent.mkdir(parents=True, exist_ok=True)
    static.to_parquet(STATIC_OUT, index=False)
    return STATIC_OUT


def load_and_join() -> pd.DataFrame:
    base = pd.read_parquet(SAMPLES)
    base["sample_id"] = base["sample_id"].astype(str)
    base["date_key"] = pd.to_datetime(base["date_key"], errors="coerce")
    for name, path in [("met", MET_PATH), ("static", STATIC_OUT), ("holiday", HOLI_PATH)]:
        if path.exists():
            feat = pd.read_parquet(path)
            feat["sample_id"] = feat["sample_id"].astype(str)
            feat["date_key"] = pd.to_datetime(feat["date_key"])
            feat = feat.drop_duplicates(subset=["sample_id", "date_key"])
            cols = [c for c in feat.columns if c not in ["sample_id", "date_key"]]
            overlap = set(base.columns) & set(cols)
            if overlap: feat = feat.rename(columns={c: f"{c}_{name}" for c in overlap})
            base = base.merge(feat, on=["sample_id", "date_key"], how="left")
    return base


def attach_bin_month_year(df, grid_deg=0.1):
    d = df.copy()
    d["date_key"] = pd.to_datetime(d["date_key"])
    d["year"], d["month"] = d["date_key"].dt.year, d["date_key"].dt.month
    i = np.floor((d["lat"] + 90) / grid_deg).astype(int)
    j = np.floor((d["lon"] + 180) / grid_deg).astype(int)
    d["bin"] = [f"{x}_{y}" for x, y in zip(i, j)]
    return d


def build_feature_matrix(df):
    d = df.copy()
    if "label" not in d.columns: raise ValueError("缺少 label")
    y = d["label"].astype(int).values
    X = d.drop(columns=["label", "sample_id", "date_key", "is_hard", "source", "date_dt", "bin", "year", "month", "lat",
                        "lon"], errors="ignore")
    X = X.select_dtypes(include=[np.number, bool])
    for c in X.columns:
        if is_bool_dtype(X[c]):
            X[c] = X[c].fillna(False).astype(bool)
        else:
            X[c] = X[c].fillna(X[c].median() if not pd.isna(X[c].median()) else 0).astype(float)
    return X, y, list(X.columns)