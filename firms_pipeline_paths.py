# -*- coding: utf-8 -*-
from pathlib import Path
import os
import pandas as pd, numpy as np, geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import BallTree
import rasterio
from datetime import datetime, timedelta, timezone

# ========= 1) 固定路径（按需改） =========
MODIS_DIR = r"H:\论文复现数据\MODIS C6.1"
SNPP_DIR  = r"H:\论文复现数据\SUOMI VIIRS C2"   # VIIRS S-NPP
J1_DIR    = r"H:\论文复现数据\J1 VIIRS C2"       # VIIRS NOAA-20
BOUNDARY  = r"H:\广东边界\广东省_省.shp"
LANDCOVER_DIR = r"H:\GD_tdlx\1"
OUTPUT_DIR= r"H:\fire\outputs"
STAGING_DIR= r"H:\fire\staging"

# ========= 2) 参数 =========
CRS_WGS84 = "EPSG:4326"
CONF_THRES = 30.0
DEDUP_RADIUS_KM = 1.0
FOREST_CLASSES = {1,2,3,4,5,6,7,8,9,10}
LC_FALLBACK = "nearest"

# ========= 3) 工具 =========
LOCAL_TZ = timezone(timedelta(hours=8))
CONF_MAP = {"low":20, "nominal":60, "high":90}

def utc_to_local_date(acq_date, acq_time):
    ds = str(acq_date).strip()
    if " " in ds: ds = ds.split()[0]
    ds = ds.replace("/", "-")
    d_utc = pd.to_datetime(ds, errors="coerce", utc=True)
    if pd.isna(d_utc):
        raise ValueError(f"Cannot parse acq_date: {acq_date!r}")
    ts = str(acq_time).strip()
    if ts.lower() in ("", "nan", "none"): hh=mm=0
    else:
        ts = ts.replace(":","").split(".")[0]
        ts = ts.zfill(4)[:4]
        hh, mm = int(ts[:2]), int(ts[2:4])
    d_utc = d_utc + pd.Timedelta(hours=hh, minutes=mm)
    return d_utc.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d")

def clip_p99_cap(s: pd.Series, p=0.99):
    s = pd.to_numeric(s, errors="coerce")
    return s.clip(upper=s.quantile(p))

def _read_dir(dir_path: Path, source_label: str) -> pd.DataFrame:
    """
    递归读取目录下全部 CSV/SHP；若只找到 ZIP 会自动解压到 STAGING_DIR/unzipped/<source>/ 再读。
    打印找到的文件数，便于排错。
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Path not found: {dir_path}")

    # 递归搜 CSV/SHP
    csvs = sorted(dir_path.rglob("*.csv")) + sorted(dir_path.rglob("*.CSV"))
    shps = sorted(dir_path.rglob("*.shp")) + sorted(dir_path.rglob("*.SHP"))

    # 若都没有，尝试 ZIP
    if not csvs and not shps:
        zips = sorted(dir_path.rglob("*.zip")) + sorted(dir_path.rglob("*.ZIP"))
        print(f"[{source_label}] no CSV/SHP, found ZIP:", len(zips))
        if zips:
            unzip_root = Path(STAGING_DIR)/"unzipped"/source_label
            unzip_root.mkdir(parents=True, exist_ok=True)
            import zipfile
            for zp in zips:
                with zipfile.ZipFile(zp, 'r') as zf:
                    sub = unzip_root / zp.stem
                    sub.mkdir(exist_ok=True)
                    zf.extractall(sub)
            csvs = sorted(unzip_root.rglob("*.csv")) + sorted(unzip_root.rglob("*.CSV"))
            shps = sorted(unzip_root.rglob("*.shp")) + sorted(unzip_root.rglob("*.SHP"))

    print(f"[{source_label}] found csv={len(csvs)}, shp={len(shps)} in {dir_path}")

    if not csvs and not shps:
        print(f"[{source_label}] WARNING: still no readable files. Check path or unzip.")
        return pd.DataFrame(columns=["lon","lat","acq_date","acq_time","satellite","instrument","confidence","frp","bright_t31","brightness","daynight","version","type","source"])

    dfs=[]

    # 读 CSV
    for f in csvs:
        try:
            df = pd.read_csv(f, encoding="utf-8", engine="python")
        except Exception:
            df = pd.read_csv(f, encoding="latin-1", engine="python")
        rename = {"latitude":"lat","LATITUDE":"lat","longitude":"lon","LONGITUDE":"lon",
                  "acq_date":"acq_date","ACQ_DATE":"acq_date","acq_time":"acq_time","ACQ_TIME":"acq_time",
                  "satellite":"satellite","SATELLITE":"satellite","instrument":"instrument","INSTRUMENT":"instrument",
                  "confidence":"confidence","CONFIDENCE":"confidence","frp":"frp","FRP":"frp",
                  "bright_t31":"bright_t31","BRIGHT_T31":"bright_t31","brightness":"brightness","BRIGHTNESS":"brightness",
                  "daynight":"daynight","DAYNIGHT":"daynight","version":"version","VERSION":"version","type":"type","TYPE":"type"}
        for k,v in list(rename.items()):
            if k in df.columns and v not in df.columns:
                df = df.rename(columns={k:v})
        if "lon" not in df.columns or "lat" not in df.columns:
            continue
        df["source"]=source_label
        keep = [c for c in ["lon","lat","acq_date","acq_time","satellite","instrument",
                            "confidence","frp","bright_t31","brightness","daynight","version","type","source"]
                if c in df.columns]
        dfs.append(df[keep].copy())

    # 读 SHP
    for f in shps:
        g = gpd.read_file(f)
        rename = {"latitude":"lat","LATITUDE":"lat","longitude":"lon","LONGITUDE":"lon",
                  "acq_date":"acq_date","ACQ_DATE":"acq_date","acq_time":"acq_time","ACQ_TIME":"acq_time",
                  "satellite":"satellite","SATELLITE":"satellite","instrument":"instrument","INSTRUMENT":"instrument",
                  "confidence":"confidence","CONFIDENCE":"confidence","frp":"frp","FRP":"frp",
                  "bright_t31":"bright_t31","BRIGHT_T31":"bright_t31","brightness":"brightness","BRIGHTNESS":"brightness",
                  "daynight":"daynight","DAYNIGHT":"daynight","version":"version","VERSION":"version","type":"type","TYPE":"type"}
        for k,v in list(rename.items()):
            if k in g.columns and v not in g.columns:
                g = g.rename(columns={k:v})
        if "lon" not in g.columns or "lat" not in g.columns:
            g = g.to_crs(CRS_WGS84)
            g["lon"] = g.geometry.x; g["lat"]=g.geometry.y
        g["source"]=source_label
        keep = [c for c in ["lon","lat","acq_date","acq_time","satellite","instrument",
                            "confidence","frp","bright_t31","brightness","daynight","version","type","source"]
                if c in g.columns]
        dfs.append(pd.DataFrame(g[keep]).copy())

    if not dfs:
        print(f"[{source_label}] WARNING: files found but no usable columns.")
        return pd.DataFrame(columns=["lon","lat","acq_date","acq_time","satellite","instrument","confidence","frp","bright_t31","brightness","daynight","version","type","source"])

    out = pd.concat(dfs, ignore_index=True)
    if out["confidence"].dtype==object:
        out["confidence"] = out["confidence"].str.lower().map(CONF_MAP).fillna(pd.to_numeric(out["confidence"], errors="coerce"))

    print(f"[{source_label}] loaded rows:", len(out))
    # 打印几列看样子
    print(out.head(2).to_string(index=False))
    return out

def clip_to_boundary(df: pd.DataFrame, boundary_path: Path) -> pd.DataFrame:
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=CRS_WGS84)
    bd = gpd.read_file(boundary_path).to_crs(CRS_WGS84)
    gclipped = gpd.overlay(gdf, bd[["geometry"]], how="intersection")
    out = pd.DataFrame(gclipped.drop(columns="geometry"))
    print("[clip] rows in:", len(df), "→ rows in GD:", len(out))
    print("[clip] by source:\n", out["source"].value_counts())
    return out

def forest_filter_by_year(df: pd.DataFrame, lc_dir: Path, forest_classes: set, fallback="nearest") -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["acq_date"].astype(str).str[:4].astype(int)
    years = sorted([int(p.stem) for p in Path(lc_dir).glob("*.tif") if p.stem.isdigit()])
    if not years: raise FileNotFoundError(f"No yearly landcover in {lc_dir}")
    keep=[]
    for yy, grp in df.groupby("year"):
        cand = Path(lc_dir)/f"{yy}.tif"
        if not cand.exists():
            if fallback=="error": raise FileNotFoundError(f"{cand} not found")
            near = min(years, key=lambda y: abs(y-yy))
            cand = Path(lc_dir)/f"{near}.tif"
        with rasterio.open(cand) as src:
            vals = [int(v[0]) for v in src.sample(list(zip(grp["lon"], grp["lat"])))]
        mask = np.isin(vals, list(forest_classes))
        keep.extend(grp.index[np.array(mask)].tolist())
    out = df.loc[keep].drop(columns="year")
    print("[forest] rows in:", len(df), "→ forest kept:", len(out))
    print("[forest] by source:\n", out["source"].value_counts())
    return out

def same_day_dedup(df: pd.DataFrame, radius_km: float, conf_thres: float) -> pd.DataFrame:
    df = df.copy()
    # 本地日键
    df["date_key"] = df.apply(lambda r: utc_to_local_date(r["acq_date"], r.get("acq_time","0000")), axis=1)

    # ---- 来源感知的质量过滤 ----
    df["conf_num"] = pd.to_numeric(df["confidence"], errors="coerce")
    is_modis = df["source"].astype(str).str.contains("MODIS", case=False, na=False)
    is_viirs = df["source"].astype(str).str.contains("VIIRS", case=False, na=False)

    # MODIS：按置信度阈值
    mask_modis = is_modis & (df["conf_num"] >= conf_thres)

    # VIIRS：不看confidence；给个温和的 FRP 下限（可按需调整/去掉）
    # 常用门槛：>= 1.0 或 0.5；你也可以设为 0 代表不做FRP过滤
    frp_min_viirs = 1.0
    mask_viirs = is_viirs & (pd.to_numeric(df["frp"], errors="coerce") >= frp_min_viirs)

    # 其他未知来源（极少见）：全放过
    mask_other = ~(is_modis | is_viirs)

    df = df[mask_modis | mask_viirs | mask_other].copy()

    # FRP 按来源做 p99 截顶，减小极端值影响
    df["frp"] = df.groupby("source")["frp"].transform(lambda s: clip_p99_cap(s, 0.99))

    # ---- 同日 1km 去重（优先 conf，其次 frp）----
    out_idx=[]
    for d, grp in df.groupby("date_key", sort=False):
        coords = np.deg2rad(grp[["lat","lon"]].values)
        tree = BallTree(coords, metric="haversine")
        R = radius_km / 6371.0088
        idx = grp.index.to_numpy()
        # 排序：confidence优先（NaN→-inf），再看frp
        conf_order = pd.to_numeric(grp["conf_num"], errors="coerce").fillna(-1e9).values
        frp_order  = pd.to_numeric(grp["frp"], errors="coerce").fillna(-1e9).values
        order = np.lexsort((-frp_order, -conf_order))
        used = np.zeros(len(grp), dtype=bool)
        for oi in order:
            if used[oi]: continue
            nn = tree.query_radius(coords[oi].reshape(1,-1), r=R)[0]
            used[nn]=True
            out_idx.append(idx[oi])

    out = df.loc[out_idx].reset_index(drop=True)
    out["sample_id"] = [f"pos_{i}" for i in range(len(out))]
    # 日志
    print("[dedup] rows in:", len(df), "→ dedup out:", len(out))
    print("[dedup] by source:\n", out["source"].value_counts())
    return out

def main():
    OUT = Path(OUTPUT_DIR); OUT.mkdir(parents=True, exist_ok=True)
    ST = Path(STAGING_DIR); ST.mkdir(parents=True, exist_ok=True)

    modis = _read_dir(Path(MODIS_DIR), "MODIS_C61")
    snpp  = _read_dir(Path(SNPP_DIR),  "VIIRS_SNPP_C2")
    j1    = _read_dir(Path(J1_DIR),    "VIIRS_NOAA20_C2")

    # 分源计数
    print("[read] counts:", len(modis), len(snpp), len(j1))

    union = pd.concat([modis,snpp,j1], ignore_index=True)
    union.to_parquet(Path(ST)/"firms_union_raw.parquet", index=False)
    print("[union] total:", len(union), "by source:\n", union["source"].value_counts())

    gd = clip_to_boundary(union, Path(BOUNDARY))
    gd.to_parquet(Path(ST)/"firms_gd.parquet", index=False)

    forest = forest_filter_by_year(gd, Path(LANDCOVER_DIR), FOREST_CLASSES, fallback=LC_FALLBACK)
    forest.to_parquet(Path(ST)/"firms_gd_forest.parquet", index=False)

    master = same_day_dedup(forest, DEDUP_RADIUS_KM, CONF_THRES)
    cols = ["sample_id","lon","lat","date_key","confidence","frp","source","satellite","instrument","daynight","version","type"]
    cols = [c for c in cols if c in master.columns]
    master[cols].to_parquet(Path(OUT)/"firms_master.parquet", index=False)

    print("[OK] 输出：", Path(OUT)/"firms_master.parquet")
    print("总数：", len(master))
    print("来源统计：\n", master["source"].value_counts())
    print("年度统计（前若干）：\n", master.groupby(master["date_key"].str[:4]).size().sort_index().head())

if __name__ == "__main__":
    main()
