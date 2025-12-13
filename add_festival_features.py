# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"G:\fire\outputs")
SAMPLES = BASE / "samples_all.parquet"
OUT = BASE / "holiday_features.parquet"
OFFICIAL_JSON = BASE.parent / "staging" / "holiday_CN.json"
OFFICIAL_JSON_FALLBACK = Path(r"E:\fire\staging\holiday_CN.json")


def _weekday_ohe(d: pd.Timestamp) -> np.ndarray:
    w = int(d.weekday())  # Monday=0
    vec = np.zeros(7, dtype=np.int8)
    vec[w] = 1
    return vec


def _is_china_holiday(d: pd.Timestamp) -> int:
    """
    极简版中国节假日近似：
    - 固定：1-1, 5-1, 10-1~10-3
    - 近似：清明/端午/中秋，按公历附近几天
    不考虑调休，但对 2001-2020 的大尺度行为建模足够。
    """
    md = (d.month, d.day)
    fixed = {(1, 1), (5, 1), (10, 1), (10, 2), (10, 3)}
    approx = {
        (4, 4), (4, 5), (4, 6),        # 清明
        (6, 7), (6, 8), (6, 9),        # 端午附近
        (9, 20), (9, 21), (9, 22), (9, 23),  # 中秋附近
    }
    return int(md in fixed or md in approx)


def _is_fire_season(d: pd.Timestamp) -> int:
    """
    森林火险高发季（广东可近似）：1-4 月、10-12 月。
    """
    return int(d.month in {1, 2, 3, 4, 10, 11, 12})


def _is_summer_vacation(d: pd.Timestamp) -> int:
    """暑假：7-8 月。"""
    return int(d.month in {7, 8})


def _is_tourism_peak(d: pd.Timestamp) -> int:
    """
    旅游/人员流动高峰：
      - 春节前后（1 月下旬 ~ 2 月中旬）
      - 五一黄金周（5-1~5-7）
      - 十一黄金周（10-1~10-7）
    """
    m, day = d.month, d.day
    if (m == 1 and day >= 15) or (m == 2 and day <= 15):
        return 1
    if m == 5 and 1 <= day <= 7:
        return 1
    if m == 10 and 1 <= day <= 7:
        return 1
    return 0


def add_festival_features() -> Path:
    df = pd.read_parquet(SAMPLES)
    df["date_key"] = pd.to_datetime(df["date_key"], errors="coerce")
    df = df.dropna(subset=["date_key"]).reset_index(drop=True)

    dates = df["date_key"]

    # weekday one-hot
    w7 = np.stack([_weekday_ohe(x) for x in dates], axis=0)
    feat = pd.DataFrame(
        w7,
        columns=[f"wd_{i}" for i in range(7)],
        index=df.index,
    )

    # 周末
    feat["is_weekend"] = (dates.dt.weekday >= 5).astype(np.int8)

    # 节假日 / 行为 proxy
    feat["is_holiday"] = dates.apply(_is_china_holiday).astype(np.int8)
    feat["is_fire_season"] = dates.apply(_is_fire_season).astype(np.int8)
    feat["is_summer_vacation"] = dates.apply(_is_summer_vacation).astype(np.int8)
    feat["is_tourism_peak"] = dates.apply(_is_tourism_peak).astype(np.int8)

    # 时间衍生特征
    feat["month"] = dates.dt.month.astype(np.int8)
    feat["dayofyear"] = dates.dt.dayofyear.astype(np.int16)

    feat["is_official_holiday"] = 0
    feat["is_transfer_workday"] = 0
    try:
        json_path = OFFICIAL_JSON if OFFICIAL_JSON.exists() else OFFICIAL_JSON_FALLBACK
        if json_path.exists():
            data = pd.read_json(json_path)
            data = data.explode("dates").reset_index(drop=True)
            if {"year", "region", "dates"}.issubset(set(data.columns)):
                rows = []
                for _, r in data.iterrows():
                    year = int(r["year"])
                    items = r["dates"]
                    if isinstance(items, list):
                        for item in items:
                            dt = item.get("date", None)
                            tp = item.get("type", "")
                            if dt:
                                rows.append({"date": dt, "type": tp})
                if rows:
                    cal = pd.DataFrame(rows)
                    cal["date"] = pd.to_datetime(cal["date"], errors="coerce")
                    cal = cal.dropna(subset=["date"])
                    cal["key"] = cal["date"].dt.strftime("%Y-%m-%d")
                    feat["key"] = dates.dt.strftime("%Y-%m-%d")
                    m = feat[["key"]].merge(cal[["key", "type"]], on="key", how="left")
                    feat["is_official_holiday"] = (m["type"] == "public_holiday").fillna(False).astype(np.int8)
                    feat["is_transfer_workday"] = (m["type"] == "transfer_workday").fillna(False).astype(np.int8)
                    feat = feat.drop(columns=["key"])
    except Exception:
        pass

    feat["sample_id"] = df["sample_id"].astype(str)
    feat["date_key"] = dates.dt.strftime("%Y-%m-%d")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(OUT, index=False, engine="pyarrow")
    print(f"[OK] holiday_features -> {OUT} shape={feat.shape}")
    return OUT


if __name__ == "__main__":
    add_festival_features()
