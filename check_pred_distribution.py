# -*- coding: utf-8 -*-
"""
check_pred_distribution.py

功能：
1. 读取生成的 2020 年预测 TIF 文件。
2. 统计预测概率值的分布（最大值、平均值、95分位数等）。
3. 告诉你：到底多少分才算“高风险”。
"""

from pathlib import Path
import numpy as np
import rasterio
from tqdm import tqdm
import matplotlib.pyplot as plt

TIF_DIR = Path(r"H:\fire\outputs\tif_2020")


def main():
    tifs = list(TIF_DIR.glob("risk_*.tif"))
    if not tifs:
        print("❌ 未找到 TIF 文件，请先运行预测脚本。")
        return

    print(f"正在扫描 {len(tifs)} 个预测文件，统计概率分布...")

    all_probs = []

    # 随机抽样 50 天的数据进行统计，避免内存爆炸
    sample_tifs = tifs[::7]  # 每周抽一天

    for p in tqdm(sample_tifs):
        with rasterio.open(p) as src:
            data = src.read(1)
            # 过滤掉 NaN (背景)
            valid_data = data[~np.isnan(data)]
            # 再次随机降采样，减少数据量
            if len(valid_data) > 10000:
                valid_data = np.random.choice(valid_data, 10000, replace=False)
            all_probs.append(valid_data)

    # 合并
    merged = np.concatenate(all_probs)

    print("\n" + "=" * 40)
    print("📊 预测概率值分布统计报告")
    print("=" * 40)
    print(f"最小值 (Min): {merged.min():.4f}")
    print(f"平均值 (Mean): {merged.mean():.4f}")
    print(f"中位数 (Median): {np.median(merged):.4f}")
    print(f"最大值 (Max): {merged.max():.4f}")
    print("-" * 40)
    print(f"Top 10% 阈值: {np.percentile(merged, 90):.4f}")
    print(f"Top 5%  阈值: {np.percentile(merged, 95):.4f}")
    print(f"Top 1%  阈值: {np.percentile(merged, 99):.4f}")
    print("=" * 40)

    # 画直方图
    plt.figure(figsize=(10, 6))
    plt.hist(merged, bins=100, log=True, color='skyblue', edgecolor='black')
    plt.title("Prediction Probability Distribution (Log Scale)")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency (Log)")
    plt.axvline(np.percentile(merged, 95), color='r', linestyle='dashed', linewidth=2, label='Top 5% Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()