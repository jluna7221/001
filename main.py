# -*- coding: utf-8 -*-
from pathlib import Path
import sys

from make_samples_all import make_samples_all_for_iter
from extract_met_features_cdmet import extract_met_features
from add_festival_features import add_festival_features
from build_static_features import build_static_features, load_and_join, attach_bin_month_year
from xgb_train_optuna import train_xgb_model
from gen_negative_samples import gen_negative_samples_for_iter
from eval_fullgrid_env import ensure_eval_fullgrid_env  # 新：森林掩膜全格网

BASE = Path(r"G:\fire\outputs")

def _ask_int(prompt: str, default: int = None) -> int:
    s = input(prompt).strip()
    if s == "" and default is not None:
        return default
    try:
        return int(s)
    except Exception:
        return default

def run_one_iter(iter_idx: int, grid_deg: float = 0.1, start_step: int = 1, eval_env_path: Path = None):
    print(f"\n========== 开始运行第 {iter_idx} 轮（从第 {start_step} 步开始） ==========")

    # 1) 样本（第 1 轮按比例；第 2 轮起不再按比例）
    if start_step <= 1:
        print("\n[Step 1] 生成/合并正负样本 samples_all ...")
        ratio = 5.0 if iter_idx == 1 else None
        samples_path = make_samples_all_for_iter(iter_idx=iter_idx, pos_to_neg_ratio=ratio)
        print(f"[Step 1] 完成: {samples_path}")
    else:
        print("\n[Step 1] 跳过：使用已有 G:\\fire\\outputs\\samples_all.parquet")

    # 2) 气象
    if start_step <= 2:
        print("\n[Step 2] 提取气象特征 met_features ...")
        met_path = extract_met_features()
        print(f"[Step 2] 完成: {met_path}")
    else:
        print("\n[Step 2] 跳过：使用已有 G:\\fire\\outputs\\met_features.parquet")

    # 3) 节日/行为
    if start_step <= 3:
        print("\n[Step 3] 生成节假日/高发季/旅游季等特征 ...")
        holiday_path = add_festival_features()
        print(f"[Step 3] 完成: {holiday_path}")
    else:
        print("\n[Step 3] 跳过：使用已有 G:\\fire\\outputs\\holiday_features.parquet")

    # 4) 静态
    if start_step <= 4:
        print("\n[Step 4] 生成静态特征 static_features ...")
        static_path = build_static_features()
        print(f"[Step 4] 完成: {static_path}")
    else:
        print("\n[Step 4] 跳过：使用已有 G:\\fire\\outputs\\static_features.parquet")

    # 5) 合并
    print("\n[Step 5] load_and_join -> 合并 met/static/holiday ...")
    df_all = load_and_join()
    df_all = attach_bin_month_year(df_all, grid_deg=grid_deg)
    print(f"[Step 5] 合并后样本数={len(df_all):,}，列数={len(df_all.columns)}")

    # 6) 训练 + 反馈（双子模型 + 线性组合 & 全格网评估）
    if start_step <= 6:
        print("\n[Step 6] 训练 XGBoost（F/H 双子模型）+ 导出 feedback ...")
        feedback_path, model_path = train_xgb_model(df_all, iter_idx=iter_idx, grid_deg=grid_deg)
        print(f"[Step 6] 完成: 模型={model_path}, 反馈={feedback_path}")
    else:
        feedback_path = BASE / f"feedback_it{iter_idx}.json"
        model_path = BASE / f"xgb_model_flamm_it{iter_idx}.json"
        print("\n[Step 6] 跳过：使用已有反馈/模型。")

    # 7) 基于 feedback 强化采样下一轮负样本（自适应配额 & 数量由 RL 自定）
    if start_step <= 7:
        print("\n[Step 7] 基于 feedback 生成下一轮负样本 ...")
        next_iter = iter_idx + 1
        neg_path = gen_negative_samples_for_iter(iter_idx=next_iter, feedback_path=feedback_path, grid_deg=grid_deg)
        print(f"[Step 7] 完成: 第 {next_iter} 轮负样本 -> {neg_path}")
    else:
        print("\n[Step 7] 跳过。")

    print(f"========== 第 {iter_idx} 轮 完成 ==========\n")

def main():
    print("========== 双训练闭环 主控（v4：森林掩膜全格网 + 双子模型 + 自适应RL） ==========")
    print("1. 从第 1 轮开始，依次跑若干轮（推荐 2 轮）")
    print("2. 只跑指定某一轮（调试/续跑）")
    print("3. 退出")

    choice = _ask_int("请选择模式 [1/2/3]，回车默认 1：", default=1)
    if choice == 3:
        print("退出。"); sys.exit(0)

    grid_deg = 0.1
    try:
        gd = input("请输入栅格分辨率 grid_deg（默认 0.1）：").strip()
        if gd:
            grid_deg = float(gd)
    except Exception:
        grid_deg = 0.1

    # 先确保“森林掩膜全格网”评估环境 einmal 构建
    print("\n[EvalEnv] 构建/检查全格网评估环境（仅首次计算，后续复用）...")
    eval_env_path = ensure_eval_fullgrid_env(grid_deg=grid_deg)
    print(f"[EvalEnv] 使用评估环境文件: {eval_env_path}")

    if choice == 1:
        start_iter = _ask_int("输入起始轮次（默认 1）：", default=1)
        end_iter = _ask_int("输入结束轮次（默认 2）：", default=2)
        start_step_first = _ask_int("首轮从第几步开始？1~7，回车默认 1：", default=1)
        if not (1 <= (start_step_first or 1) <= 7): start_step_first = 1
        for it in range(start_iter, end_iter + 1):
            ss = start_step_first if it == start_iter else 1
            run_one_iter(iter_idx=it, grid_deg=grid_deg, start_step=ss, eval_env_path=eval_env_path)
    elif choice == 2:
        it = _ask_int("输入要运行的轮次（默认 1）：", default=1)
        start_step = _ask_int("该轮从第几步开始？1~7，回车默认 1：", default=1)
        if not (1 <= (start_step or 1) <= 7): start_step = 1
        run_one_iter(iter_idx=it, grid_deg=grid_deg, start_step=start_step, eval_env_path=eval_env_path)
    else:
        print("无效选择，退出。")

if __name__ == "__main__":
    main()
