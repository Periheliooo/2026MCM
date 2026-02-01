import numpy as np
import pandas as pd
import os
from problem_4_model import MoonColonyModel

def run_test():
    print("="*60)
    print("MCM Problem 4 Model - Unit Test")
    print("="*60)

    # 1. 实例化模型
    # 注意：确保 harbor.xlsx (或csv) 在 data/ 目录下
    try:
        model = MoonColonyModel('harbor.xlsx') 
        print(f"[Success] 模型初始化成功。")
        print(f" - 基地数量: {model.num_bases}")
        print(f" - 数据路径: {model.data_path}")
    except Exception as e:
        print(f"[Error] 模型初始化失败: {e}")
        return

    # 2. 验证火箭成本公式
    print("-" * 30)
    print("验证火箭成本衰减公式 (Cost per Ton):")
    years_to_check = [2050, 2060, 2100, 2150]
    for y in years_to_check:
        cost = model.get_rocket_cost_per_ton(y)
        print(f"  Year {y}: ${cost:,.2f} / ton")

    # 3. 构造测试输入 (Chromosome)
    # 模拟一个 100 年的规划
    # 策略：前50年仅用太空电梯，后50年每年增加 100万吨火箭运输
    n_years = 100
    
    # 太空电梯：满负荷运行
    x_se = np.full(n_years, model.SE_CAPACITY_YEAR)
    
    # 火箭：前50年为0，后50年为 1,000,000 吨 (注意：这可能超过全球运力，用于测试约束)
    x_rocket = np.zeros(n_years)
    x_rocket[50:] = 1_000_000 

    # 4. 运行评估
    print("-" * 30)
    print(f"运行评估 (Simulating {n_years} years)...")
    objs, constrs = model.evaluate(x_se, x_rocket)

    # 5. 输出结果
    print("评估结果:")
    print(f"  [目标1] 总成本 (NPV):     ${objs[0]:.4f} Billion")
    print(f"  [目标2] 完工时间:         {objs[1]:.2f} Years")
    print(f"  [目标3] 环境影响指数:     {objs[2]:.4f} Units")
    print("-" * 15)
    print(f"  [约束1] 需求违背量:       {constrs[0]:,.0f} tons")
    print(f"  [约束2] 运力违背量:       {constrs[1]:,.0f} tons (Expected > 0 if over-scheduled)")
    print(f"  [约束3] 代际公平违背:     {constrs[2]:.4f}")

    # 6. 简单的贪心策略验证
    print("-" * 30)
    print("基地数据概览 (按成本排序):")
    print(model.sorted_harbors[['基地名称', '2050年单次成本系数', 'Geo_Coeff', 'Lat_Value']].head())

if __name__ == "__main__":
    run_test()