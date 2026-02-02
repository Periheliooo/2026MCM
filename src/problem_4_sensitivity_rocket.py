import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from problem_4_model import MoonColonyModel
from problem_4_solver import MoonOptimizationProblem, HeuristicSampling

def run_rocket_cost_sensitivity_advanced():
    # 稍微缩小范围，聚焦于变化的临界点
    multipliers = [0.8, 1.0, 1.2, 1.5]
    
    results = {}
    
    N_YEARS = 120
    POP_SIZE = 80
    N_GEN = 150 # 如果跑得慢可以改成 100

    print(f"--- 启动改进版敏感性分析 (结构性变化) ---")

    for mult in multipliers:
        print(f"\n[Scenario] Cost Multiplier: {mult}x")
        
        # 1. 初始化与参数修改
        model = MoonColonyModel('harbor.xlsx')
        model.rocket_cost_base *= mult
        model.rocket_cost_A *= mult
        
        # 2. 求解
        problem = MoonOptimizationProblem(model, n_years=N_YEARS)
        algorithm = NSGA2(
            pop_size=POP_SIZE, n_offsprings=40,
            sampling=HeuristicSampling(model, N_YEARS),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=0.01, eta=20), eliminate_duplicates=True
        )
        termination = get_termination("n_gen", N_GEN)
        
        res = minimize(problem, algorithm, termination, seed=42, verbose=False)
        
        if res.F is not None:
            # 简单过滤
            mask = (res.CV.flatten() <= 1e-3)
            if not np.any(mask): mask = np.full(len(res.F), True) # 兜底
            
            F_final = res.F[mask]
            X_final = res.X[mask]
            
            # --- 关键：计算物理指标 ---
            # X结构: [SE_0..SE_N, Rocket_0..Rocket_N]
            se_flows = X_final[:, :N_YEARS]
            rocket_flows = X_final[:, N_YEARS:]
            
            total_se = np.sum(se_flows, axis=1)
            total_rocket = np.sum(rocket_flows, axis=1)
            total_transport = total_se + total_rocket
            
            # 火箭占比 (0.0 - 1.0)
            rocket_ratios = np.divide(total_rocket, total_transport, 
                                    out=np.zeros_like(total_rocket), where=total_transport!=0)
            
            # 存储扩展结果: [Cost, Time, Env, RocketRatio]
            # 我们把 RocketRatio 拼接到结果矩阵的第4列方便绘图
            extended_F = np.column_stack((F_final, rocket_ratios))
            results[mult] = extended_F
            
            print(f"  Avg Rocket Ratio: {np.mean(rocket_ratios)*100:.1f}%")
        else:
            print("  Failed.")

    # --- 3. 改进的可视化 ---
    plot_structural_sensitivity(results, multipliers)

def plot_structural_sensitivity(results, multipliers):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(multipliers)))
    markers = ['o', 's', '^', 'D']

    # --- 图1: 策略变化 (Rocket Usage % vs Time) ---
    # 这张图直接回答：贵了以后，我是不是少用了？
    for i, mult in enumerate(multipliers):
        if mult not in results: continue
        data = results[mult]
        
        # data[:, 1] 是 Time
        # data[:, 3] 是 Rocket Ratio (我们刚计算的)
        
        # 按时间排序平滑曲线
        sorted_idx = np.argsort(data[:, 1])
        data_sorted = data[sorted_idx]
        
        # 绘制趋势线 (Lowess平滑或直接连线)
        ax1.scatter(data_sorted[:, 1], data_sorted[:, 3] * 100, 
                    color=colors[i], marker=markers[i], alpha=0.6, s=30)
        
        # 拟合一条平滑曲线看趋势 (可选)
        try:
            z = np.polyfit(data_sorted[:, 1], data_sorted[:, 3] * 100, 2)
            p = np.poly1d(z)
            x_line = np.linspace(data_sorted[:, 1].min(), data_sorted[:, 1].max(), 100)
            ax1.plot(x_line, p(x_line), color=colors[i], label=f'{mult}x Cost', linewidth=2)
        except:
            pass # 点太少如果不拟合就只画点

    ax1.set_xlabel('Completion Time (Years)', fontsize=12)
    ax1.set_ylabel('Rocket Transport Share (%)', fontsize=12)
    ax1.set_title('Strategy Shift: Rocket Dependency vs Time', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # --- 图2: 归一化成本 (Normalized Cost vs Time) ---
    # 这张图回答：剔除涨价因素后，效率损失了多少？
    for i, mult in enumerate(multipliers):
        if mult not in results: continue
        data = results[mult]
        
        # 归一化成本 = 原始总成本 / 倍率
        # 如果曲线重合，说明物理方案没变；如果分离，说明方案变了
        norm_cost = data[:, 0] / mult 
        
        sorted_idx = np.argsort(data[:, 1])
        ax2.plot(data[sorted_idx, 1], norm_cost[sorted_idx], 
                 color=colors[i], marker=markers[i], label=f'{mult}x (Normalized)', 
                 linestyle='--', alpha=0.7)

    ax2.set_xlabel('Completion Time (Years)', fontsize=12)
    ax2.set_ylabel('Normalized Cost ($B / Multiplier)', fontsize=12)
    ax2.set_title('Efficiency Loss: Normalized Cost vs Time', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/sensitivity_structural.png', dpi=300)
    print("\n图表已保存至 images/sensitivity_structural.png")
    plt.show()

if __name__ == "__main__":
    run_rocket_cost_sensitivity_advanced()