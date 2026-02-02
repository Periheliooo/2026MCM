import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# 复用之前的模型和问题定义
from problem_4_model import MoonColonyModel
from problem_4_solver import MoonOptimizationProblem, HeuristicSampling

# ==========================================
# 1. 设置绘图风格 (科学美观)
# ==========================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.alpha'] = 0.4

def run_decay_sensitivity():
    print("="*60)
    print("MCM Model 4: Environmental Self-Purification (Decay Rate) Sensitivity")
    print("="*60)
    
    # 1. 定义敏感度参数范围
    # 我们测试衰减率的倍数: 
    # 0.5x (净化慢/恶劣) -> 2.0x (净化快/理想)
    # 基础值通常为 0.1
    multipliers = [0.5, 0.8, 1.0, 1.5, 2.0]
    
    results_list = []
    summary_metrics = []
    
    # 2. 循环运行优化
    for mult in multipliers:
        print(f"\n>> Running Scenario: Decay Rate x {mult} ...")
        
        # A. 初始化模型
        try:
            model = MoonColonyModel('harbor.xlsx')
        except:
            model = MoonColonyModel('../data/harbor.xlsx') # Fallback
            
        # B. 修改衰减率参数
        original_decay = model.decay_rate
        model.decay_rate = min(1.0, original_decay * mult) # 确保不超过 100%
        print(f"   Current Decay Rate: {model.decay_rate:.3f} (Base: {original_decay:.3f})")
        
        # C. 定义优化问题
        N_YEARS = 130 
        problem = MoonOptimizationProblem(model, n_years=N_YEARS)
        
        # D. 配置算法 (快速扫描模式)
        algorithm = NSGA2(
            pop_size=60,         
            n_offsprings=30,
            sampling=HeuristicSampling(model, N_YEARS),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=0.01, eta=20),
            eliminate_duplicates=True
        )
        
        termination = get_termination("n_gen", 100) 
        
        # E. 求解
        res = minimize(problem, algorithm, termination, seed=42, verbose=False)
        
        # F. 提取数据
        if res.F is not None:
            feasible_mask = (res.CV.flatten() <= 1e-5)
            F_valid = res.F[feasible_mask]
            
            if len(F_valid) > 0:
                print(f"   Solutions found: {len(F_valid)}")
                
                df_temp = pd.DataFrame(F_valid, columns=['Cost', 'Time', 'Env'])
                df_temp['Multiplier'] = mult
                df_temp['Decay_Label'] = f"{mult}x (Rate={model.decay_rate:.2f})"
                results_list.append(df_temp)
                
                # 计算关键指标
                min_cost = df_temp['Cost'].min()
                min_time = df_temp['Time'].min()
                
                summary_metrics.append({
                    'Multiplier': mult,
                    'Decay_Rate': model.decay_rate,
                    'Min_Cost': min_cost,
                    'Min_Time': min_time
                })
            else:
                print("   No feasible solutions found (Constraints too strict?).")
        else:
            print("   Optimization failed.")

    if not results_list:
        print("Error: No results generated.")
        return

    all_pareto_df = pd.concat(results_list, ignore_index=True)
    summary_df = pd.DataFrame(summary_metrics)
    
    # 保存数据
    os.makedirs('data', exist_ok=True)
    all_pareto_df.to_csv('data/sensitivity_decay_pareto.csv', index=False)
    summary_df.to_csv('data/sensitivity_decay_summary.csv', index=False)
    
    # ==========================================
    # 3. 可视化 (Scientific Visualization)
    # ==========================================
    os.makedirs('images', exist_ok=True)
    
    # --- 图 1: Pareto 前沿演变 ---
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    # 颜色逻辑: 净化慢(0.5x)是坏事(红色/深色)，净化快(2.0x)是好事(蓝色/浅色)
    # 使用 'viridis' 或 'coolwarm' 
    palette = sns.color_palette("RdYlBu", n_colors=len(multipliers))
    
    sns.scatterplot(
        data=all_pareto_df, 
        x='Time', y='Cost', 
        hue='Decay_Label', 
        palette=palette,
        style='Decay_Label',
        s=80, alpha=0.8, edgecolor='w', linewidth=0.5,
        ax=ax1
    )
    
    # 绘制平滑趋势线
    for i, mult in enumerate(multipliers):
        subset = all_pareto_df[all_pareto_df['Multiplier'] == mult].sort_values('Time')
        if len(subset) > 2:
            ax1.plot(subset['Time'], subset['Cost'], color=palette[i], alpha=0.4, linewidth=1.5)

    ax1.set_title('Impact of Environmental Self-Purification (Decay Rate) on Costs', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Project Duration (Years)', fontsize=14)
    ax1.set_ylabel('Total Cost (Billion USD)', fontsize=14)
    ax1.legend(title='Decay Rate Multiplier', title_fontsize=12, fontsize=11, loc='upper right', frameon=True, shadow=True)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 标注箭头：低衰减率 -> 高成本
    df_slow = all_pareto_df[all_pareto_df['Multiplier'] == min(multipliers)] # 0.5x
    df_fast = all_pareto_df[all_pareto_df['Multiplier'] == max(multipliers)] # 2.0x
    
    if not df_slow.empty and not df_fast.empty:
        x_fast, y_fast = df_fast['Time'].mean(), df_fast['Cost'].mean()
        x_slow, y_slow = df_slow['Time'].mean(), df_slow['Cost'].mean()
        
        ax1.annotate(
            "Slower Natural Recovery\n(Requires Costly Restrictions)",
            xy=(x_slow, y_slow), xycoords='data',
            xytext=(x_fast + 5, y_fast - 300), textcoords='data',
            arrowprops=dict(arrowstyle="->", color='darkred', lw=2, connectionstyle="arc3,rad=-0.2"),
            fontsize=12, fontweight='bold', color='darkred',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", alpha=0.8)
        )

    plt.tight_layout()
    plt.savefig('images/sensitivity_decay_pareto.png', dpi=300)
    print("图表 1 已生成: images/sensitivity_decay_pareto.png")
    
    # --- 图 2: 关键指标变化 (成本/时间 vs 衰减率) ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    color_cost = '#e74c3c' # Red for Cost
    color_time = '#2980b9' # Blue for Time
    
    # 绘制最低成本
    line1 = ax2.plot(summary_df['Decay_Rate'], summary_df['Min_Cost'], 
             marker='o', markersize=10, linewidth=2.5, color=color_cost, label='Min Cost (Global)')
    
    ax2.set_xlabel('Environmental Decay Rate (Higher = Faster Recovery)', fontsize=14)
    ax2.set_ylabel('Minimum Cost (Billion USD)', fontsize=14, color=color_cost)
    ax2.tick_params(axis='y', labelcolor=color_cost)
    # 反转 X 轴? 不，直接看趋势即可。数值越大越好。
    # 成本应该随 Decay Rate 增加而降低 (反比关系)
    
    ax2.set_title('Sensitivity: How Nature\'s Recovery Speed Affects Budget', fontsize=16, fontweight='bold', pad=15)
    
    # 绘制最短时间 (Twin Axis)
    ax2_right = ax2.twinx()
    line2 = ax2_right.plot(summary_df['Decay_Rate'], summary_df['Min_Time'], 
                   marker='s', markersize=10, linewidth=2.5, color=color_time, linestyle='--', label='Min Duration')
    ax2_right.set_ylabel('Minimum Duration (Years)', fontsize=14, color=color_time)
    ax2_right.tick_params(axis='y', labelcolor=color_time)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper center', frameon=True)
    
    ax2.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    # 添加一个临界区标注 (如果曲线很陡峭)
    # 假设在 0.05 处成本极高
    if summary_df['Min_Cost'].max() > summary_df['Min_Cost'].min() * 1.5:
        max_cost_x = summary_df.loc[summary_df['Min_Cost'].idxmax(), 'Decay_Rate']
        ax2.axvspan(max_cost_x - 0.01, max_cost_x + 0.01, color='red', alpha=0.1)
        ax2.text(max_cost_x, ax2.get_ylim()[1]*0.9, "Critical\nRisk Zone", color='red', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('images/sensitivity_decay_metrics.png', dpi=300)
    plt.show() # 如果在本地运行可取消注释
    print("图表 2 已生成: images/sensitivity_decay_metrics.png")

if __name__ == "__main__":
    run_decay_sensitivity()