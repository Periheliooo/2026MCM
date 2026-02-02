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
from matplotlib.colors import LinearSegmentedColormap

# 复用之前的模型和问题定义
from problem_4_model import MoonColonyModel
from problem_4_solver import MoonOptimizationProblem, HeuristicSampling

# ==========================================
# 1. 设置绘图风格 (科学美观)
# ==========================================
# 使用 Seaborn 的高级风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.alpha'] = 0.4

def run_sensitivity_analysis():
    print("="*60)
    print("MCM Model 4: Environmental Threshold Sensitivity Analysis")
    print("="*60)
    
    # 1. 定义敏感度参数范围
    # 我们测试环境承载力阈值的倍数: 0.5x (严苛) -> 2.0x (宽松)
    multipliers = [0.2, 0.5, 0.8, 1.0, 1.25]
    
    # 存储结果的列表
    results_list = []
    summary_metrics = []
    
    # 2. 循环运行优化
    for mult in multipliers:
        print(f"\n>> Running Scenario: Threshold x {mult} ...")
        
        # A. 初始化模型
        try:
            model = MoonColonyModel('harbor.xlsx')
        except:
            # Fallback for relative path
            model = MoonColonyModel('../data/harbor.xlsx')
            
        # B. 修改环境阈值参数
        original_threshold = model.env_threshold_A
        model.env_threshold_A = original_threshold * mult
        print(f"   Current Env Threshold (A): {model.env_threshold_A:,.0f} (Base: {original_threshold:,.0f})")
        
        # C. 定义优化问题 (保持参数一致)
        N_YEARS = 130 
        problem = MoonOptimizationProblem(model, n_years=N_YEARS)
        
        # D. 配置算法 (适当减少代数以加快分析速度，实际论文中可设为 200)
        algorithm = NSGA2(
            pop_size=60,         # 略微减小种群
            n_offsprings=30,
            sampling=HeuristicSampling(model, N_YEARS),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=0.01, eta=20),
            eliminate_duplicates=True
        )
        
        termination = get_termination("n_gen", 100) # 100代
        
        # E. 求解
        res = minimize(problem, algorithm, termination, seed=42, verbose=False)
        
        # F. 提取数据
        if res.F is not None:
            # 筛选可行解
            feasible_mask = (res.CV.flatten() <= 1e-5)
            F_valid = res.F[feasible_mask]
            
            if len(F_valid) > 0:
                print(f"   Solutions found: {len(F_valid)}")
                
                # 存入 DataFrame
                df_temp = pd.DataFrame(F_valid, columns=['Cost', 'Time', 'Env'])
                df_temp['Multiplier'] = mult
                df_temp['Threshold_Label'] = f"{mult}x ($A={model.env_threshold_A/1e6:.1f}M$)"
                results_list.append(df_temp)
                
                # 计算关键指标 (用于图2)
                min_cost = df_temp['Cost'].min()
                min_time = df_temp['Time'].min()
                # 寻找折衷解 (Normalize后距离原点最近)
                # 在第 94 行之前，筛选出数值列
                numeric_df = df_temp.select_dtypes(include=['number'])

                # 对数值列进行归一化
                norm_df = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min() + 1e-9)

                # 如果你还需要保留原来的非数值列，可以再合并回来
                # df_final = pd.concat([norm_df, df_temp.select_dtypes(exclude=['number'])], axis=1)
                dist = np.sqrt((norm_df**2).sum(axis=1))
                knee_idx = dist.idxmin()
                knee_cost = df_temp.loc[knee_idx, 'Cost']
                
                summary_metrics.append({
                    'Multiplier': mult,
                    'Threshold': model.env_threshold_A,
                    'Min_Cost': min_cost,
                    'Min_Time': min_time,
                    'Knee_Cost': knee_cost
                })
            else:
                print("   No feasible solutions found for this scenario.")
        else:
            print("   Optimization failed.")

    # 合并数据
    if not results_list:
        print("Error: No results generated.")
        return

    all_pareto_df = pd.concat(results_list, ignore_index=True)
    summary_df = pd.DataFrame(summary_metrics)
    
    # 保存数据备份
    os.makedirs('data', exist_ok=True)
    all_pareto_df.to_csv('data/sensitivity_pareto_results.csv', index=False)
    summary_df.to_csv('data/sensitivity_summary_metrics.csv', index=False)
    
    # ==========================================
    # 3. 可视化 (Scientific Visualization)
    # ==========================================
    os.makedirs('images', exist_ok=True)
    
    # --- 图 1: Pareto 前沿演变 (Cost vs Time) ---
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    # 自定义渐变色 (从红色/严苛 到 蓝色/宽松)
    palette = sns.color_palette("rocket_r", n_colors=len(multipliers))
    
    sns.scatterplot(
        data=all_pareto_df, 
        x='Time', y='Cost', 
        hue='Threshold_Label', 
        palette=palette,
        style='Threshold_Label',
        s=80, alpha=0.8, edgecolor='w', linewidth=0.5,
        ax=ax1
    )
    
    # 拟合平滑曲线 (Pareto Front Approximation) 以增强视觉效果
    for i, mult in enumerate(multipliers):
        subset = all_pareto_df[all_pareto_df['Multiplier'] == mult].sort_values('Time')
        if len(subset) > 2:
            # 简单的滚动平均或多项式拟合来画线
            # 这里直接画折线即可，因为是Pareto前沿
            ax1.plot(subset['Time'], subset['Cost'], color=palette[i], alpha=0.4, linewidth=1.5)

    ax1.set_title('Sensitivity of Pareto Front to Environmental Threshold ($A$)', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Project Duration (Years)', fontsize=14)
    ax1.set_ylabel('Total Cost (Billion USD)', fontsize=14)
    ax1.legend(title='Env. Threshold Multiplier', title_fontsize=12, fontsize=11, loc='upper right', frameon=True, shadow=True)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 标注趋势箭头
    # 找两个极端的前沿重心
    df_low = all_pareto_df[all_pareto_df['Multiplier'] == min(multipliers)]
    df_high = all_pareto_df[all_pareto_df['Multiplier'] == max(multipliers)]
    if not df_low.empty and not df_high.empty:
        x_start, y_start = df_high['Time'].mean(), df_high['Cost'].mean()
        x_end, y_end = df_low['Time'].mean(), df_low['Cost'].mean()
        ax1.annotate(
            "Stricter Environmental\nConstraints Increase Costs",
            xy=(x_end, y_end), xycoords='data',
            xytext=(x_start + 10, y_start - 500), textcoords='data',
            arrowprops=dict(arrowstyle="->", color='black', lw=1.5, connectionstyle="arc3,rad=0.2"),
            fontsize=12, fontweight='bold', color='#333333',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
        )

    plt.tight_layout()
    plt.savefig('images/sensitivity_pareto_fronts.png', dpi=300)
    print("图表 1 已生成: images/sensitivity_pareto_fronts.png")
    
    # --- 图 2: 关键指标变化 (Cost of Sustainability) ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # 双轴图
    color_cost = '#2ecc71' # Green
    color_time = '#3498db' # Blue
    
    # 绘制最低成本变化
    line1 = ax2.plot(summary_df['Multiplier'], summary_df['Min_Cost'], 
             marker='o', markersize=10, linewidth=2.5, color=color_cost, label='Min Cost (Global)')
    
    ax2.set_xlabel('Threshold Multiplier (Relative to Baseline)', fontsize=14)
    ax2.set_ylabel('Minimum Cost (Billion USD)', fontsize=14, color=color_cost)
    ax2.tick_params(axis='y', labelcolor=color_cost)
    ax2.set_title('Trade-off: Cost of Sustainability', fontsize=16, fontweight='bold', pad=15)
    
    # 绘制最短时间变化 (Twin axis)
    ax2_right = ax2.twinx()
    line2 = ax2_right.plot(summary_df['Multiplier'], summary_df['Min_Time'], 
                   marker='s', markersize=10, linewidth=2.5, color=color_time, linestyle='--', label='Min Duration')
    ax2_right.set_ylabel('Minimum Duration (Years)', fontsize=14, color=color_time)
    ax2_right.tick_params(axis='y', labelcolor=color_time)
    
    # 添加图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right', frameon=True)
    
    ax2.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('images/sensitivity_metrics.png', dpi=300)
    print("图表 2 已生成: images/sensitivity_metrics.png")
    
    plt.show() # 如果在本地运行可取消注释

if __name__ == "__main__":
    run_sensitivity_analysis()