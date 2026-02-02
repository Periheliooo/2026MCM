import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from problem_4_model import MoonColonyModel

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif' # 如果有中文字体可改为 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

def visualize_specific_solution(best_idx=0):
    """
    可视化指定索引的最优解详情
    :param best_idx: 在 nsga2_pareto_results.csv 中的行号
    """
    # 1. 加载数据
    try:
        df_X = pd.read_csv('data/nsga2_pareto_solutions_X.csv')
        df_F = pd.read_csv('data/nsga2_pareto_results.csv')
    except FileNotFoundError:
        print("错误：找不到解文件。请先运行 solver 并确保保存了 X 和 F 数据。")
        return

    # 检查索引是否越界
    if best_idx >= len(df_X):
        print(f"索引 {best_idx} 超出范围 (最大 {len(df_X)-1})。默认使用索引 0。")
        best_idx = 0

    # 获取指定解的变量
    x_solution = df_X.iloc[best_idx].values
    n_years = len(x_solution) // 2
    x_se = x_solution[:n_years]
    x_rocket = x_solution[n_years:]
    
    # 2. 复现模型计算过程 (获取逐年数据)
    try:
        model = MoonColonyModel('harbor.xlsx')
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return
    
    years = np.arange(model.START_YEAR, model.START_YEAR + n_years)
    
    # 记录详细数据的列表
    history = {
        'Year': [],
        'SE_Flow': [],
        'Rocket_Flow': [],
        'Accumulated_Transport': [],
        'Env_Impact_Yearly': [],       # [Fix] 之前漏了这个的 append
        'Env_Impact_Accum (At)': [],
        'Env_Threshold (A)': [],
        'Cost_Yearly': []
    }
    
    current_A_t = 0
    total_transported = 0
    
    # 预计算参数
    base_caps_tons = model.base_capacities_launches * model.ROCKET_PAYLOAD
    
    print(f"正在分析第 {best_idx} 个最优解...")
    print(f"目标值: Cost=${df_F.iloc[best_idx]['Cost(Billion)']:.2f}B, "
          f"Time={df_F.iloc[best_idx]['Time(Years)']:.1f}y, "
          f"Env={df_F.iloc[best_idx]['Env_Impact']:.2f}")

    for t_idx, year in enumerate(years):
        # 恢复流
        m_se = max(0, x_se[t_idx])
        m_rocket = max(0, x_rocket[t_idx])
        
        # 截断逻辑 (如果已经运完)
        if total_transported >= model.TOTAL_DEMAND:
            m_se = 0
            m_rocket = 0
            
        total_transported += (m_se + m_rocket)
        
        # --- 环境计算 (详细复现) ---
        
        # 1. SE Direct Impact
        env_se = model.c_SE * m_se * model.geo_factor_SE
        
        # 2. Rocket Direct Impact & Cost (贪心分配估算)
        env_rocket = 0
        rem_load = m_rocket
        # 获取当年火箭单价用于成本估算
        unit_cost = model.get_rocket_cost_per_ton(year)
        cost_rocket = 0
        
        for b_idx in range(model.num_bases):
            if rem_load <= 1e-3: break
            fill = min(rem_load, base_caps_tons[b_idx])
            launches = fill / model.ROCKET_PAYLOAD
            
            # 环境: 基础 * 发射数 * 地理系数
            env_rocket += model.c_R * launches * model.base_geo_coeffs[b_idx]
            # 成本: 量 * 单价 * 成本系数
            cost_rocket += fill * unit_cost * model.base_cost_coeffs[b_idx]
            
            rem_load -= fill
            
        # 3. Space Debris Impact
        n_launch = np.ceil(m_rocket / model.ROCKET_PAYLOAD)
        debris = (n_launch * model.debris_coeff_launch + 
                  n_launch * model.prob_fail * model.debris_coeff_fail) * model.cost_per_debris_unit
        
        # 4. At 更新 (累积影响)
        act_input = model.a_SE * env_se + model.a_R * env_rocket
        current_A_t = current_A_t * (1 - model.decay_rate) + act_input
        
        # 5. Dt 计算 (阈值惩罚) - [Fix] 之前漏了这部分计算
        if current_A_t > model.env_threshold_A:
            overdraft = (current_A_t - model.env_threshold_A) / model.env_threshold_A
            D_t = model.c_st * (1 + model.k_penalty * (overdraft ** model.r_penalty))
        else:
            D_t = 0
            
        # 6. 当年总环境影响
        total_yearly_impact = (env_se + env_rocket + debris) + D_t
        
        # Cost SE
        cost_se = model.SE_FIXED_COST + m_se * model.SE_VAR_COST
        
        # --- 记录数据 (Fix: 确保所有列表都 append) ---
        history['Year'].append(year)
        history['SE_Flow'].append(m_se / 1e4) # 单位：万吨
        history['Rocket_Flow'].append(m_rocket / 1e4) # 单位：万吨
        history['Accumulated_Transport'].append(total_transported / 1e6) # 单位：百万吨
        history['Env_Impact_Yearly'].append(total_yearly_impact) # [Fix] 补上这个
        history['Env_Impact_Accum (At)'].append(current_A_t)
        history['Env_Threshold (A)'].append(model.env_threshold_A)
        history['Cost_Yearly'].append((cost_se + cost_rocket)/1e9) # 单位：Billion
        
    # 创建 DataFrame
    df_hist = pd.DataFrame(history)

    # ... [前面的数据处理逻辑保持不变] ...

    # 1. 更加现代的样式设置
    plt.style.use('seaborn-v0_8-paper') # 使用更适合论文的精细风格
    sns.set_palette("viridis") # 使用科学配色
    
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 0.8])
    
    # --- 图A: 运输策略 (现代化的面积图) ---
    ax1 = fig.add_subplot(gs[0, :])
    # 使用渐变感配色
    colors = ['#5dade2', '#ec7063']
    ax1.stackplot(df_hist['Year'], df_hist['SE_Flow'], df_hist['Rocket_Flow'], 
                  labels=['太空电梯 (Space Elevator)', '化学火箭 (Rocket)'], 
                  colors=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    
    ax1.set_title(f'Strategy Profile: Solution #{best_idx}', loc='left', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Volume (10k Tons/yr)', fontsize=12)
    
    # 动态标注：完成度
    comp_year = df_hist[df_hist['Accumulated_Transport'] >= 100]['Year'].min()
    ax1.axvline(comp_year, color='#27ae60', linestyle=':', linewidth=2)
    ax1.annotate(f'Mission Target Reached: {int(comp_year)}', xy=(comp_year, ax1.get_ylim()[1]*0.5), 
                 xytext=(comp_year+2, ax1.get_ylim()[1]*0.6),
                 arrowprops=dict(arrowstyle='->', color='#27ae60'), color='#1e8449', fontweight='bold')

    # --- 图B: 环境压力 (双轴或带状图) ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(df_hist['Year'], df_hist['Env_Threshold (A)'], color='gray', alpha=0.1, label='Safe Zone')
    ax2.plot(df_hist['Year'], df_hist['Env_Impact_Accum (At)'], color='#8e44ad', lw=2.5, label='Actual Stress ($A_t$)')
    ax2.axhline(model.env_threshold_A, color='#c0392b', ls='--', lw=1.5, label='Threshold')
    
    # 局部高亮：如果超标则标红
    over_limit = df_hist['Env_Impact_Accum (At)'] > df_hist['Env_Threshold (A)']
    ax2.fill_between(df_hist['Year'], df_hist['Env_Impact_Accum (At)'], df_hist['Env_Threshold (A)'], 
                     where=over_limit, color='#e74c3c', alpha=0.3, label='Exceedance')
    
    ax2.set_title('Environmental Carrying Capacity', fontsize=13, fontweight='semibold')
    ax2.legend(frameon=True, loc='upper left', fontsize=9)

    # --- 图C: 年度环境流量 (柱状图) ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(df_hist['Year'], df_hist['Env_Impact_Yearly'], color='#f39c12', alpha=0.7, width=0.8)
    ax3.set_title('Annual Environmental Impact Flow', fontsize=13, fontweight='semibold')
    ax3.set_ylabel('Impact Index')

    # --- 图D: 成本分布 (带平滑线的散点) ---
    ax4 = fig.add_subplot(gs[2, :])
    ax4.fill_between(df_hist['Year'], df_hist['Cost_Yearly'], color='#2ecc71', alpha=0.15)
    ax4.plot(df_hist['Year'], df_hist['Cost_Yearly'], color='#27ae60', marker='o', ms=4, lw=1.5, mec='white')
    
    ax4.set_title('Capital Expenditure (CapEx/OpEx) Profile', fontsize=13, fontweight='semibold')
    ax4.set_ylabel('Cost (Billion USD)')
    ax4.set_xlabel('Timeline (Year)')

    # 全局美化
    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.savefig(f'images/optimal_solution_{best_idx}_pro.png', dpi=300, bbox_inches='tight')
    plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_final_topsis_3d_pareto(df_F, knee_idx=32):
    """
    绘制包含膝点、投影线和 TOPSIS 得分映射的专业 3D 帕累托图
    """
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. 准备数据
    x = df_F['Cost(Billion)']
    y = df_F['Time(Years)']
    z = df_F['Env_Impact']
    
    # 假设你已经算出了综合得分 (如果没有，请将下行替换为固定值如 s=50)
    # 映射得分到点的大小，得分越高点越大
    scores = df_F['Score'] if 'Score' in df_F.columns else np.linspace(20, 100, len(df_F))
    sizes = scores * 1.5  # 缩放因子
    
    # 2. 绘制基础帕累托前沿
    # 使用渐变色 cmap='plasma' 增加科技感
    sc = ax.scatter(x, y, z, c=z, cmap='plasma', s=sizes, alpha=0.4, edgecolors='w', linewidth=0.3)
    
    # 3. 绘制通用背景投影 (仅到底面，保持画面整洁)
    for xi, yi, zi in zip(x, y, z):
        ax.plot([xi, xi], [yi, yi], [z.min(), zi], color='lightgray', linestyle='-', linewidth=0.3, alpha=0.1)
    
    # 4. 膝点高亮与多维投影
    if knee_idx in df_F.index:
        knee = df_F.loc[knee_idx]
        kx, ky, kz = knee['Cost(Billion)'], knee['Time(Years)'], knee['Env_Impact']
        
        # 膝点本体：红色巨大五角星
        ax.scatter(kx, ky, kz, color='#FF0000', marker='*', s=500, edgecolors='k', linewidth=1.2, label='Knee Point (Best Selection)', zorder=20)
        
        # 膝点专属投影线：连接到三个轴的平面，增强空间定位
        ax.plot([kx, kx], [ky, ky], [z.min(), kz], color='#FF0000', linestyle='--', linewidth=1.5, alpha=0.8) # Z轴向
        ax.plot([x.min(), kx], [ky, ky], [kz, kz], color='#FF0000', linestyle=':', linewidth=1.2, alpha=0.5)  # X轴向
        ax.plot([kx, kx], [y.min(), ky], [kz, kz], color='#FF0000', linestyle=':', linewidth=1.2, alpha=0.5)  # Y轴向
        
        # 膝点数值标注：使用带框的文本
        label_txt = f"  RECOMENDED SOLUTION\n  -------------------\n  Cost: ${kx:,.0f}B\n  Time: {ky:.1f}y\n  Env: {kz/1e6:.2f}M"
        ax.text(kx, ky, kz * 1.05, label_txt, color='black', fontsize=9, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='#FF0000', boxstyle='round,pad=0.5'))

    # 5. 视觉精修
    ax.set_xlabel('\nTotal Cost ($B)', fontsize=10, fontweight='bold')
    ax.set_ylabel('\nCompletion Time (Years)', fontsize=10, fontweight='bold')
    ax.set_zlabel('\nEnvironmental Impact Index', fontsize=10, fontweight='bold')
    
    # 去除冗余网格色块
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # 颜色条标注
    cbar = plt.colorbar(sc, ax=ax, shrink=0.4, aspect=15, pad=0.05)
    cbar.set_label('Impact Severity', rotation=270, labelpad=15)
    
    # 视角微调
    ax.view_init(elev=22, azim=-125)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.9), frameon=True, shadow=True)
    
    plt.title('Multi-Objective Strategic Selection\n(Knee Point Analysis for Moon Colony Logistics)', fontsize=15, fontweight='bold', pad=5)
    plt.savefig('images/final_topsis_3d_pareto_pro.png', dpi=300)
    plt.show()

# 调用示例
# plot_refined_3d_pareto_with_knee(df_F, knee_idx=32)

if __name__ == "__main__":
    # 使用 select_best_solution.py 推荐的 ID
    visualize_specific_solution(best_idx=0)
    # 绘制包含膝点高亮的 3D Pareto 图
    try:
        df_F = pd.read_csv('data/nsga2_pareto_results.csv')
        plot_final_topsis_3d_pareto(df_F, knee_idx=88)
    except FileNotFoundError:
        print("错误：找不到 Pareto 结果文件。请先运行 solver 并确保保存了结果数据。")
    # 可根据需要更改 best_idx 和 knee_idx 参数