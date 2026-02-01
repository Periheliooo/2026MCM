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
    
    # 3. 绘图
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2)
    
    # 图A: 运输策略堆叠图 (Transport Strategy)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.stackplot(df_hist['Year'], df_hist['SE_Flow'], df_hist['Rocket_Flow'], 
                  labels=['Space Elevator', 'Rocket'], 
                  colors=['#3498db', '#e74c3c'], alpha=0.8)
    ax1.set_title(f'Optimal Transport Strategy (Solution ID: {best_idx})', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Annual Transport Volume (10k Tons)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 标注完工时间
    complete_row = df_hist[df_hist['Accumulated_Transport'] >= 100].head(1)
    if not complete_row.empty:
        end_year = complete_row['Year'].values[0]
        ax1.axvline(end_year, color='green', linestyle='--', linewidth=2, label='Completion')
        ax1.text(end_year+1, ax1.get_ylim()[1]*0.8, f'Completed: {end_year}', color='green', fontweight='bold')

    # 图B: 环境累积压力 (At vs Threshold)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df_hist['Year'], df_hist['Env_Impact_Accum (At)'], color='purple', linewidth=2, label='Accumulated Impact ($A_t$)')
    ax2.plot(df_hist['Year'], df_hist['Env_Threshold (A)'], color='red', linestyle='--', label='Carrying Capacity ($A$)')
    ax2.fill_between(df_hist['Year'], df_hist['Env_Impact_Accum (At)'], color='purple', alpha=0.1)
    ax2.set_title('Environmental Stress Dynamics ($A_t$)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accumulated Impact Index', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图C: 年度总环境影响 (Yearly Impact)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df_hist['Year'], df_hist['Env_Impact_Yearly'], color='orange', linewidth=2, label='Yearly Total Impact')
    ax3.set_title('Annual Environmental Impact Flow', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Impact Index', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 图D: 成本分布 (Cost Profile)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(df_hist['Year'], df_hist['Cost_Yearly'], color='darkgreen', marker='o', markersize=3, alpha=0.6)
    ax4.set_title('Annual Expenditure Profile', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Cost (Billion USD)', fontsize=12)
    ax4.set_xlabel('Year')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    save_path = f'images/optimal_solution_{best_idx}_details.png'
    plt.savefig(save_path, dpi=300)
    print(f"图表已生成: {save_path}")
    # plt.show()

if __name__ == "__main__":
    # 使用 select_best_solution.py 推荐的 ID
    visualize_specific_solution(best_idx=0)