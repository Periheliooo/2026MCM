import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import os

# -----------------------------------------------------------
# 1. 初始化设置与参数定义
# -----------------------------------------------------------
np.random.seed(42)

# 路径设置 (保持原有逻辑)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设数据文件在上一级目录的data文件夹中，如果在同级目录请调整路径
try:
    df_result = pd.read_excel(os.path.join(current_dir, '..', 'data', 'problem1_result.xlsx'))
    df_harbor = pd.read_excel(os.path.join(current_dir, '..', 'data', 'harbor.xlsx'))
except FileNotFoundError:
    # 如果找不到文件，尝试在当前目录读取（方便调试）
    try:
        df_result = pd.read_excel('problem1_result.xlsx')
        df_harbor = pd.read_excel('harbor.xlsx')
    except:
        print("Warning: Data files not found. Using mock data for demonstration if needed.")
        # 这里你可以添加生成模拟数据的代码，或者报错停止

# 核心参数
SE_CAPACITY_PLAN = 536999.25   # 太空电梯设计年运力 (吨)
SE_FIXED_COST = 4.12e9         # 太空电梯年固定维护成本 ($)
SE_VAR_COST_PER_TON = 60000    # 太空电梯单位变动成本 ($/吨)
TARGET_MASS = 1e8              # 目标总运输量 (1亿吨)
ROCKET_NOMINAL_PAYLOAD = 125   # 火箭名义运力（用于规划发射次数）

# 不确定性参数
FAIL_PROB_ROCKET = 0.05        # 火箭发射失败概率
SE_EFFICIENCY_MEAN = 0.95      # 太空电梯平均效率
SE_EFFICIENCY_STD = 0.05       # 太空电梯效率标准差
SIMULATIONS = 1000             # 模拟次数

# 预处理基地数据
if 'df_harbor' in locals():
    harbors = df_harbor.set_index('基地名称')
    base_coeffs = harbors['2050年单次成本系数'].to_dict()
    base_max_launches = harbors['2050年最大年发射次数'].to_dict()
    base_names = list(base_coeffs.keys())
else:
    base_names = [] # Fallback

# -----------------------------------------------------------
# [NEW] 动态成本计算函数
# -----------------------------------------------------------
def get_rocket_price_per_ton(year):
    """
    根据拟合公式计算当年的火箭基准成本 ($/ton)
    公式: Cost(t) = 100 + 12906.93 * (t - 2005)^(-0.8872)
    假设公式输出为 $/kg, 因此需 * 1000 转换为 $/ton
    """
    t_val = year - 2005
    if t_val <= 0: t_val = 0.1 # 防止计算错误
    
    cost_per_kg = 100 + 12906.93 * (t_val ** -0.8872)
    return cost_per_kg * 1000

# -----------------------------------------------------------
# 2. 单次模拟函数
# -----------------------------------------------------------
def run_simulation():
    cumulative_mass = 0
    total_cost = 0
    year = 2051
    backlog = 0 
    
    while cumulative_mass < TARGET_MASS:
        # A. 获取当年的计划量
        if year <= 2141:
            try:
                row = df_result[df_result['Year'] == year]
                if not row.empty:
                    # --- 修改前的代码 ---
                    # se_plan = row['x_t'].values[0]
                    # rocket_plans = {base: row[base].values[0] ...}

                    # --- 修改后的代码 (策略一) ---
                    SAFETY_FACTOR = 1.06  # 引入 6% 的冗余缓冲

                    # 1. 电梯计划加量 (但不超过物理上限)
                    raw_se_plan = row['x_t'].values[0]
                    se_plan = min(raw_se_plan * SAFETY_FACTOR, SE_CAPACITY_PLAN) # 假设原设计就是上限，则无法增加，或者假设可以超频运行

                    # 2. 火箭计划加量
                    raw_rocket_plans = {base: row[base].values[0] for base in base_names if base in row.columns}
                    rocket_plans = {}
                    for base, amount in raw_rocket_plans.items():
                        # 尝试增加 6% 的发射量
                        planned_amount = amount * SAFETY_FACTOR
    
                        # 必须检查是否超过了该基地的最大年发射次数限制！
                        # 这是一个物理硬约束
                        max_cap_mass = base_max_launches.get(base, 0) * ROCKET_NOMINAL_PAYLOAD
                        rocket_plans[base] = min(planned_amount, max_cap_mass)
                else:
                    se_plan = SE_CAPACITY_PLAN
                    rocket_plans = {base: 0 for base in base_names}
            except:
                se_plan = SE_CAPACITY_PLAN
                rocket_plans = {base: 0 for base in base_names}
        else:
            # 超期处理
            try:
                row = df_result[df_result['Year'] == 2141]
                se_plan = row['x_t'].values[0]
                rocket_plans = {base: row[base].values[0] for base in base_names if base in row.columns}
            except:
                se_plan = SE_CAPACITY_PLAN
                rocket_plans = {base: 0 for base in base_names}

        # 在计算 rocket_plans 之后，进入发射循环之前插入逻辑：

        # 如果年份进入最后10年 (2131-2141)，或者积压量太大 (> 200万吨)
        is_panic_mode = (year > 2130) or (backlog > 2000000)

        if is_panic_mode:
            # 强制所有基地满负荷发射
            for base in base_names:
                # 直接把计划量设为该基地的理论上限
                max_possible = base_max_launches.get(base, 0) * ROCKET_NOMINAL_PAYLOAD
                rocket_plans[base] = max_possible

        # [NEW] 计算当年的火箭基准价格
        current_base_price_per_ton = get_rocket_price_per_ton(year)

        # B. 模拟太空电梯
        # 在电梯模拟部分加入“黑天鹅”事件
        if np.random.random() < 0.01: # 1%概率
            eff = 0.0 # 全年停运维修
        else:
            eff = np.random.normal(SE_EFFICIENCY_MEAN, SE_EFFICIENCY_STD)
        eff = np.clip(eff, 0.5, 1.0)
        se_actual = se_plan * eff
        se_cost = SE_FIXED_COST + (se_actual * SE_VAR_COST_PER_TON)
        
        # C. 模拟火箭发射
        rocket_mass_total = 0
        rocket_cost_total = 0
        remaining_backlog = backlog
        
        sorted_bases = sorted(base_names, key=lambda x: base_coeffs[x])
        
        for base in sorted_bases:
            mass_planned = rocket_plans.get(base, 0)
            # 规划时仍使用名义运力 (125t) 计算所需发射次数
            launches_planned = int(np.ceil(mass_planned / ROCKET_NOMINAL_PAYLOAD))
            max_launches = base_max_launches.get(base, 0)
            
            # 积压补偿逻辑
            launches_extra = 0
            if remaining_backlog > 0:
                spare_launches = max_launches - launches_planned
                if spare_launches > 0:
                    needed_launches = int(np.ceil(remaining_backlog / ROCKET_NOMINAL_PAYLOAD))
                    launches_extra = min(spare_launches, needed_launches)
                    remaining_backlog -= launches_extra * ROCKET_NOMINAL_PAYLOAD
            
            total_launches = launches_planned + launches_extra
            
            # 模拟成功次数
            successes = np.random.binomial(total_launches, 1 - FAIL_PROB_ROCKET)
            
            # [NEW] 运力波动模拟: 成功发射的运力在 [100, 150] 之间随机
            if successes > 0:
                # 生成 successes 个介于 100 到 150 之间的随机数并求和
                batch_payloads = np.random.uniform(100, 150, size=successes)
                actual_mass = np.sum(batch_payloads)
            else:
                actual_mass = 0
            
            # 成本计算: 
            # 假设发射成本按名义运力(125t)对应的火箭价格支付，不随实际载荷波动
            # 单次发射成本 = 125 * 当年基准单价 * 基地系数
            cost_per_launch = ROCKET_NOMINAL_PAYLOAD * current_base_price_per_ton * base_coeffs[base]
            actual_cost = total_launches * cost_per_launch
            
            rocket_mass_total += actual_mass
            rocket_cost_total += actual_cost

        # D. 更新状态
        total_mass_delivered = se_actual + rocket_mass_total
        cumulative_mass += total_mass_delivered
        total_cost += (se_cost + rocket_cost_total)
        
        target_demand = se_plan + sum(rocket_plans.values()) + backlog
        new_backlog = target_demand - total_mass_delivered
        backlog = max(0, new_backlog)
        
        year += 1
        if year > 2200: break
            
    return year - 1, total_cost

# -----------------------------------------------------------
# 3. 运行与绘图
# -----------------------------------------------------------
results_years = []
results_costs = []

if 'base_names' in locals() and len(base_names) > 0:
    print(f"Starting {SIMULATIONS} simulations with Dynamic Cost Model...")
    for i in range(SIMULATIONS):
        y, c = run_simulation()
        results_years.append(y)
        results_costs.append(c)

    df_sim = pd.DataFrame({'Completion_Year': results_years, 'Total_Cost': results_costs})

    # 输出统计
    print("\n--- Updated Simulation Results ---")
    print(df_sim.describe())
    
    # 绘图

    # -----------------------------------------------------------
    # 4. 论文级优化绘图代码 (针对 problem2_optimized_plot.png 的修复)
    # -----------------------------------------------------------
    print("Generating publication-quality plot...")

    # [设置]：配置学术风格 (模拟 LaTeX 字体效果)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.5,   # 边框加粗
        "grid.alpha": 0.5,       # 网格变淡
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # --- 左图：项目完成年份 (离散分布) ---
    ax1 = axes[0]
    
    # 获取唯一的年份和计数，手动绘制条形图以确保离散感
    unique_years, counts = np.unique(df_sim['Completion_Year'], return_counts=True)
    # 计算频率而不是绝对数量（更科学）
    freqs = counts / SIMULATIONS 
    
    # 绘制 Bar Chart 而不是 Histogram/KDE
    bars = ax1.bar(unique_years, freqs, color='#4A90E2', alpha=0.8, edgecolor='black', linewidth=1.2, width=0.6)
    
    # 标注参考线
    ax1.axvline(x=2141, color='#D0021B', linestyle='--', linewidth=2, label='Deadline (2141)')
    
    # 格式化轴
    ax1.set_xlabel('Completion Year')
    ax1.set_ylabel('Probability')
    ax1.set_title('(a) Distribution of Completion Year', loc='left', fontweight='bold')
    
    # 强制整数刻度
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0)) # Y轴显示百分比
    
    # 添加图例 (去框，更简洁)
    ax1.legend(frameon=False, loc='upper left')

    # 在Bar上方标注数值（如果是极少数几个年份，这很有用）
    for bar in bars:
        height = bar.get_height()
        if height > 0.05: # 只在柱子较高时标注，避免拥挤
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1%}', ha='center', va='bottom', fontsize=10)


    # --- 右图：总成本分布 (直方图替代小提琴图) ---
    ax2 = axes[1]
    
    # 使用直方图展示成本分布细节
    sns.histplot(df_sim['Total_Cost'], ax=ax2, 
                 bins=30,             # 增加分箱数以看到细节
                 stat='probability',  # 显示概率
                 color='#50E3C2', 
                 edgecolor='black', 
                 alpha=0.7,
                 kde=True,            # 这里可以用KDE，因为钱是连续变量
                 line_kws={'color': '#006B5F', 'linewidth': 2}) # KDE线颜色加深

    ax2.set_xlabel('Total Cost (USD)')
    ax2.set_ylabel('Probability')
    ax2.set_title('(b) Distribution of Total Cost', loc='left', fontweight='bold')

    # 科学计数法 / 单位格式化
    def trillion_fmt(x, pos):
        return f'${x/1e12:.2f}T'
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(trillion_fmt))
    
    # 标注均值和置信区间 (95% CI)
    mean_cost = df_sim['Total_Cost'].mean()
    ci_lower = np.percentile(df_sim['Total_Cost'], 2.5)
    ci_upper = np.percentile(df_sim['Total_Cost'], 97.5)
    
    # 绘制均值线
    ax2.axvline(mean_cost, color='k', linestyle='-', linewidth=1.5, label=f'Mean: ${mean_cost/1e12:.2f}T')
    # 绘制置信区间阴影
    ax2.axvspan(ci_lower, ci_upper, color='gray', alpha=0.15, label='95% CI')
    
    ax2.legend(frameon=False, loc='upper right')

    # --- 布局调整 ---
    plt.tight_layout()
    plt.subplots_adjust(top=0.9) # 留一点点头部空间，虽然去掉了大标题，但防止甚至切掉(a)(b)

    save_path = 'problem2_publication_ready.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{save_path}'")
    plt.show()
else:
    print("Data not loaded correctly, cannot run simulation.")