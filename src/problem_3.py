import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# ==========================================
# 1. 参数设置 (Parameters)
# ==========================================
# 基础设定
np.random.seed(101)  # 固定种子以复现结果
DAYS = 365
POPULATION = 100000
WATER_PER_CAPITA = 40  # L/day
RECYCLE_RATE = 0.80
NET_DEMAND_PER_PERSON = WATER_PER_CAPITA * (1 - RECYCLE_RATE)
DAILY_DEMAND_TONS = (POPULATION * NET_DEMAND_PER_PERSON) / 1000  # 100 tons

# 成本参数 (2145年)
COST_ROCKET = 261.0      # $/kg
COST_ELEVATOR = 60.0     # $/kg
COST_STORAGE = 0.50      # $/kg/day (高昂存储费)
COST_PENALTY = 2000.0   # $/kg (不可接受的缺水)

# 运输参数
LEAD_TIME_ROCKET = 3     # days
LEAD_TIME_ELEVATOR = 11  # days
ROCKET_CAPACITY = 125    # tons

# 波动性设定 (高波动 + 随机灾难)
VOLATILITY = 0.6         # 60% standard deviation
SPIKE_PROB = 0.02        # 2% chance of a massive spike (leak/accident)
SPIKE_MULTIPLIER = 3.0   # Spike is 3x normal demand

# ==========================================
# 2. 仿真引擎 (Simulation Engine)
# ==========================================
def generate_demand(days):
    """生成符合正态分布且带有尖峰的随机需求"""
    base = np.random.normal(DAILY_DEMAND_TONS, DAILY_DEMAND_TONS * VOLATILITY, days)
    # 添加尖峰
    spikes = np.random.rand(days) < SPIKE_PROB
    base[spikes] += DAILY_DEMAND_TONS * SPIKE_MULTIPLIER
    return np.maximum(base, 20) # 保证最小需求

# 全局需求序列（固定以便比较）
GLOBAL_DEMAND = generate_demand(DAYS)

def run_simulation(safety_stock, reorder_point):
    """
    运行一年的物流模拟
    输入: safety_stock (安全库存目标), reorder_point (火箭触发阈值)
    输出: 总成本, 库存历史, 火箭发射记录, 成本细分
    """
    inventory = safety_stock
    arrivals = {}  # 记录到达物资: {day: amount}
    
    total_cost_trans_ele = 0
    total_cost_trans_roc = 0
    total_cost_hold = 0
    total_cost_penalty = 0
    
    history_inv = []
    history_rocket_order = [] # 记录触发火箭的那一天订购了多少
    
    # --- 策略 A: 太空电梯 (Base Load) ---
    # 每天固定发送平均需求 (Push Strategy)
    # 初始化管道（假设前11天已经有货在路上）
    for d in range(DAYS + LEAD_TIME_ELEVATOR):
        arr_day = d + LEAD_TIME_ELEVATOR
        ship_amt = DAILY_DEMAND_TONS
        if arr_day not in arrivals: arrivals[arr_day] = 0
        arrivals[arr_day] += ship_amt
        # 仅计算仿真期内的发货成本
        if d < DAYS:
            total_cost_trans_ele += ship_amt * 1000 * COST_ELEVATOR

    # --- 每日循环 ---
    for day in range(DAYS):
        # 1. 物资到达 (Receive)
        if day in arrivals:
            inventory += arrivals[day]
        
        # 2. 策略 B: 火箭应急 (Emergency Pull)
        # 检查是否低于再订货点
        rocket_ordered_qty = 0
        if inventory < reorder_point:
            deficit = safety_stock - inventory
            if deficit > 0:
                # 计算需要多少枚火箭
                rockets_needed = int(np.ceil(deficit / ROCKET_CAPACITY))
                qty_tons = rockets_needed * ROCKET_CAPACITY
                
                # 安排到达
                arr_day = day + LEAD_TIME_ROCKET
                if arr_day not in arrivals: arrivals[arr_day] = 0
                arrivals[arr_day] += qty_tons
                
                # 记录成本
                cost = qty_tons * 1000 * COST_ROCKET
                total_cost_trans_roc += cost
                rocket_ordered_qty = qty_tons
        
        history_rocket_order.append(rocket_ordered_qty)
        
        # 3. 满足需求 (Fulfill Demand)
        demand = GLOBAL_DEMAND[day]
        if inventory >= demand:
            inventory -= demand
        else:
            shortage = demand - inventory
            inventory = 0
            total_cost_penalty += shortage * 1000 * COST_PENALTY
            
        # 4. 库存持有成本 (Holding Cost)
        total_cost_hold += inventory * 1000 * COST_STORAGE
        history_inv.append(inventory)
        
    total_cost = total_cost_trans_ele + total_cost_trans_roc + total_cost_hold + total_cost_penalty
    cost_breakdown = {
        'Elevator': total_cost_trans_ele, 
        'Rocket': total_cost_trans_roc, 
        'Storage': total_cost_hold, 
        'Penalty': total_cost_penalty
    }
    
    return total_cost, history_inv, history_rocket_order, cost_breakdown

# ==========================================
# 3. 优化与可视化 (Optimization & Plotting)
# ==========================================
# 设置绘图风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif' # 防止中文乱码需额外设置，此处用英文

# --- 步骤 A: 网格搜索优化 ---
print("Running Optimization Grid Search...")
ss_range = np.linspace(400, 2000, 20)  # 安全库存范围
rop_range = np.linspace(100, 1500, 20) # 再订货点范围

results_matrix = np.zeros((len(ss_range), len(rop_range)))
best_cost = float('inf')
best_params = (0, 0)

for i, ss in enumerate(ss_range):
    for j, rop in enumerate(rop_range):
        if rop >= ss: # 逻辑约束：再订货点不能高于安全库存太多(简单处理，设为高成本)
            results_matrix[i, j] = np.nan 
            continue
            
        c, _, _, _ = run_simulation(ss, rop)
        results_matrix[i, j] = c
        
        if c < best_cost:
            best_cost = c
            best_params = (ss, rop)

opt_ss, opt_rop = best_params
print(f"Optimal Strategy Found: Safety Stock={opt_ss:.1f}t, ROP={opt_rop:.1f}t")

# 运行最优方案以获取绘图数据
_, hist_inv, hist_roc, costs = run_simulation(opt_ss, opt_rop)

# --- 图表 1: 混合运输策略时序图 (Hybrid Strategy Time Series) ---
fig, ax1 = plt.subplots(figsize=(14, 7))

# 绘制库存 (左轴)
color_inv = '#2E86C1' # Strong Blue
ax1.plot(hist_inv, color=color_inv, linewidth=1.5, label='Water Inventory')
ax1.set_xlabel('Day of Year (2145)', fontsize=12)
ax1.set_ylabel('Inventory Level (Tons)', color=color_inv, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color_inv)
ax1.set_ylim(0, max(hist_inv)*1.2)

# 辅助线
ax1.axhline(y=opt_ss, color='green', linestyle='--', alpha=0.7, label=f'Safety Stock Target ({int(opt_ss)}t)')
ax1.axhline(y=opt_rop, color='#D35400', linestyle='--', alpha=0.7, label=f'Rocket Trigger Threshold ({int(opt_rop)}t)')

# 绘制火箭订单 (右轴)
ax2 = ax1.twinx()
color_roc = '#C0392B' # Dark Red
# 将0值过滤掉以便绘图更干净
roc_days = [i for i, x in enumerate(hist_roc) if x > 0]
roc_amts = [x for x in hist_roc if x > 0]
ax2.bar(roc_days, roc_amts, color=color_roc, alpha=0.6, width=4, label='Emergency Rocket Orders')
ax2.set_ylabel('Emergency Supply Quantity (Tons)', color=color_roc, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color_roc)
ax2.set_ylim(0, max(roc_amts)*3) # 留出空间

# 图例合并
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, fancybox=True, framealpha=0.9)

plt.title('Dual-Mode Inventory Control: Space Elevator Baseload + Rocket Response', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('Figure1_Inventory_Dynamics.png', dpi=300)
plt.show()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

# 1. 计算绘图的鲁棒阈值 (Robust Scaling)
# 找出矩阵中非空值的最小值
v_min = np.nanmin(results_matrix)
# 设置视觉上限为最小值的 5 倍，超过这个成本的策略在论文中统一视为“不可接受的高风险区”
# 这样可以强行拉开低成本区域的颜色差异
v_max = v_min * 5 

# 2. 准备绘图数据
plot_df = pd.DataFrame(results_matrix, 
                         index=np.round(ss_range, 0).astype(int), 
                         columns=np.round(rop_range, 0).astype(int))

plt.figure(figsize=(12, 9))

# 3. 绘制热力图
# cmap='viridis' 在深色背景下对比度更高，亮黄色代表低成本
# mask=pd.isnull(results_matrix) 确保不可行域不参与颜色映射
ax = sns.heatmap(plot_df, 
                 norm=LogNorm(vmin=v_min, vmax=v_max),
                 cmap='viridis', 
                 cbar_kws={'label': 'Annual Total Cost ($)', 'shrink': 0.8},
                 linewidths=0.05,
                 linecolor='#444444',
                 mask=pd.isnull(results_matrix))

# 4. 优化不可行域颜色 (强制区分)
# 使用浅灰色背景，使其与可行域（深蓝到黄）产生强烈视觉反差
ax.set_facecolor('#D3D3D3') 

# 5. 重新点亮星标 (使用高对比度颜色)
ss_idx = np.where(ss_range == opt_ss)[0][0]
rop_idx = np.where(rop_range == opt_rop)[0][0]
# 用鲜艳的红色或橙色，并增加白边，确保在亮黄色或深蓝色背景下都清晰
plt.scatter(rop_idx + 0.5, ss_idx + 0.5, s=400, marker='*', 
            color='#FF4500', edgecolors='white', linewidths=2, label='Global Optimum', zorder=5)

# 6. 视觉装饰
plt.title('Strategic Cost Optimization Landscape (Robust Scaling)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Reorder Point (Tons) - Rocket Trigger Threshold', fontsize=12)
plt.ylabel('Safety Stock (Tons) - Inventory Target', fontsize=12)

# 调整坐标轴刻度密度
ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
ax.yaxis.set_major_locator(ticker.MaxNLocator(10))
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('Figure2_Cost_Optimization_Heatmap.png', dpi=300)
plt.show()