import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os

# ==========================================
# 1. 环境与数据读取
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
pareto_path = os.path.join(current_dir, '..', 'data', 'scenario_c_pareto_corrected.xlsx')
detailed_path = os.path.join(current_dir, '..', 'data', 'scenario_c_sequences_detailed.xlsx')
img_save_path = os.path.join(current_dir, '..', 'images', 'scenario_c_pareto_final_v2.png')

try:
    pareto_df = pd.read_excel(pareto_path)
    detailed_df = pd.read_excel(detailed_path)
except FileNotFoundError:
    print(f"错误：找不到文件。请确保先运行了 scenario_c_corrected.py 生成数据。")
    exit()

# ==========================================
# 2. 数据处理：计算电梯运输比例
# ==========================================
# 按目标年份分组求和
grouped_transport = detailed_df.groupby('Target_Completion_Year')[['x_t', 'y_t']].sum().reset_index()
grouped_transport['Total_Mass'] = grouped_transport['x_t'] + grouped_transport['y_t']
grouped_transport['Elevator_Ratio'] = grouped_transport['x_t'] / grouped_transport['Total_Mass']

# 合并回主表
merged_df = pd.merge(pareto_df, grouped_transport[['Target_Completion_Year', 'Elevator_Ratio']], 
                     on='Target_Completion_Year', how='left')
merged_df = merged_df.sort_values(by='Target_Completion_Year')

# 采样 (保留首尾，中间稀疏)
sampled_df = merged_df.iloc[::5].copy()
if merged_df.iloc[-1]['Target_Completion_Year'] != sampled_df.iloc[-1]['Target_Completion_Year']:
    sampled_df = pd.concat([sampled_df, merged_df.iloc[[-1]]])

# ==========================================
# 3. 绘图设置
# ==========================================
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 8)) # 稍微加高一点图表以容纳高成本点

# 定义颜色映射
cmap = plt.cm.Spectral

# --- A. 绘制背景连线 (Pareto Front) ---
ax.plot(sampled_df['Target_Completion_Year'], sampled_df['Total_Cost_Billion'], 
        color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

# --- B. 绘制中间混合点 (圆形) ---
# 排除最后一个点(三角形)
middle_points = sampled_df.iloc[:-1]
sc = ax.scatter(middle_points['Target_Completion_Year'], middle_points['Total_Cost_Billion'], 
                c=middle_points['Elevator_Ratio'], cmap=cmap, vmin=0, vmax=1,
                s=70, marker='o', edgecolor='k', alpha=0.9, zorder=2, label='Mixed Strategy')

# --- C. 绘制最右侧点：纯电梯/电梯主导 (三角形) ---
right_point = sampled_df.iloc[-1]
ax.scatter(right_point['Target_Completion_Year'], right_point['Total_Cost_Billion'],
           c=[right_point['Elevator_Ratio']], cmap=cmap, vmin=0, vmax=1,
           s=120, marker='^', edgecolor='k', zorder=3, label='Elevator Dominant')

# --- D. 绘制最上方点：纯火箭基准 (方块) ---
# 用户指定数据: 2177年, 31200 Billion
pure_rocket_year = 2177
pure_rocket_cost = 31200
pure_rocket_ratio = 0.0 # 纯火箭意味着电梯比例为0

ax.scatter(pure_rocket_year, pure_rocket_cost,
           c=[pure_rocket_ratio], cmap=cmap, vmin=0, vmax=1,
           s=100, marker='s', edgecolor='k', zorder=3, label='Pure Rocket (Baseline)')

# --- E. 添加 Colorbar ---
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Elevator Transport Ratio (Mass %)', rotation=270, labelpad=20, fontsize=11)
cbar.set_ticks(np.linspace(0, 1, 6))
cbar.ax.set_yticklabels(['0% (Rocket)', '20%', '40%', '60%', '80%', '100% (Elevator)'])

# ==========================================
# 4. 标注文字 (Annotation)
# ==========================================

# 1. 标注最左侧混合点 (Rocket Dominant Mix) - 位置调整到右下方以避免遮挡标题
left_point = sampled_df.iloc[0]
ax.annotate(f"Fastest Mixed\n(Rocket Heavy)\nRatio: {left_point['Elevator_Ratio']:.1%}",
            xy=(left_point['Target_Completion_Year'], left_point['Total_Cost_Billion']),
            xytext=(20, -40), textcoords='offset points', # 向右下方偏移
            arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle="arc3,rad=.2"),
            fontsize=10, fontweight='bold', color='#333333')

# 2. 标注最右侧点 (Elevator Dominant) - 三角形
ax.annotate(f"Lowest Cost\n(Elevator Dominant)\nRatio: {right_point['Elevator_Ratio']:.1%}",
            xy=(right_point['Target_Completion_Year'], right_point['Total_Cost_Billion']),
            xytext=(-10, 30), textcoords='offset points', # 向上方偏移
            arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle="arc3,rad=.2"),
            fontsize=10, fontweight='bold', color='#333333', ha='center')

# 3. 标注纯火箭点 (Pure Rocket) - 方块
ax.annotate(f"Pure Rocket Baseline\nYear: {pure_rocket_year}\nCost: ${pure_rocket_cost:,.0f}B",
            xy=(pure_rocket_year, pure_rocket_cost),
            xytext=(-40, -50), textcoords='offset points', # 向左下方偏移
            arrowprops=dict(facecolor='darkred', arrowstyle='->'),
            fontsize=10, fontweight='bold', color='darkred')

# ==========================================
# 5. 图表修饰
# ==========================================
ax.set_title('Scenario C: Optimization vs. Pure Rocket Baseline', fontsize=15, fontweight='bold', pad=20)
ax.set_xlabel('Completion Year', fontsize=12)
ax.set_ylabel('Total Cost (Billion $)', fontsize=12)

# 设置货币格式
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

# 网格线
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# 图例 (手动控制图例以显示形状)
# 创建自定义图例句柄
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='s', color='w', label='Pure Rocket', markerfacecolor=cmap(0.0), markersize=10, markeredgecolor='k'),
    Line2D([0], [0], marker='o', color='w', label='Mixed Strategy', markerfacecolor=cmap(0.5), markersize=10, markeredgecolor='k'),
    Line2D([0], [0], marker='^', color='w', label='Pure Elevator', markerfacecolor=cmap(1.0), markersize=10, markeredgecolor='k')
]
ax.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.9)

plt.tight_layout()
plt.savefig(img_save_path, dpi=300)
print(f"最终图表已保存至: {img_save_path}")
plt.show()