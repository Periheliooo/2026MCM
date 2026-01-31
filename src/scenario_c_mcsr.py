import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 读取修正后的帕累托数据
try:
    # 假设你使用的是之前生成的 excel 文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pareto_path = os.path.join(current_dir, '..', 'data', 'scenario_c_pareto_corrected.xlsx')
    df = pd.read_excel(pareto_path)
    # df = pd.read_excel('scenario_c_pareto_corrected.xlsx')
except:
    # 或者读取 csv 版本
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pareto_csv_path = os.path.join(current_dir, '..', 'data', 'scenario_c_pareto_corrected.csv')
    df = pd.read_csv(pareto_csv_path)
    # df = pd.read_csv('scenario_c_pareto_corrected.xlsx - Sheet1.csv')

# 确保按年份排序
df = df.sort_values(by='Target_Completion_Year')

# 1. 采样: 每隔5个点取一个 (步长=5)
# 这样可以去除一些微小的波动，更清晰地看到大趋势
df_sampled = df.iloc[::5].copy().reset_index(drop=True)

# 2. 计算边际成本节省率 (MCSR)
# MCSR = (本期成本 - 下期成本) / (下期年份 - 本期年份)
# diff() 计算的是 (本行 - 上一行)，所以我们要 shift(-1) 来计算 (下一行 - 本行)
delta_cost = df_sampled['Total_Cost_Billion'].diff().shift(-1) # C_{t+1} - C_t (这是负值)
delta_time = df_sampled['Target_Completion_Year'].diff().shift(-1) # T_{t+1} - T_t (这是正值 5)

# 因为成本是下降的，我们要看节省了多少，所以取负号
df_sampled['MCSR'] = - delta_cost / delta_time

# 3. 保存结果
current_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(current_dir, '..', 'data', 'scenario_c_mcsr.xlsx')
df_sampled.to_excel(output_file, index=False)
print(f"MCSR计算完成，已保存至: {output_file}")
print(df_sampled[['Target_Completion_Year', 'Total_Cost_Billion', 'MCSR']].head(10))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# --- 1. 设置全局风格 ---
# 使用 Seaborn 的 whitegrid 风格，背景干净，网格线柔和
sns.set_theme(style="white", context="talk") 
plt.rcParams['font.family'] = 'sans-serif' # 建议使用无衬线字体，如 Arial, Helvetica, DejaVu Sans
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# --- 2. 创建画布 ---
fig, ax = plt.subplots(figsize=(12, 7))

# 数据准备 (为了方便演示，直接使用你代码中的 df_sampled)
x = df_sampled['Target_Completion_Year'][:-1]
y = df_sampled['MCSR'][:-1]

# --- 3. 绘制主图 (面积图风格) ---
# 使用 fill_between 创建渐变效果或半透明填充，比单纯折线更有质感
color_line = "#005a8d" # 深蓝色主线
color_fill = "#0077b6" # 填充色

# 绘制线条
ax.plot(x, y, color=color_line, linewidth=2.5, linestyle='-', zorder=10)

# 绘制填充 (添加透明度 alpha)
ax.fill_between(x, y, color=color_fill, alpha=0.15)

# 绘制数据点 (只绘制部分关键点，或者把点变小)
# 这里我们每隔一个点画一个，避免拥挤，且设计为空心圆
ax.scatter(x, y, color='white', edgecolor=color_line, s=60, lw=2, zorder=11, label='Sampled MCSR')

# --- 4. 视觉优化 (去噪) ---
# 去除上方和右侧的边框 (Spines)
sns.despine(top=True, right=True, left=False, bottom=False)

# 设置网格线 (只保留横向网格，且设为虚线，颜色浅)
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)
ax.xaxis.grid(False) # 去掉纵向网格，减少干扰

# --- 5. 坐标轴与标签美化 ---
ax.set_title('Marginal Cost Saving Rate Analysis', fontsize=20, fontweight='bold', pad=20, loc='center')
ax.set_xlabel('Target Completion Year', fontsize=14, labelpad=10, color='#333333')
ax.set_ylabel('Cost Saving Rate (Billion $ / Year)', fontsize=14, labelpad=10, color='#333333')

# 格式化 Y 轴：加上 $ 符号
formatter = ticker.FormatStrFormatter('$%.0f')
ax.yaxis.set_major_formatter(formatter)

# --- 6. 关键点标注 (美化版) ---
# 颜色与主线保持一致，视觉上更和谐
text_color = color_line 

# A. 标注最高点 (起点)
# 不用箭头，改用垂直虚线引导视线，更专业
ymax = y.max()
xmax = x[y.idxmax()]

# 画一条细虚线垂直落地
ax.vlines(x=xmax, ymin=0, ymax=ymax, color=text_color, linestyle=':', alpha=0.5, linewidth=1)

# 文字直接写在点旁边，加粗数值，弱化描述
ax.text(xmax + 5, ymax, 
        f'Peak Savings\n', 
        fontsize=12, color='gray', va='bottom', ha='left') # 描述文字灰色
ax.text(xmax + 5, ymax, 
        f'${ymax:.0f} B/Yr', 
        fontsize=16, fontweight='bold', color=text_color, va='top', ha='left') # 数值文字加粗且用主色

# B. 标注末尾点
# 直接放在点的右侧或上方，取消箭头
ymin = y.iloc[-1]
xmin = x.iloc[-1]

ax.text(xmin, ymin + 15, 
        f'Stabilized at ${ymin:.0f} B/Yr', 
        fontsize=12, fontweight='bold', color=text_color, ha='center')

# --- 7. 保存与显示 ---
plt.tight_layout()
save_path = os.path.join(current_dir, '..', 'images', 'mcsr_plot_improved.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight') # dpi=300 保证高清
print(f"美化图表已保存至: {save_path}")
plt.show()