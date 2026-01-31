import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# --- 1. 准备工作 ---
os.makedirs('images', exist_ok=True)

# 历史数据 (SpaceX 关键节点)
years_data = np.array([2006, 2010, 2018])
cost_data = np.array([13012.69, 3130.73, 1520.38])
C_MIN = 100  # 用户指定的新地板价

# --- 2. 定义幂律模型 ---
def model_power(t, A, k):
    return C_MIN + A * np.power(t - 2005, -k)

# --- 3. 拟合 ---
p0 = [13000, 0.9] 
popt, pcov = curve_fit(model_power, years_data, cost_data, p0=p0)
A_opt, k_opt = popt

# --- 4. 关键年份预测 ---
years_to_mark = [2050, 2100, 2150]
predictions = {}
print("="*40)
print(f"拟合公式: Cost(t) = 100 + {A_opt:.2f} * (t - 2005)^(-{k_opt:.4f})")
for year in years_to_mark:
    cost = model_power(year, A_opt, k_opt)
    predictions[year] = cost
    print(f"{year}年: ${cost:.2f} / kg")

# --- 5. 美化绘图 (对数坐标版) ---
# 设置风格
plt.style.use('seaborn-v0_8-whitegrid')

# 创建画布
fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

# 定义配色
COLOR_HISTORICAL = '#D64045'  # 砖红色
COLOR_PREDICT = '#1B3B6F'     # 深海军蓝
COLOR_TEXT = '#333333'        # 深灰字体
COLOR_FLOOR = '#21295C'       # 深紫色

# 1. 绘制拟合曲线
t_range = np.linspace(2005.5, 2160, 1000)
cost_fit = model_power(t_range, A_opt, k_opt)

# 核心修改：使用 set_yscale('log')
ax.plot(t_range, cost_fit, color=COLOR_PREDICT, linewidth=2.5, 
        label='Predicted Cost Curve (Power Law)', zorder=2)
ax.set_yscale('log') 

# 2. 添加填充效果
ax.fill_between(t_range, cost_fit, C_MIN, color=COLOR_PREDICT, alpha=0.1)

# 3. 绘制地板价
ax.axhline(C_MIN, color=COLOR_FLOOR, linestyle='--', linewidth=1, alpha=0.7, 
           label=f'Physical Limit / Floor (${C_MIN})', zorder=1)

# 4. 绘制历史数据
ax.scatter(years_data, cost_data, color=COLOR_HISTORICAL, s=100, 
           edgecolor='white', linewidth=1.5, zorder=5, label='Historical (SpaceX Data)')

# 为历史数据添加标签
rocket_names = ["Falcon 1", "Falcon 9\n(v1.0)", "Falcon Heavy"]
for i, txt in enumerate(rocket_names):
    ax.annotate(txt, (years_data[i], cost_data[i]), 
                xytext=(12, 0), textcoords='offset points', 
                fontsize=9, color=COLOR_HISTORICAL, fontweight='bold', va='center')

# 5. 标记未来预测点
for year in years_to_mark:
    val = predictions[year]
    ax.scatter(year, val, color=COLOR_PREDICT, s=80, edgecolor='white', linewidth=1.5, zorder=5)
    # 引线 (vlines)
    ax.vlines(year, C_MIN, val, color=COLOR_PREDICT, linestyle=':', alpha=0.4)
    
    # 标注框 (放在点下方，避免挡住曲线)
    label_text = f"{year}\n${val:.0f}"
    ax.annotate(label_text, 
                xy=(year, val), 
                xytext=(0, -30), # 向下偏移
                textcoords='offset points',
                ha='center', va='top',
                fontsize=9, fontweight='bold', color='white',
                bbox=dict(boxstyle="round,pad=0.4", fc=COLOR_PREDICT, ec="none", alpha=0.8))

# --- 关键修改：自定义 Y 轴刻度 (让对数轴看起来像普通轴) ---
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
# 手动设定刻度，覆盖默认的 10^n
custom_ticks = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
ax.set_yticks(custom_ticks)
ax.get_yaxis().set_minor_formatter(ticker.NullFormatter()) # 隐藏不需要的小刻度数字

# 标题与调整
ax.set_title(f'Rocket Launch Cost Prediction (Log Scale)', 
             fontsize=16, fontweight='bold', color=COLOR_TEXT, pad=20)
ax.set_xlabel('Year', fontsize=12, fontweight='bold', color=COLOR_TEXT)
ax.set_ylabel('Launch Cost ($/kg) - Logarithmic Scale', fontsize=12, fontweight='bold', color=COLOR_TEXT)

# 范围调整
ax.set_xlim(2000, 2170)
ax.set_ylim(80, 25000) # 稍微留一点余地给 $100 以下，看着不挤

# 杂项
ax.grid(True, linestyle='-', alpha=0.3, which='major') # 只显示主要网格
ax.legend(frameon=False, loc='upper right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

save_path = 'images/rocket_cost_decay_log.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Modified chart saved to: {save_path}")