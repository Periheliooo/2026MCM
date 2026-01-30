import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

# --- 1. 准备工作 ---
os.makedirs('images', exist_ok=True)

# 历史数据 (SpaceX 关键节点)
years_data = np.array([2006, 2010, 2018])
cost_data = np.array([13012.69, 3130.73, 1520.38])
C_MIN = 100  # 用户指定的新地板价

# --- 2. 定义幂律模型 ---
# Cost(t) = C_min + A * (t - 2005)^(-k)
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
print(f"Scenario B (C_min={C_MIN}) 预测结果")
print("="*40)
print(f"拟合公式: Cost(t) = 100 + {A_opt:.2f} * (t - 2005)^(-{k_opt:.4f})")
print("-" * 40)

for year in years_to_mark:
    cost = model_power(year, A_opt, k_opt)
    predictions[year] = cost
    print(f"{year}年: ${cost:.2f} / kg")
print("="*40)

# --- 5. 绘图 ---
plt.figure(figsize=(10, 6))

# 绘制历史数据
plt.scatter(years_data, cost_data, color='red', s=60, label='Historical Data', zorder=5)

# 绘制拟合曲线
t_range = np.linspace(2005.5, 2160, 500)
cost_fit = model_power(t_range, A_opt, k_opt)
plt.plot(t_range, cost_fit, 'b-', linewidth=2, label='Fitted Curve (Power Law)')

# 标记关键年份 (使用普通圆点)
for year in years_to_mark:
    val = predictions[year]
    plt.scatter(year, val, color='green', s=50, zorder=6) # 普通圆点
    # 添加注释 (年份+价格)
    plt.annotate(f'{year}\n${val:.0f}', 
                 xy=(year, val), 
                 xytext=(0, 15), 
                 textcoords='offset points',
                 ha='center', fontsize=9,
                 arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

# 地板价参考线
plt.axhline(C_MIN, color='gray', linestyle='--', alpha=0.6, label=f'Floor Price (${C_MIN})')

plt.title(f'Rocket Launch Cost Prediction (C_min=${C_MIN})', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Cost ($/kg)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.4)

# 保存
save_path = 'images/rocket_cost_decay_cmin100.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"图表已保存至: {save_path}")