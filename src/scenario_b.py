import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 参数设置 ---
TOTAL_DEMAND = 100_000_000  # 总需求: 1亿公吨
PAYLOAD_PER_LAUNCH = 125    # 单次载荷: 125公吨
START_YEAR = 2050           # 开始年份

# 成本模型参数 (来自之前的拟合结果)
C_MIN = 100
A_OPT = 12906.93
K_OPT = 0.8872

# --- 2. 定义成本函数 ---
def get_base_cost_per_kg(year):
    """根据幂律模型计算当年的基础成本 ($/kg)"""
    # 避免年份小于2005导致计算错误 (虽然我们从2050开始)
    if year <= 2005: return 10000 
    return C_MIN + A_OPT * np.power(year - 2005, -K_OPT)

# --- 3. 读取基地数据 ---
os.makedirs('data', exist_ok=True)
file_path = 'data/harbor.xlsx'
try:
    harbor_df = pd.read_excel(file_path)
    # 清理列名 (去除可能存在的空格)
    harbor_df.columns = [c.strip() for c in harbor_df.columns]
    
    # 提取关键列: "2050年最大年发射次数", "2050年单次成本系数"
    # 假设列名固定，如果不一样请根据打印出的 info 修改
    col_launches = '2050年最大年发射次数'
    col_coeff = '2050年单次成本系数'
    
    print("基地数据加载成功:")
    print(harbor_df[['基地名称', col_launches, col_coeff]])
    
except Exception as e:
    print(f"错误: 无法读取基地数据文件 ({e})")
    exit()

# --- 4. 开始模拟 ---
current_year = START_YEAR
remaining_material = TOTAL_DEMAND
total_cost_accumulated = 0
history = []

print("\n开始模拟任务 (Scenario B)...")

while remaining_material > 0:
    # 4.1 计算当年基础成本
    base_cost_kg = get_base_cost_per_kg(current_year)
    
    # 4.2 计算当年全系统的总运力和总成本
    # 当年运力 = 所有基地发射次数之和 * 单次载荷
    # 当年成本 = Sum(某基地发射次数 * 单次载荷(kg) * 该基地成本系数 * 基础成本)
    
    yearly_transported = 0
    yearly_cost = 0
    
    for index, row in harbor_df.iterrows():
        launches = row[col_launches]
        coeff = row[col_coeff]
        
        # 该基地的运力 (公吨)
        site_capacity = launches * PAYLOAD_PER_LAUNCH
        
        # 该基地的成本 (美元)
        # 成本 = 运量(kg) * 单价($/kg) * 系数
        site_cost = (site_capacity * 1000) * base_cost_kg * coeff
        
        yearly_transported += site_capacity
        yearly_cost += site_cost
    
    # 4.3 扣除运量
    # 如果剩余量小于当年运力，则按比例计算成本（最后一年）
    if remaining_material <= yearly_transported:
        ratio = remaining_material / yearly_transported
        actual_transported = remaining_material
        actual_cost = yearly_cost * ratio
        remaining_material = 0
    else:
        actual_transported = yearly_transported
        actual_cost = yearly_cost
        remaining_material -= actual_transported
    
    total_cost_accumulated += actual_cost
    
    # 4.4 记录数据
    history.append({
        'Year': current_year,
        'Base_Cost_Per_Kg': base_cost_kg,
        'Transported_Mass_Tons': actual_transported,
        'Yearly_Cost_Billions': actual_cost / 1e9,
        'Cumulative_Cost_Billions': total_cost_accumulated / 1e9,
        'Remaining_Mass_Tons': remaining_material
    })
    
    current_year += 1

# --- 5. 结果整理与输出 ---
results_df = pd.DataFrame(history)

total_years = current_year - START_YEAR
avg_annual_cost = total_cost_accumulated / total_years
final_year = current_year - 1

print("="*50)
print("Scenario B (Rocket Only) 模拟结果")
print("="*50)
print(f"总耗时: {total_years} 年 (完成年份: {final_year})")
print(f"总成本: ${total_cost_accumulated/1e9:.2f} Billion (十亿美元)")
print(f"年均成本: ${avg_annual_cost/1e9:.2f} Billion")
print(f"平均每吨成本: ${total_cost_accumulated/TOTAL_DEMAND:.2f}")
print("-" * 50)
print(f"系统年总运力: {results_df['Transported_Mass_Tons'].max():,.0f} 公吨/年")
print("="*50)

# --- 6. 保存与绘图 ---
os.makedirs('data', exist_ok=True)
os.makedirs('images', exist_ok=True)

# 保存表格
results_df.to_excel('data/scenario_b_simulation.xlsx', index=False)
print("详细数据已保存至 data/scenario_b_simulation.xlsx")

# 绘图: 剩余物资 vs 累计成本
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('Remaining Material (Million Tons)', color=color)
ax1.plot(results_df['Year'], results_df['Remaining_Mass_Tons']/1e6, color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Cumulative Cost (Billion $)', color=color)
ax2.plot(results_df['Year'], results_df['Cumulative_Cost_Billions'], color=color, linestyle='--', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Scenario B: Rocket Transport Progress & Cost (Fixed Capacity)')
plt.savefig('images/scenario_b_simulation.png', dpi=300)
print("图表已保存至 images/scenario_b_simulation.png")