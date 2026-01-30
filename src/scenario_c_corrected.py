import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 获取当前脚本(scenario_c_optimized.py) 所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接路径：当前目录(src) -> 上一级(..) -> data -> 文件名
harbor_path = os.path.join(current_dir, '..', 'data', 'harbor.xlsx')
sequences_path = os.path.join(current_dir, '..', 'data', 'scenario_c_sequences.xlsx')

try:
    harbor_df = pd.read_excel(harbor_path)
    sequences_df = pd.read_excel(sequences_path)
except FileNotFoundError:
    print(f"错误：找不到文件。尝试读取的路径是：\n{harbor_path}")
    exit()

# ==========================================
# 2. 处理基地数据 (Harbor)
# ==========================================
# 计算每个基地的年最大运载能力: 年发射次数 * 单次载量(125吨)
ROCKET_PAYLOAD = 125
harbor_df['Annual_Capacity'] = harbor_df['2050年最大年发射次数'] * ROCKET_PAYLOAD

# 按成本系数从小到大排序 (贪心算法的关键)
harbor_df = harbor_df.sort_values(by='2050年单次成本系数', ascending=True)

# 提取排序后的基地信息
bases = harbor_df['基地名称'].values
base_caps = harbor_df['Annual_Capacity'].values
base_coeffs = harbor_df['2050年单次成本系数'].values

print("基地优先级排序 (成本系数低 -> 高):")
print(harbor_df[['基地名称', '2050年单次成本系数', 'Annual_Capacity']])

# ==========================================
# 3. 定义成本函数
# ==========================================
SE_COST_VAR = 60000       # 电梯可变成本: 6w/吨
SE_COST_MAINT = 4.12e9    # 电梯维护成本: 4.12 Billion/年

# 火箭基准价格曲线 (基于之前的拟合参数)
a_fit = 698.0736
b_fit = -0.2003
def get_base_rocket_price(year):
    # 返回 $/ton
    if year <= 2049: return 1e9
    return (a_fit * np.power(year - 2049, b_fit)) * 1000

# ==========================================
# 4. 贪心分配与成本计算
# ==========================================
# 这是一个逐行处理函数
def process_row(row):
    # 获取该年的总需求
    y_t = row['y_t']
    x_t = row['x_t']
    year = row['Year']
    
    # 1. 计算电梯成本 (固定单价)
    se_cost = x_t * SE_COST_VAR
    
    # 2. 计算火箭成本 (贪心分配)
    base_price = get_base_rocket_price(year)
    remaining_y = y_t
    rocket_cost = 0
    
    # 初始化该行的基地分配数据
    allocations = {base: 0.0 for base in bases}
    
    # 遍历基地 (已按成本排序)
    for i, base_name in enumerate(bases):
        if remaining_y <= 0:
            break
            
        cap = base_caps[i]
        coeff = base_coeffs[i]
        
        # 分配量 = min(剩余需求, 该基地容量)
        take = min(remaining_y, cap)
        
        allocations[base_name] = take
        
        # 累加成本: 量 * 基准价 * 系数
        rocket_cost += take * base_price * coeff
        
        remaining_y -= take
    
    # 将结果打包返回
    # 包含: SE成本, Rocket总成本, 各基地分配量...
    return pd.Series([se_cost, rocket_cost] + [allocations[b] for b in bases])

# 应用函数到每一行
print("正在计算每年的基地分配与成本...")
new_columns = ['Annual_SE_Cost', 'Annual_Rocket_Cost'] + list(bases)
sequences_df[new_columns] = sequences_df.apply(process_row, axis=1)

# ==========================================
# 5. 汇总生成新的帕累托数据
# ==========================================
print("正在汇总帕累托前沿数据...")
summary_data = []

# 按目标完成年份分组汇总
grouped = sequences_df.groupby('Target_Completion_Year')

for target_year, group in grouped:
    duration = len(group) # 持续时长
    
    # 汇总各项成本
    total_se_var = group['Annual_SE_Cost'].sum()
    total_rocket = group['Annual_Rocket_Cost'].sum()
    total_maint = duration * SE_COST_MAINT # 维护成本
    
    grand_total = total_se_var + total_rocket + total_maint
    
    summary_data.append({
        'Target_Completion_Year': target_year,
        'Duration': duration,
        'Total_Cost_Billion': grand_total / 1e9
    })

pareto_new_df = pd.DataFrame(summary_data)

# ==========================================
# 6. 保存文件与绘图
# ==========================================
# 保存详细分配表
sequences_df.to_excel(os.path.join(current_dir, '..', 'data', 'scenario_c_sequences_detailed.xlsx'), index=False)
print("已保存详细分配表: scenario_c_sequences_detailed.xlsx")

# 保存帕累托汇总表
pareto_new_df.to_excel(os.path.join(current_dir, '..', 'data', 'scenario_c_pareto_corrected.xlsx'), index=False)
print("已保存帕累托汇总表: scenario_c_pareto_corrected.xlsx")

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(pareto_new_df['Target_Completion_Year'], pareto_new_df['Total_Cost_Billion'], linewidth=2, label='Corrected Cost')
plt.scatter(pareto_new_df['Target_Completion_Year'], pareto_new_df['Total_Cost_Billion'], s=15, c='red')

plt.title('Scenario C: Corrected Pareto Front (With Base Costs)')
plt.xlabel('Completion Year')
plt.ylabel('Total Cost (Billion $)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.savefig(os.path.join(current_dir, '..', 'images', 'scenario_c_pareto_corrected.png'))
print("已保存图表: scenario_c_pareto_corrected.png")
plt.show()