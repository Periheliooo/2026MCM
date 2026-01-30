import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 初始化目录
# ==========================================
os.makedirs("data", exist_ok=True)
os.makedirs("images", exist_ok=True)

# ==========================================
# 2. 参数设置
# ==========================================
M_TOTAL = 1e8             # 总目标: 1亿吨
SE_COST_VAR = 60000       # 电梯可变成本: 60,000 $/ton
SE_COST_MAINT = 4.12e9    # 电梯维护成本: 4.12 Billion $/year
SE_CAPACITY = 100000000 / 186.22  # 电梯年运力: ~53.7万吨

# 火箭参数
ROCKET_PAYLOAD = 125      # 单枚火箭载量 (吨)
TOTAL_LAUNCHES = 6250     # 2050年总发射次数 (1500+600+1200+...)
ROCKET_CAPACITY = TOTAL_LAUNCHES * ROCKET_PAYLOAD  # 火箭年运力

print(f"参数确认: 电梯年运力 {SE_CAPACITY:.2f} 吨, 火箭年运力 {ROCKET_CAPACITY:.2f} 吨")

# 火箭成本函数 (基于Scenario B拟合结果)
a_fit = 698.0736
b_fit = -0.2003
def get_rocket_cost(year):
    if year <= 2049: return 1e9
    return (a_fit * np.power(year - 2049, b_fit)) * 1000  # $/ton

# ==========================================
# 3. 帕累托计算与序列生成
# ==========================================
min_years = M_TOTAL / (SE_CAPACITY + ROCKET_CAPACITY)
start_T = 2050 + int(np.ceil(min_years))
end_T = 2236 

pareto_list = []
seq_list = []

print(f"正在计算 {start_T} 到 {end_T} 年的方案...")

for T in range(start_T, end_T + 1):
    yrs = np.arange(2051, T + 1)
    ny = len(yrs)
    
    # 成本计算
    c_m = ny * SE_COST_MAINT
    
    # 电梯分配
    tot_se = ny * SE_CAPACITY
    m_se = min(M_TOTAL, tot_se)
    c_se = m_se * SE_COST_VAR
    x_val = m_se / ny
    x_s = np.full(ny, x_val)
    
    # 火箭分配
    m_rem = M_TOTAL - m_se
    y_s = np.zeros(ny)
    c_roc = 0
    
    if m_rem > 0:
        if m_rem > ny * ROCKET_CAPACITY: continue
        
        r_costs = [get_rocket_cost(y) for y in yrs]
        m_need = m_rem
        # 倒序分配 (优先用后期便宜的火箭)
        for i in range(ny - 1, -1, -1):
            fill = min(m_need, ROCKET_CAPACITY)
            y_s[i] = fill
            c_roc += fill * r_costs[i]
            m_need -= fill
            if m_need <= 1e-9: break
            
    tot_c = c_m + c_se + c_roc
    
    # 记录概要结果
    pareto_list.append({
        'Completion_Year': T,
        'Duration': ny,
        'Total_Cost_Billion': tot_c / 1e9,
    })
    
    # 记录详细序列 (用于导出Excel)
    tmp_df = pd.DataFrame({
        'Target_Completion_Year': T,
        'Year': yrs,
        'x_t': x_s,
        'y_t': y_s
    })
    seq_list.append(tmp_df)

# ==========================================
# 4. 导出文件与绘图
# ==========================================
# 合并所有序列
df_seq = pd.concat(seq_list, ignore_index=True)
df_pareto = pd.DataFrame(pareto_list)

# 导出 Excel
seq_path = "data/scenario_c_sequences.xlsx"
summary_path = "data/scenario_c_pareto_summary.xlsx"

df_seq.to_excel(seq_path, index=False)
df_pareto.to_excel(summary_path, index=False)
print(f"数据已导出至: {seq_path}")

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(df_pareto['Completion_Year'], df_pareto['Total_Cost_Billion'])
plt.scatter(df_pareto['Completion_Year'], df_pareto['Total_Cost_Billion'], s=10, c='red')
plt.title('Scenario C: Pareto Front (Time vs Cost)')
plt.xlabel('Completion Year')
plt.ylabel('Total Cost (Billion $)')
plt.grid(True)

img_path = "images/scenario_c_pareto_front.png"
plt.savefig(img_path)
print(f"图表已保存至: {img_path}")