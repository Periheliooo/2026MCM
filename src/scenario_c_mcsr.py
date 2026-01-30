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

# 4. 绘制折线图
plt.figure(figsize=(10, 6))
# 切片 [:-1] 是因为最后一个点的 MCSR 是 NaN (没有下一个点来计算)
plt.plot(df_sampled['Target_Completion_Year'][:-1], df_sampled['MCSR'][:-1], 
         marker='o', linestyle='-', color='b', label='MCSR')

plt.title('Marginal Cost Saving Rate (MCSR) Analysis')
plt.xlabel('Completion Year')
plt.ylabel('Cost Saving Rate (Billion $ / Year)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.savefig(os.path.join(current_dir, '..', 'images', 'mcsr_plot.png'))
print("图表已保存至: mcsr_plot.png")
plt.show()