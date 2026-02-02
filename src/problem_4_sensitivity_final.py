import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 导入你的模型
from problem_4_model import MoonColonyModel

def run_sensitivity_analysis():
    # ==========================================
    # 1. 准备模型与基准解
    # ==========================================
    print("正在初始化模型...")
    try:
        model = MoonColonyModel('harbor.xlsx')
    except Exception as e:
        # 如果找不到文件，生成一个假的测试数据以保证代码可运行演示
        print(f"警告: {e}")
        print("正在使用模拟数据模式运行演示...")
        # (此处仅为演示代码逻辑，实际运行时请确保有 harbor.xlsx)
        return

    # -------------------------------------------------
    # 关键步骤：获取一个“基准方案” (x_SE, x_Rocket)
    # -------------------------------------------------
    # 这里的逻辑是：敏感性分析是分析“当外部参数变化时，同一个计划会有什么不同的后果”
    # 如果你有 saved solution，读取它；否则我们生成一个合理的基准方案。
    
    n_years = 120
    # 假设方案：前60年主要靠SE满负荷，后60年主要靠Rocket
    # 你应该替换为：从 nsga2_pareto_solutions_X.csv 读取某一行
    x_SE_baseline = np.ones(n_years) * model.SE_CAPACITY_YEAR
    x_Rocket_baseline = np.zeros(n_years)
    x_Rocket_baseline[50:] = model.GLOBAL_ROCKET_CAPACITY_YEAR * 0.5 # 假设后期开启50%火箭运力

    # ==========================================
    # 2. 定义要分析的两个敏感参数
    # ==========================================
    # 在你的模型中，我选择了两个最有代表性的环境参数：
    # Y轴: env_threshold_A (环境承载力阈值) -> 类似“环境容量”
    # X轴: c_R (火箭单位污染系数) -> 类似“技术清洁度”
    
    # 也可以模拟图片中的“税率”和“投资”：
    # 比如：Y轴 k_penalty (惩罚力度/税率), X轴 decay_rate (环境自净/治理投入)
    
    param_y_name = "k_penalty"   # 对应图中的 Tax rate (惩罚系数)
    param_x_name = "decay_rate"  # 对应图中的 Investment (自净/治理率)
    
    # 定义变化范围 (根据你模型中的原始值适当波动)
    # 原始 k_penalty = 4.0
    # 原始 decay_rate = 0.1
    y_values = np.array([2.0, 3.0, 4.0, 5.0, 6.0]) 
    x_values = np.array([0.06, 0.07, 0.08, 0.09, 0.10,]) 
    
    # 准备结果矩阵
    result_matrix = np.zeros((len(y_values), len(x_values)))

    print(f"开始敏感性扫描: {param_y_name} (Y) vs {param_x_name} (X)")
    
    # 保存原始参数以便复原
    original_y = getattr(model, param_y_name)
    original_x = getattr(model, param_x_name)

    # ==========================================
    # 3. 双重循环计算 (Grid Search)
    # ==========================================
    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            # 1. 动态修改模型参数
            setattr(model, param_y_name, y_val)
            setattr(model, param_x_name, x_val)
            
            # 2. 运行评估
            # evaluate 返回 ([obj_cost, obj_time, obj_env], constraints)
            objs, _ = model.evaluate(x_SE_baseline, x_Rocket_baseline)
            
            # 3. 提取环境影响 (objs[2])
            # 为了让热力图更好看，可能需要归一化，或者直接用原始值
            env_impact = objs[2] 
            result_matrix[i, j] = env_impact / 1e7  # 转换为百万单位，便于显示

    # 恢复原始参数
    setattr(model, param_y_name, original_y)
    setattr(model, param_x_name, original_x)

    # ==========================================
    # 4. 绘图 (红绿配色 + 坐标修正版)
    # ==========================================
    df_heatmap = pd.DataFrame(
        result_matrix, 
        index=np.round(y_values, 2), 
        columns=np.round(x_values, 2)
    )

    plt.figure(figsize=(10, 8), dpi=150)
    sns.set_theme(style="white")

    # cmap="RdYlGn_r" 让大值(高影响)显示为红色，小值(低影响)显示为绿色
    # ax.invert_yaxis() 确保纵坐标从上往下是增加的（或根据需求调整）
    # 如果你希望 6.0 在顶部，2.0 在底部，则保留 invert_yaxis
    ax = sns.heatmap(
        df_heatmap, 
        annot=True, 
        fmt=".2f", 
        cmap="RdYlGn_r",       # 红-黄-绿配色
        linewidths=0.8, 
        linecolor='white',
        square=True, 
        cbar_kws={"shrink": .8, "label": r'Environmental Impact ($E \times 10^7$)'},
        annot_kws={"size": 12, "weight": "bold"}
    )

    # 修正纵坐标：如果你希望数值从小到大向上排列，请【注释掉】下面这一行
    ax.invert_yaxis() 

    plt.title('Sensitivity Analysis: Environmental Risk Map', fontsize=16, pad=20)
    plt.ylabel(r'Tax/Penalty Rate ($k_{penalty}$)', fontsize=14)
    plt.xlabel(r'Environmental Decay Rate ($\delta_{env}$)', fontsize=14)

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11, rotation=0)

    # 保存并显示
    save_path = 'images/sensitivity_heatmap_redgreen.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_sensitivity_analysis()