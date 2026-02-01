import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def select_optimal_solution(file_path='data/nsga2_pareto_results.csv'):
    # 1. 读取数据
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return
    
    df = pd.read_csv(file_path)
    print(f"成功加载数据，共有 {len(df)} 个 Pareto 最优解。")
    
    # 提取目标矩阵 (Cost, Time, Env)
    # 注意：这三个目标都是"越小越好" (Minimization)
    data = df.values
    
    # 2. 数据标准化 (Min-Max Normalization)
    # 将所有指标缩放到 [0, 1] 范围，且方向一致（越小越好）
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data)
    
    # --- 算法 A: 熵权法确定权重 (Entropy Weight Method) ---
    # 计算比重
    # 为了避免 log(0)，加一个微小量
    P = (data_norm + 1e-9) / (data_norm + 1e-9).sum(axis=0)
    
    # 计算熵值 e_j
    k = -1 / np.log(len(df))
    e = k * (P * np.log(P)).sum(axis=0)
    
    # 计算差异系数 d_j
    d = 1 - e
    
    # 计算权重 w_j
    weights = d / d.sum()
    
    print("\n--- 熵权法计算结果 (客观权重) ---")
    print(f"成本权重 (Cost): {weights[0]:.4f}")
    print(f"时间权重 (Time): {weights[1]:.4f}")
    print(f"环境权重 (Env):  {weights[2]:.4f}")
    
    # --- 算法 B: TOPSIS 综合评价 ---
    # 1. 构建加权归一化矩阵
    Z = data_norm * weights
    
    # 2. 确定正理想解 (Z_plus) 和 负理想解 (Z_minus)
    # 因为我们已经归一化且都是越小越好，所以正理想解是 [min, min, min]
    # 但注意：MinMax缩放后，最小值是0，最大值是1
    Z_plus = Z.min(axis=0)  # 理想解 (0, 0, 0)
    Z_minus = Z.max(axis=0) # 负理想解 (w1, w2, w3)
    
    # 3. 计算欧几里得距离
    D_plus = np.sqrt(((Z - Z_plus) ** 2).sum(axis=1))
    D_minus = np.sqrt(((Z - Z_minus) ** 2).sum(axis=1))
    
    # 4. 计算相对贴近度 C_i (分数越高越好)
    C = D_minus / (D_plus + D_minus)
    
    # 5. 排序并取最优
    best_topsis_idx = np.argmax(C)
    
    # --- 算法 C: 简单的几何折衷点 (Knee Point / Distance to Utopia) ---
    # 不考虑权重，直接找距离归一化原点 (0,0,0) 最近的点
    # 这通常被认为是"性价比"最高的均衡点
    dist_to_origin = np.sqrt((data_norm ** 2).sum(axis=1))
    best_knee_idx = np.argmin(dist_to_origin)
    
    # --- 输出结果 ---
    print("\n" + "="*50)
    print("推荐的最优解 (TOPSIS Method)")
    print("="*50)
    print(f"索引 ID: {best_topsis_idx}")
    print(f"综合得分: {C[best_topsis_idx]:.4f}")
    print("-" * 30)
    print(f"总成本 (Cost):   ${df.iloc[best_topsis_idx]['Cost(Billion)']:.2f} Billion")
    print(f"完工时间 (Time): {df.iloc[best_topsis_idx]['Time(Years)']:.1f} Years")
    print(f"环境影响 (Env):  {df.iloc[best_topsis_idx]['Env_Impact']:.2f}")
    
    print("\n" + "="*50)
    print("推荐的均衡解 (Utopia Point Distance)")
    print("="*50)
    print(f"索引 ID: {best_knee_idx}")
    print("-" * 30)
    print(f"总成本 (Cost):   ${df.iloc[best_knee_idx]['Cost(Billion)']:.2f} Billion")
    print(f"完工时间 (Time): {df.iloc[best_knee_idx]['Time(Years)']:.1f} Years")
    print(f"环境影响 (Env):  {df.iloc[best_knee_idx]['Env_Impact']:.2f}")

    # --- 可视化 ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制所有点
    sc = ax.scatter(df['Cost(Billion)'], df['Time(Years)'], df['Env_Impact'], 
                    c=C, cmap='viridis', alpha=0.6, label='Pareto Solutions')
    
    # 标记 TOPSIS 最优点
    ax.scatter(df.iloc[best_topsis_idx]['Cost(Billion)'], 
               df.iloc[best_topsis_idx]['Time(Years)'], 
               df.iloc[best_topsis_idx]['Env_Impact'], 
               c='red', s=100, marker='*', label='TOPSIS Best')

    # 标记 Knee Point 最优点
    ax.scatter(df.iloc[best_knee_idx]['Cost(Billion)'], 
               df.iloc[best_knee_idx]['Time(Years)'], 
               df.iloc[best_knee_idx]['Env_Impact'], 
               c='orange', s=100, marker='^', label='Knee Point')
    
    ax.set_xlabel('Cost ($B)')
    ax.set_ylabel('Time (Years)')
    ax.set_zlabel('Env Impact')
    ax.set_title('Optimal Solution Selection')
    plt.legend()
    
    # 保存图片
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/optimal_selection.png')
    print("\n图表已保存至 images/optimal_selection.png")
    # plt.show()

if __name__ == "__main__":
    select_optimal_solution()