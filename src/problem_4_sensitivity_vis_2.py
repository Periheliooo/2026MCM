import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.interpolate import make_interp_spline
import os

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk") # 适合报告/PPT的字体大小

def smooth_curve(x, y, points=300, k=3):
    """
    使用 B-Spline 对散乱的 Pareto 前沿进行平滑处理
    x, y: 原始坐标点
    points: 插值点数量
    k: 平滑阶数 (3=cubic)
    """
    # 1. 必须先对数据进行排序
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # 2. 去重 (Spline 不允许重复的 x)
    x_unique, unique_indices = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_indices]
    
    # 3. 如果点太少，直接返回线性插值
    if len(x_unique) < k + 1:
        return x_unique, y_unique

    # 4. 创建平滑曲线
    # 使用 make_interp_spline 创建平滑函数
    try:
        spl = make_interp_spline(x_unique, y_unique, k=k)
        x_new = np.linspace(x_unique.min(), x_unique.max(), points)
        y_new = spl(x_new)
        return x_new, y_new
    except Exception as e:
        return x_unique, y_unique

def plot_beautiful_sensitivity(results_dict):
    """
    绘制高级美观的敏感性分析图
    :param results_dict: 字典 {multiplier: np.array([[Cost, Time, Env, RocketRatio], ...])}
    """
    
    # 准备配色方案 (使用渐变色)
    multipliers = sorted(list(results_dict.keys()))
    palette = sns.color_palette("magma_r", n_colors=len(multipliers))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # --- 左图：平滑的 Pareto 前沿 (Cost vs Time) ---
    for i, mult in enumerate(multipliers):
        data = results_dict[mult]
        # data[:, 0] = Cost, data[:, 1] = Time
        
        # 提取并平滑
        x_raw = data[:, 1] # Time
        y_raw = data[:, 0] # Cost
        
        # 过滤掉极端的离群点 (可选)
        mask = y_raw < np.percentile(y_raw, 95)
        x_raw, y_raw = x_raw[mask], y_raw[mask]

        x_smooth, y_smooth = smooth_curve(x_raw, y_raw)
        
        # 绘制主曲线
        color = palette[i]
        ax1.plot(x_smooth, y_smooth, label=f'{mult}x Cost Scenario', 
                 color=color, linewidth=3, alpha=0.9)
        
        # 绘制半透明填充 (增强层次感，避免线条混淆)
        # 填充曲线到上方某个高值，或者曲线下方
        ax1.fill_between(x_smooth, y_smooth, y2=0, color=color, alpha=0.1)

    ax1.set_xlabel("Completion Time (Years)", fontweight='bold')
    ax1.set_ylabel("Total Cost NPV ($ Billion)", fontweight='bold')
    ax1.set_title("Pareto Front Shift: Cost vs Time", fontsize=16, pad=20)
    ax1.set_xlim(left=min([d[:,1].min() for d in results_dict.values()]) * 0.9)
    ax1.set_ylim(bottom=0)
    ax1.legend(frameon=True, fancybox=True, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # --- 右图：策略结构变化 (Rocket Dependency) ---
    # 使用散点 + 趋势线 (Lowess Regression)
    all_x = []
    all_y = []
    all_hue = []
    
    for mult in multipliers:
        data = results_dict[mult]
        # data[:, 1] = Time, data[:, 3] = RocketRatio
        n_points = len(data)
        all_x.extend(data[:, 1])
        all_y.extend(data[:, 3] * 100) # 转为百分比
        all_hue.extend([f"{mult}x"] * n_points)
    
    df_plot = pd.DataFrame({'Time': all_x, 'RocketUsage': all_y, 'Scenario': all_hue})
    
    # 使用 Seaborn 的 lmplot 或 lineplot 自动计算置信区间
    # 注意：lmplot 是 figure-level，这里我们用 regplot 叠加
    
    for i, mult in enumerate(multipliers):
        subset = df_plot[df_plot['Scenario'] == f"{mult}x"]
        color = palette[i]
        
        # 绘制带置信区间的平滑回归线
        sns.regplot(
            data=subset, x='Time', y='RocketUsage', 
            ax=ax2, order=2, # 使用2阶多项式拟合趋势
            scatter_kws={'s': 30, 'alpha': 0.3}, # 散点淡化
            line_kws={'linewidth': 3},
            color=color, label=f'{mult}x Scenario',
            ci=None #如果不想要阴影带就把这个设为None，如果想要设为95
        )

    ax2.set_xlabel("Completion Time (Years)", fontweight='bold')
    ax2.set_ylabel("Rocket Transport Share (%)", fontweight='bold')
    ax2.set_title("Strategy Adaptation: Rocket Dependency", fontsize=16, pad=20)
    ax2.set_ylim(0, 100)
    ax2.legend(title="Cost Multiplier")
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    save_path = 'images/advanced_sensitivity_plot.png'
    plt.savefig(save_path, dpi=300)
    print(f"图表已保存至 {save_path}")
    # plt.show() # 如果在 notebook 中运行请取消注释

# ==========================================
# 模拟数据生成 (如果您有真实数据，请替换此部分)
# ==========================================
def generate_mock_data():
    results = {}
    multipliers = [0.8, 1.0, 1.2, 1.5]
    
    for m in multipliers:
        # 模拟数据：时间越短，成本越高；倍率越高，整体越贵
        # Time: 20 - 80 years
        times = np.linspace(25, 85, 50) + np.random.normal(0, 2, 50)
        
        # Cost 模型: Base / (Time - 10) * Multiplier
        costs = (2000 / (times - 20)) * m * (1 + np.random.normal(0, 0.05, 50)) + 50 * m
        
        # Rocket Ratio 模型: 越快(时间短)越依赖火箭; 越贵(倍率高)越少用火箭
        # 基础依赖度
        base_ratio = 1.0 - (times - 25) / 60.0 
        # 价格弹性调整
        price_elasticity = 1.0 / (m ** 2) 
        ratios = base_ratio * price_elasticity
        ratios = np.clip(ratios, 0.05, 0.95)
        
        # 拼装 [Cost, Time, Env(dummy), RocketRatio]
        # 注意：Raw data 可能是乱序的
        data = np.column_stack((costs, times, np.zeros_like(costs), ratios))
        
        # 打乱顺序模拟真实 NSGA2 输出
        np.random.shuffle(data)
        
        results[m] = data
    
    return results

if __name__ == "__main__":
    # 1. 获取数据 (这里使用模拟数据，请替换为您真实代码中的 results 字典)
    # 您的真实代码中 results = {0.8: F_final_08, 1.0: F_final_10, ...}
    # 且确保 F_final 包含第4列 (RocketRatio)
    mock_results = generate_mock_data()
    
    # 2. 绘图
    plot_beautiful_sensitivity(mock_results)
    plt.show()