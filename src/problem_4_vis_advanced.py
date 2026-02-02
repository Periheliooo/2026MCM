import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata

# ==========================================
# 1. 配置绘图风格 (学术/商业级)
# ==========================================
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

def generate_mock_data_if_missing(filepath):
    """如果找不到真实数据，生成符合物理规律的模拟数据用于演示"""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    
    print("提示: 未找到真实数据文件，生成演示数据以展示图表效果...")
    multipliers = [0.5, 0.8, 1.0, 1.5, 2.0]
    data = []
    for mult in multipliers:
        base_time = 100 - (mult - 1.0) * 30 
        base_cost = 3500 / (mult ** 0.8)
        times = np.linspace(base_time - 10, base_time + 20, 30)
        for t in times:
            c = base_cost * (base_time / (t - 5)) + np.random.normal(0, 50)
            env = 1e6 * mult * (1 + np.random.normal(0, 0.05))
            data.append({'Cost': c, 'Time': t, 'Env': env, 'Multiplier': mult, 'Threshold_Label': f"{mult}x"})
    return pd.DataFrame(data)

def visualize_advanced():
    print("="*60)
    print("Generating Advanced Impact Visualizations (Fixing Heatmap Gap)...")
    print("="*60)
    
    df = generate_mock_data_if_missing('data/sensitivity_pareto_results.csv')
    os.makedirs('images', exist_ok=True)
    
    # ==========================================
    # 图表 A: 相对敏感度龙卷风图 (Tornado Chart)
    # ==========================================
    print("绘制图表 A: 相对敏感度龙卷风图...")
    summary = df.groupby('Multiplier').agg({'Cost': 'min', 'Time': 'min'}).reset_index()
    baseline = summary[summary['Multiplier'] == 1.0].iloc[0]
    summary['Cost_Change_Pct'] = (summary['Cost'] - baseline['Cost']) / baseline['Cost'] * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(summary))
    cost_colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in summary['Cost_Change_Pct']]
    bars = ax.barh(y_pos, summary['Cost_Change_Pct'], align='center', color=cost_colors, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Env Threshold {x}x" for x in summary['Multiplier']], fontweight='bold')
    ax.axvline(0, color='black', linewidth=1, linestyle='-')
    ax.set_xlabel('Percentage Change in Minimum Cost relative to Baseline (%)', fontsize=12)
    ax.set_title('Sensitivity "Tornado": How Environmental Constraints Impact Budget', fontsize=14, fontweight='bold', pad=20)
    
    for bar in bars:
        width_val = bar.get_width()
        label_x_pos = width_val + (1 if width_val > 0 else -6)
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width_val:+.1f}%', va='center', color='black', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('images/vis_impact_tornado.png', dpi=300)
    
    # ==========================================
    # 图表 B & C: 3D 曲面与修复后的热力图
    # ==========================================
    print("准备网格数据并修复空白区域...")
    
    x = df['Time']
    y = df['Multiplier']
    z = df['Cost']
    
    # 创建规则网格
    xi = np.linspace(x.min(), x.max(), 100) # 增加分辨率
    yi = np.linspace(y.min(), y.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # [关键修改] 混合插值法
    # 1. 线性插值 (Linear): 保证数据内部的平滑渐变
    Zi = griddata((x, y), z, (Xi, Yi), method='linear')
    
    # 2. 最近邻插值 (Nearest): 填补线性插值无法覆盖的角落 (NaN)
    mask = np.isnan(Zi)
    if np.any(mask):
        Zi_nearest = griddata((x, y), z, (Xi, Yi), method='nearest')
        Zi[mask] = Zi_nearest[mask] # 仅替换 NaN 区域
    
    # --- 绘制 图表 B: 3D 响应曲面 ---
    print("绘制图表 B: 3D 响应曲面...")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xi, Yi, Zi, cmap='viridis', edgecolor='none', alpha=0.9, antialiased=True)
    ax.set_xlabel('\nProject Duration (Years)', fontsize=11)
    ax.set_ylabel('\nEnv Threshold Multiplier', fontsize=11)
    ax.set_zlabel('\nTotal Cost (Billion $)', fontsize=11)
    ax.set_title('3D Solution Landscape (Interpolated)', fontsize=15, fontweight='bold')
    ax.view_init(elev=30, azim=-120)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Total Cost ($B)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/vis_impact_3d_surface.png', dpi=300)
    
    # --- 绘制 图表 C: 修复后的热力图 ---
    print("绘制图表 C: 修复后的热力图...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 使用 plasma_r 配色：黄色(亮)=便宜，紫色(暗)=贵
    im = ax.imshow(Zi, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', 
                   aspect='auto', cmap='plasma_r') 
    
    # 添加等高线 (Contour)
    levels = np.linspace(np.nanmin(Zi), np.nanmax(Zi), 10)
    cont = ax.contour(Xi, Yi, Zi, levels=levels, colors='white', linewidths=0.5, alpha=0.5)
    ax.clabel(cont, inline=True, fontsize=8, fmt='$%.0fB')
    
    ax.set_xlabel('Project Duration (Years)', fontsize=12)
    ax.set_ylabel('Environmental Threshold Multiplier', fontsize=12)
    ax.set_title('Feasibility Heatmap: Minimum Cost Landscape', fontsize=14, fontweight='bold', pad=15)
    
    cbar = plt.colorbar(im)
    cbar.set_label('Minimum Cost (Billion $)', fontsize=12)
    
    # 标注极值区
    ax.text(x.max()*0.95, y.min() + (y.max()-y.min())*0.1, 'Infeasible/High Cost\n(Extrapolated)', 
            color='white', ha='right', fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('images/vis_impact_heatmap.png', dpi=300)

    print("\n所有修复后的图表已生成至 images/ 文件夹。")

if __name__ == "__main__":
    visualize_advanced()