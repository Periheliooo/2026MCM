import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import os

# Load data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'problem1_result.xlsx')
df = pd.read_excel(data_path)

# Translation mapping (Chinese to English)
translation = {
    'x_t': 'Space Elevator',
    '法属圭亚那': 'French Guiana',
    '印度萨迪什': 'Satish Dhawan',
    '德州': 'Texas',
    '佛罗里达': 'Florida',
    '加州': 'California',
    '弗吉尼亚': 'Virginia',
    '中国太原': 'Taiyuan',
    '哈萨克斯坦': 'Kazakhstan',
    '新西兰马希亚': 'Mahia NZ',
    '阿拉斯加': 'Alaska'
}

# Rename columns for plotting
df_eng = df.rename(columns=translation)
base_cols_eng = [translation[col] for col in ['法属圭亚那', '印度萨迪什', '德州', '佛罗里达', '加州', '弗吉尼亚', '中国太原', '哈萨克斯坦', '新西兰马希亚', '阿拉斯加']]
labels_eng = ['Space Elevator'] + base_cols_eng

# Prepare data for stackplot
years = df_eng['Year']
data_stack = [df_eng['Space Elevator']] + [df_eng[col] for col in base_cols_eng]

# Style settings
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

fig, ax = plt.subplots()

# Colors: Space Elevator blue, Rockets varied warm/distinct colors
colors = ['#1f77b4'] + sns.color_palette("husl", len(base_cols_eng))

# Stackplot
ax.stackplot(years, data_stack, labels=labels_eng, colors=colors, alpha=0.9, edgecolor='none')

# Vertical line for Rocket start
rocket_start_idx = df.index[df['y_t'] > 0].tolist()
if rocket_start_idx:
    start_year = df.loc[rocket_start_idx[0], 'Year']
    # Add vertical line
    ax.axvline(x=start_year, color='#333333', linestyle='--', alpha=0.8, linewidth=1.5)
    
    # Calculate max y for positioning
    max_y = (df['x_t'] + df['y_t']).max()
    
    # Text annotation in blank space (Top Left area)
    ax.annotate(f'Rocket Transport Starts ({int(start_year)})', 
                xy=(start_year, max_y * 0.6), # Pointing to the dashed line
                xytext=(years.min() + 2, max_y * 0.9), # Text placed in the empty top-left area
                arrowprops=dict(facecolor='#333333', arrowstyle='->', connectionstyle="arc3,rad=.2"),
                fontsize=14, fontweight='bold', color='#333333',
                backgroundcolor='white')

# Formatting
ax.set_xlim(years.min(), years.max())
# Format Y axis with commas
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

ax.set_xlabel('Year')
ax.set_ylabel('Total Annual Transport Mass')
ax.set_title('Annual Transport Plan (Space Elevator & Rockets) 2051-2141', pad=20)

# Legend placed outside
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), title="Transport Mode / Base", frameon=True)

plt.tight_layout()
# Save or show
img_save_path = os.path.join(current_dir, '..', 'images', 'transport_plan_english.png')
plt.savefig(img_save_path, dpi=300)
plt.show()