import pandas as pd
import os

# --- 1. 物理与运营参数定义 ---
TOTAL_MATERIAL_NEED = 100_000_000  # 总材料需求：1亿公吨
CAPACITY_PER_HARBOR = 179_000      # 单个港口年运力
NUM_HARBORS = 3                    # 港口数量

# --- 2. 成本参数定义 (核心修改部分) ---
# 运输单价 (每公吨)
COST_EARTH_TO_APEX = 500_000       
COST_APEX_TO_MOON = 100_000        
UNIT_TRANSPORT_COST = COST_EARTH_TO_APEX + COST_APEX_TO_MOON # 60万/吨

# 维护成本 (每年固定支出)
# 用户指定：4.12 * 10^9 (41.2亿)
ANNUAL_MAINTENANCE_COST = 4.12 * 10**9 

def calculate_scenario_a():
    # A. 计算时间线
    total_annual_capacity = CAPACITY_PER_HARBOR * NUM_HARBORS # 537,000 吨/年
    years_required = TOTAL_MATERIAL_NEED / total_annual_capacity
    
    # B. 计算运输总成本 (仅与重量有关)
    total_transport_cost = TOTAL_MATERIAL_NEED * UNIT_TRANSPORT_COST
    
    # C. 计算维护总成本 (与时间有关：年数 * 年维护费)
    total_maintenance_cost = years_required * ANNUAL_MAINTENANCE_COST
    
    # D. 总成本叠加
    total_cost_project = total_transport_cost + total_maintenance_cost
    
    # E. 计算年均支出 (Total Cost / Years)
    annual_spending = total_cost_project / years_required
    
    # --- 3. 结果格式化 (转换为 Billion/十亿 单位，方便阅读) ---
    res = {
        "Scenario": "A - Space Elevator Only (With Maint.)",
        "Total Material (Tons)": f"{TOTAL_MATERIAL_NEED:,}",
        "Timeline (Years)": round(years_required, 2),
        "Completion Year": round(2050 + years_required, 1),
        
        # 成本明细
        "Unit Transport Cost (per Ton)": f"{UNIT_TRANSPORT_COST:,}",
        "Annual Maintenance Cost": f"{ANNUAL_MAINTENANCE_COST/1e9} Billion",
        
        # 总额 (单位：十亿/Billion)
        "Total Transport Cost (Billion)": round(total_transport_cost / 1e9, 2),
        "Total Maintenance Cost (Billion)": round(total_maintenance_cost / 1e9, 2),
        "GRAND TOTAL COST (Billion)": round(total_cost_project / 1e9, 2),
        
        # 每年平均预算
        "Avg Annual Spending (Billion)": round(annual_spending / 1e9, 2)
    }
    
    return res

def save_to_xlsx(data):
    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame([data])
    file_path = 'data/scenario_a_results.xlsx'
    df.to_excel(file_path, index=False)
    print(f"数据已更新并导出至: {file_path}")

if __name__ == "__main__":
    results = calculate_scenario_a()
    print("--- Scenario A 更新结果 ---")
    for k, v in results.items():
        print(f"{k}: {v}")
    save_to_xlsx(results)