import pandas as pd
import os

# 定义物理参数
TOTAL_MATERIAL_NEED = 100_000_000  # 总材料需求：1亿公吨 
CAPACITY_PER_HARBOR = 179_000      # 每港口年运力：17.9万公吨 
NUM_HARBORS = 3                    # 港口总数：3个 

# 用户设定的成本参数 (单位：货币单位/公吨)
COST_EARTH_TO_APEX = 500_000       # 50万
COST_APEX_TO_MOON = 100_000        # 10万
UNIT_COST = COST_EARTH_TO_APEX + COST_APEX_TO_MOON

def calculate_scenario_a():
    # 1. 计算时间线
    total_annual_capacity = CAPACITY_PER_HARBOR * NUM_HARBORS
    years_required = TOTAL_MATERIAL_NEED / total_annual_capacity
    
    # 2. 计算成本
    total_cost_raw = TOTAL_MATERIAL_NEED * UNIT_COST
    annual_cost_raw = total_annual_capacity * UNIT_COST
    
    # 3. 转换为更合理的展示单位 (以“百亿元”为单位，方便论文呈现)
    unit_label = "10 Billion (百亿元)"
    total_cost_formatted = total_cost_raw / 10_000_000_000
    annual_cost_formatted = annual_cost_raw / 10_000_000_000
    
    # 4. 整理结果
    results = {
        "Scenario": "A - Space Elevator Only",
        "Total Material (Metric Tons)": f"{TOTAL_MATERIAL_NEED:,}",
        "Total Annual Capacity (Tons/Year)": f"{total_annual_capacity:,}",
        "Timeline (Years)": round(years_required, 2),
        "Unit Cost (per Ton)": f"{UNIT_COST:,}",
        f"Total Cost ({unit_label})": round(total_cost_formatted, 2),
        f"Annual Cost ({unit_label})": round(annual_cost_formatted, 2),
        "Completion Year": round(2050 + years_required, 1)
    }
    
    return results

def save_to_xlsx(data):
    # 确保 data 目录存在
    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame([data])
    # 导出为 Excel 文件
    file_path = 'data/scenario_a_results.xlsx'
    df.to_excel(file_path, index=False)
    print(f"结果已成功导出至: {file_path}")

if __name__ == "__main__":
    res = calculate_scenario_a()
    print("--- 场景 A 计算结果 (仅太空电梯) ---")
    for key, value in res.items():
        print(f"{key}: {value}")
    
    save_to_xlsx(res)