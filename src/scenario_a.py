import pandas as pd
import os

# 定义常量
TOTAL_MATERIAL_NEED = 100_000_000  # 1亿公吨 
CAPACITY_PER_HARBOR = 179_000      # 每港口年运力 
NUM_HARBORS = 3                    # 港口总数 

# 用户设定的成本参数
COST_EARTH_TO_APEX = 500_000       # 每公吨
COST_APEX_TO_MOON = 100_000        # 每公吨

def calculate_scenario_a():
    # 1. 计算时间线
    total_annual_capacity = CAPACITY_PER_HARBOR * NUM_HARBORS
    years_required = TOTAL_MATERIAL_NEED / total_annual_capacity
    
    # 2. 计算成本
    unit_cost = COST_EARTH_TO_APEX + COST_APEX_TO_MOON
    total_cost = TOTAL_MATERIAL_NEED * unit_cost
    
    # 3. 整理结果
    results = {
        "Scenario": "A - Space Elevator Only",
        "Total Material (Metric Tons)": TOTAL_MATERIAL_NEED,
        "Total Annual Capacity": total_annual_capacity,
        "Timeline (Years)": round(years_required, 2),
        "Unit Cost (per Ton)": unit_cost,
        "Total Cost": total_cost,
        "Completion Year": round(2050 + years_required, 2) # [cite: 10]
    }
    
    return results

def save_results(data):
    # 确保 data 目录存在
    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame([data])
    df.to_csv('data/scenario_a_results.csv', index=False)
    print("Results saved to data/scenario_a_results.csv")

if __name__ == "__main__":
    res = calculate_scenario_a()
    print("--- Scenario A Analysis ---")
    for key, value in res.items():
        print(f"{key}: {value:,}" if isinstance(value, (int, float)) else f"{key}: {value}")
    
    save_results(res)