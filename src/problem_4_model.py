import numpy as np
import pandas as pd
import os

class MoonColonyModel:
    def __init__(self, harbor_data_filename='harbor.xlsx'):
        """
        初始化模型，加载数据和参数
        :param harbor_data_filename: 基地数据文件名 (默认假设在 ../data/ 目录下)
        """
        # ==========================================
        # 1. 稳健的路径处理与数据加载
        # ==========================================
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.data_path = os.path.join(project_root, 'data', harbor_data_filename)

        if not os.path.exists(self.data_path):
            self.data_path = harbor_data_filename
            if not os.path.exists(self.data_path):
                 raise FileNotFoundError(f"错误：无法找到基地数据文件。请确保文件位于 '{os.path.join(project_root, 'data')}' 目录下。")

        try:
            if self.data_path.endswith('.csv'):
                self.harbor_df = pd.read_csv(self.data_path)
            else:
                self.harbor_df = pd.read_excel(self.data_path)
        except Exception as e:
            raise Exception(f"读取文件失败: {e}")

        self.num_bases = len(self.harbor_df)

        # ==========================================
        # 2. 基地参数预处理
        # ==========================================
        # 解析纬度
        def parse_lat(x):
            if isinstance(x, (int, float)): return float(x)
            s = str(x).replace('°N','').replace('°S','').strip()
            return float(s) if s else 0.0
        
        self.harbor_df['Lat_Value'] = self.harbor_df['纬度'].apply(parse_lat)

        # 计算地理环境敏感系数
        self.harbor_df['Geo_Coeff'] = 0.5 + (self.harbor_df['Lat_Value'] / 90.0) * 0.5

        # 预排序：按成本系数从小到大排序 (用于贪心策略)
        self.sorted_harbors = self.harbor_df.sort_values(by='2050年单次成本系数').reset_index(drop=True)
        self.base_capacities_launches = self.sorted_harbors['2050年最大年发射次数'].values # 次/年
        self.base_cost_coeffs = self.sorted_harbors['2050年单次成本系数'].values
        self.base_geo_coeffs = self.sorted_harbors['Geo_Coeff'].values

        # ==========================================
        # 3. 核心物理与经济参数
        # ==========================================
        self.TOTAL_DEMAND = 1e8          
        self.SE_CAPACITY_YEAR = 537000   
        self.ROCKET_PAYLOAD = 125        
        self.START_YEAR = 2050
        
        # [NEW] 计算全球火箭总运力上限 (Constraint Bound)
        # 上限 = sum(所有基地的最大发射次数) * 单次载荷
        self.total_launches_global = np.sum(self.base_capacities_launches)
        self.GLOBAL_ROCKET_CAPACITY_YEAR = self.total_launches_global * self.ROCKET_PAYLOAD
        
        print(f"--- 模型初始化 ---")
        print(f"全球火箭基地数: {self.num_bases}")
        print(f"全球火箭年最大发射次数: {self.total_launches_global:,.0f}")
        print(f"全球火箭年运力上限: {self.GLOBAL_ROCKET_CAPACITY_YEAR:,.0f} 吨/年")

        self.DISCOUNT_RATE_MONEY = 0.0  
        self.SE_FIXED_COST = 4.12e9      
        self.SE_VAR_COST = 60000         

        # 火箭成本拟合公式: Cost(t) = Base + A * (t - 2005)^k
        self.rocket_cost_base = 100.0
        self.rocket_cost_A = 12906.93
        self.rocket_cost_k = -0.8872
        self.rocket_cost_t0 = 2005

        # ==========================================
        # 4. 环境影响模型参数
        # ==========================================
        self.DISCOUNT_RATE_ENV = 0.02    
        self.c_SE = 0.1   
        self.c_R = 0.5    
        self.geo_factor_SE = 2.5 
        self.env_threshold_A = 1e6       
        self.c_st = 1e11                  
        self.k_penalty = 4.0             
        self.r_penalty = 1.5             
        self.decay_rate = 0.1            
        self.a_SE = 1.0
        self.a_R = 1.0
        self.debris_coeff_launch = 0   
        self.debris_coeff_fail = 0     
        self.prob_fail = 0.05            
        self.cost_per_debris_unit = 1e4  

    def get_rocket_cost_per_ton(self, year):
        t_val = year - self.rocket_cost_t0
        if t_val <= 0: t_val = 0.1 
        cost_per_kg = self.rocket_cost_base + self.rocket_cost_A * np.power(t_val, self.rocket_cost_k)
        return cost_per_kg * 1000 

    def evaluate(self, x_SE_flow, x_Rocket_flow):
        n_years = len(x_SE_flow)
        years = np.arange(self.START_YEAR, self.START_YEAR + n_years)
        
        total_transported = 0
        total_cost_npv = 0
        total_env_impact = 0
        
        yearly_impacts = np.zeros(n_years)
        current_A_t = 0  
        completion_year_idx = n_years - 1
        is_completed = False
        
        total_rocket_capacity_violation = 0

        # 预计算基地运力(吨)
        base_caps_tons = self.base_capacities_launches * self.ROCKET_PAYLOAD

        for t_idx, year in enumerate(years):
            m_se = max(0, x_SE_flow[t_idx])
            m_rocket_demand = max(0, x_Rocket_flow[t_idx])
            
            if is_completed or total_transported >= self.TOTAL_DEMAND:
                m_se = 0
                m_rocket_demand = 0
                if not is_completed:
                    completion_year_idx = t_idx - 1 
                    is_completed = True
            
            # 成本计算 (SE)
            c_se_total = self.SE_FIXED_COST + m_se * self.SE_VAR_COST
            
            # 成本计算 (Rocket - 贪心分配)
            c_rocket_total = 0
            remaining_load = m_rocket_demand
            actual_rocket_transported = 0
            env_rocket_accum = 0 
            
            rocket_unit_price = self.get_rocket_cost_per_ton(year)
            
            for b_idx in range(self.num_bases):
                if remaining_load <= 1e-3: break
                
                cap = base_caps_tons[b_idx]
                fill = min(remaining_load, cap)
                
                c_rocket_total += fill * rocket_unit_price * self.base_cost_coeffs[b_idx]
                launches = fill / self.ROCKET_PAYLOAD
                env_rocket_accum += self.c_R * launches * self.base_geo_coeffs[b_idx]
                
                remaining_load -= fill
                actual_rocket_transported += fill
            
            # 检查运力违背 (如果超过了全球总上限)
            if remaining_load > 1e-3:
                total_rocket_capacity_violation += remaining_load
                c_rocket_total += remaining_load * rocket_unit_price * 10.0 # 惩罚
            
            # 环境影响计算
            env_se_direct = self.c_SE * m_se * self.geo_factor_SE
            
            total_launches = np.ceil(m_rocket_demand / self.ROCKET_PAYLOAD)
            expected_failures = total_launches * self.prob_fail
            debris_units = (total_launches * self.debris_coeff_launch) + \
                           (expected_failures * self.debris_coeff_fail)
            env_debris = debris_units * self.cost_per_debris_unit
            
            activity_input = (self.a_SE * env_se_direct) + (self.a_R * env_rocket_accum)
            current_A_t = current_A_t * (1 - self.decay_rate) + activity_input
            
            if current_A_t > self.env_threshold_A:
                overdraft = (current_A_t - self.env_threshold_A) / self.env_threshold_A
                D_t = self.c_st * (1 + self.k_penalty * (overdraft ** self.r_penalty))
            else:
                D_t = 0
            
            total_yearly_impact = (env_se_direct + env_rocket_accum + env_debris) + D_t
            yearly_impacts[t_idx] = total_yearly_impact
            
            p_t = (1 + self.DISCOUNT_RATE_ENV) ** -(t_idx)
            total_env_impact += total_yearly_impact * p_t
            
            d_t = (1 + self.DISCOUNT_RATE_MONEY) ** -(t_idx)
            total_cost_npv += (c_se_total + c_rocket_total) * d_t
            
            total_transported += (m_se + actual_rocket_transported)

        # 目标与约束输出
        obj_cost = total_cost_npv / 1e9
        
        if total_transported < self.TOTAL_DEMAND - 1.0: 
            obj_time = n_years + (self.TOTAL_DEMAND - total_transported) / 1e5
        else:
            obj_time = float(completion_year_idx + 1)
            
        obj_env = total_env_impact / 1e6
        
        viol_demand = max(0, self.TOTAL_DEMAND - total_transported)
        viol_se = np.sum(np.maximum(0, x_SE_flow - self.SE_CAPACITY_YEAR))
        viol_cap = viol_se + total_rocket_capacity_violation
        
        mid = int((completion_year_idx + 1) / 2)
        if mid > 0:
            early_avg = np.mean(yearly_impacts[:mid])
            late_avg = np.mean(yearly_impacts[mid:completion_year_idx+1])
            viol_equity = max(0, late_avg - 2 * early_avg)
        else:
            viol_equity = 0.0
            
        return [obj_cost, obj_time, obj_env], [viol_demand, viol_cap, viol_equity]