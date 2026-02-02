import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# 导入修改后的模型
from problem_4_model import MoonColonyModel

# ==========================================
# 1. 定义优化问题
# ==========================================
class MoonOptimizationProblem(ElementwiseProblem):
    def __init__(self, model, n_years):
        self.model = model
        self.n_years = n_years
        
        # 变量: n_years(SE) + n_years(Rocket)
        n_var = 2 * n_years
        n_obj = 3
        n_constr = 3
        
        xl = np.zeros(n_var)
        xu = np.zeros(n_var)
        
        # 设置上界
        xu[:n_years] = self.model.SE_CAPACITY_YEAR 
        self.max_rocket_cap = self.model.GLOBAL_ROCKET_CAPACITY_YEAR
        xu[n_years:] = self.max_rocket_cap
        
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        x_se = x[:self.n_years]
        x_rocket = x[self.n_years:]
        objs, constrs = self.model.evaluate(x_se, x_rocket)
        out["F"] = objs
        out["G"] = constrs

# ==========================================
# 2. 扩大范围的启发式初始化
# ==========================================
class HeuristicSampling(Sampling):
    def __init__(self, model, n_years):
        super().__init__()
        self.model = model
        self.n_years = n_years

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var))
        global_rocket_cap = self.model.GLOBAL_ROCKET_CAPACITY_YEAR
        
        for i in range(n_samples):
            # 策略 A: 激进型 - 从第0年开始就全力加速
            if i < n_samples * 0.2: 
                X[i, :self.n_years] = self.model.SE_CAPACITY_YEAR
                rocket_usage = np.random.uniform(0.5, 1.0, self.n_years) * global_rocket_cap
                X[i, self.n_years:] = rocket_usage
                continue

            # 策略 B: 混合启发式
            start_year = np.random.randint(0, self.n_years - 10)
            
            # 1. 填充 SE
            se_usage = np.random.uniform(0.8, 1.0, self.n_years) * self.model.SE_CAPACITY_YEAR
            X[i, :self.n_years] = se_usage
            
            # 2. 填充 Rocket
            X[i, self.n_years : self.n_years + start_year] = 0
            
            duration = self.n_years - start_year
            if duration > 0:
                growth_curve = np.linspace(0, 1, duration)
                peak_ratio = np.random.uniform(0.5, 1.0) 
                peak_cap = peak_ratio * global_rocket_cap
                
                rocket_flow = growth_curve * peak_cap
                rocket_flow *= np.random.uniform(0.8, 1.2, duration)
                rocket_flow = np.minimum(rocket_flow, global_rocket_cap)
                
                X[i, self.n_years + start_year:] = rocket_flow
            
        return X

# ==========================================
# 3. 主求解流程
# ==========================================
def solve_pareto():
    print("正在初始化模型与优化器...")
    try:
        phys_model = MoonColonyModel('harbor.xlsx') 
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 规划期 120年
    N_YEARS = 120 
    problem = MoonOptimizationProblem(phys_model, n_years=N_YEARS)
    
    print(f"优化配置: 规划期 {N_YEARS} 年")
    print(f"搜索空间火箭上限: {problem.max_rocket_cap:,.0f} 吨/年")

    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=50,
        sampling=HeuristicSampling(phys_model, N_YEARS),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.01, eta=20),
        eliminate_duplicates=True
    )
    
    print(f"开始进化计算 (Generations=200, Pop=100)...")
    termination = get_termination("n_gen", 200) 
    
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=42,
                   save_history=True,
                   verbose=True)
    
    print(f"优化完成! 耗时: {res.exec_time:.2f}s")
    
    # ==========================================
    # 4. 结果处理 (修正部分)
    # ==========================================
    if res.F is not None:
        # 筛选可行解
        feasible_mask = (res.CV.flatten() <= 1e-5) 
        
        F_feasible = res.F[feasible_mask]
        X_feasible = res.X[feasible_mask] # <--- [修正] 同时提取对应的 X
        
        if len(F_feasible) == 0:
            print("警告: 仍未找到完全满足约束的解。尝试显示 CV 最小的解...")
            best_cv_idx = np.argmin(res.CV.flatten())
            print(f"最小违背量 (CV): {res.CV[best_cv_idx]}")
            
            # 如果没有可行解，使用所有解作为结果（虽然不满足约束，但供参考）
            F_final = res.F
            X_final = res.X # <--- [修正] 赋值 X_final
        else:
            print(f"成功找到 {len(F_feasible)} 个 Pareto 最优解。")
            F_final = F_feasible
            X_final = X_feasible # <--- [修正] 赋值 X_final
            
        result_df = pd.DataFrame(F_final, columns=['Cost(Billion)', 'Time(Years)', 'Env_Impact'])
        
        # 保存 X_final (决策变量)
        # 这一步非常重要，供后续可视化使用
        X_df = pd.DataFrame(X_final)
        # 为列命名方便阅读 (SE_0...SE_N, Rocket_0...Rocket_N)
        cols = [f'SE_Year_{i}' for i in range(N_YEARS)] + [f'Rocket_Year_{i}' for i in range(N_YEARS)]
        X_df.columns = cols
        
        os.makedirs('data', exist_ok=True)
        X_df.to_csv('data/nsga2_pareto_solutions_X.csv', index=False)
        print("决策变量已保存至 data/nsga2_pareto_solutions_X.csv")

        # 选出代表性解
        if len(F_final) > 0:
            best_cost_idx = np.argmin(F_final[:, 0])
            best_time_idx = np.argmin(F_final[:, 1])
            best_env_idx = np.argmin(F_final[:, 2])
            
            print("\n--- 代表性解 (Representative Solutions) ---")
            print(f"1. 最低成本: ${F_final[best_cost_idx, 0]:.2f}B, {F_final[best_cost_idx, 1]:.1f}年, Env:{F_final[best_cost_idx, 2]:.2f}")
            print(f"2. 最快完工: ${F_final[best_time_idx, 0]:.2f}B, {F_final[best_time_idx, 1]:.1f}年, Env:{F_final[best_time_idx, 2]:.2f}")
            print(f"3. 最佳环境: ${F_final[best_env_idx, 0]:.2f}B, {F_final[best_env_idx, 1]:.1f}年, Env:{F_final[best_env_idx, 2]:.2f}")

        result_df.to_csv('data/nsga2_pareto_results.csv', index=False)
        print("Pareto 目标值已保存至 data/nsga2_pareto_results.csv")
        
        # 绘图
        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            img = ax.scatter(F_final[:, 0], F_final[:, 1], F_final[:, 2], c=F_final[:, 2], cmap='viridis')
            ax.set_xlabel('Total Cost ($B)')
            ax.set_ylabel('Time (Years)')
            ax.set_zlabel('Env Impact')
            ax.set_title(f'Pareto Front (N={N_YEARS}, Equity=2.0x)')
            fig.colorbar(img, label='Env Impact')
            
            os.makedirs('images', exist_ok=True)
            plt.savefig('images/pareto_3d_scatter.png')
            print("图表已保存至 images/pareto_3d_scatter.png")
        except Exception as e:
            print(f"绘图失败: {e}")

    else:
        print("未找到解。")

if __name__ == "__main__":
    solve_pareto()