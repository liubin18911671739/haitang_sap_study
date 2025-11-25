"""
结构方程模型 (SEM) - 机制-效能路径分析
Structural Equation Modeling - Mechanism-Effectiveness Path Analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import semopy
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


# ========================
# 1. 构建 SEM 模型语法
# ========================
def build_sem_model_syntax() -> str:
    """
    构建 SEM 模型语法字符串
    
    路径:
    CoCreate(7维) → Engage(OSE) → ΔGenAI(4维) → ProductQuality
    
    Returns:
        semopy 格式的模型语法
    """
    model_syntax = """
    # 测量模型 (CFA)
    
    # 共创过程潜变量
    CoCreate =~ pos_interdep + indiv_account + collaboration + shared_mental + safe_env + creative_comm + group_reflect
    
    # 参与度潜变量
    Engage =~ ose_skill + ose_emotion + ose_particip + ose_perf
    
    # GenAI素养增量潜变量
    DeltaGenAI =~ delta_ai_knowledge_total + delta_ai_skill_total + delta_ai_ethics_total + delta_innovation_teamwork_total
    
    # 结构模型 (路径)
    
    # 路径1: 共创 → 参与度
    Engage ~ CoCreate
    
    # 路径2: 参与度 → GenAI增量
    DeltaGenAI ~ Engage
    
    # 路径3: GenAI增量 → 产出质量
    product_score ~ DeltaGenAI
    
    # 路径4: 共创 → GenAI增量 (直接效应)
    DeltaGenAI ~ CoCreate
    
    # 路径5: 参与度 → 产出质量 (直接效应)
    product_score ~ Engage
    """
    return model_syntax


# ========================
# 2. 简化版 SEM（仅用观测变量）
# ========================
def build_simplified_sem_syntax() -> str:
    """
    构建简化版 SEM（用于小样本场景）
    使用维度总分作为观测变量
    
    Returns:
        简化版模型语法
    """
    model_syntax = """
    # 简化版结构模型（观测变量版本）
    
    # 共创总分
    cocreate_total =~ pos_interdep + indiv_account + collaboration + shared_mental + safe_env + creative_comm + group_reflect
    
    # 参与度总分
    engage_total =~ ose_skill + ose_emotion + ose_particip + ose_perf
    
    # 路径模型
    engage_total ~ cocreate_total
    product_score ~ engage_total + cocreate_total
    """
    return model_syntax


# ========================
# 3. 运行 SEM 模型
# ========================
def run_sem_model(
    df: pd.DataFrame,
    model_syntax: str,
    model_name: str = "SEM_Model",
) -> Tuple[Optional[semopy.Model], Dict]:
    """
    运行 SEM 模型并返回拟合结果
    
    Args:
        df: 数据集（必须包含所有模型变量）
        model_syntax: semopy 格式的模型语法
        model_name: 模型名称
    
    Returns:
        (拟合的模型对象, 拟合指标字典)
    """
    try:
        # 创建模型
        model = semopy.Model(model_syntax)
        
        # 拟合模型
        model.fit(df)
        
        # 提取拟合指标
        fit_stats = extract_fit_indices(model)
        
        print(f"\n{'='*50}")
        print(f"模型: {model_name}")
        print(f"{'='*50}")
        print(f"拟合成功！")
        print(f"样本量: {len(df)}")
        
        return model, fit_stats
    
    except Exception as e:
        print(f"\n模型 {model_name} 拟合失败: {e}")
        return None, {}


# ========================
# 4. 提取拟合指标
# ========================
def extract_fit_indices(model: semopy.Model) -> Dict:
    """
    提取 SEM 拟合指标
    
    Args:
        model: 拟合的 semopy 模型
    
    Returns:
        拟合指标字典
    """
    try:
        stats = semopy.calc_stats(model)
        
        fit_indices = {
            "Chi2": stats.get("chi2", np.nan),
            "Chi2_df": stats.get("DoF", np.nan),
            "Chi2_p": stats.get("chi2_p", np.nan),
            "CFI": stats.get("CFI", np.nan),
            "TLI": stats.get("TLI", np.nan),
            "RMSEA": stats.get("RMSEA", np.nan),
            "SRMR": stats.get("SRMR", np.nan),
            "AIC": stats.get("AIC", np.nan),
            "BIC": stats.get("BIC", np.nan),
        }
        
        return fit_indices
    
    except Exception as e:
        print(f"拟合指标提取失败: {e}")
        return {}


# ========================
# 5. 提取路径系数
# ========================
def extract_path_coefficients(model: semopy.Model) -> pd.DataFrame:
    """
    提取标准化路径系数
    
    Args:
        model: 拟合的 semopy 模型
    
    Returns:
        包含路径系数的 DataFrame
    """
    try:
        # 获取参数估计
        estimates = model.inspect()
        
        # 筛选回归路径（~）
        paths = estimates[estimates["op"] == "~"].copy()
        
        # 重命名列以便阅读
        paths = paths.rename(columns={
            "lval": "Dependent",
            "rval": "Independent",
            "Estimate": "Coefficient",
            "Std. Err": "SE",
            "z-value": "z",
            "p-value": "p",
        })
        
        # 选择关键列
        result = paths[["Independent", "Dependent", "Coefficient", "SE", "z", "p"]]
        
        return result
    
    except Exception as e:
        print(f"路径系数提取失败: {e}")
        return pd.DataFrame()


# ========================
# 6. 提取因子载荷
# ========================
def extract_factor_loadings(model: semopy.Model) -> pd.DataFrame:
    """
    提取因子载荷（测量模型部分）
    
    Args:
        model: 拟合的 semopy 模型
    
    Returns:
        包含因子载荷的 DataFrame
    """
    try:
        estimates = model.inspect()
        
        # 筛选载荷（=~）
        loadings = estimates[estimates["op"] == "=~"].copy()
        
        loadings = loadings.rename(columns={
            "lval": "Latent",
            "rval": "Indicator",
            "Estimate": "Loading",
            "Std. Err": "SE",
            "z-value": "z",
            "p-value": "p",
        })
        
        result = loadings[["Latent", "Indicator", "Loading", "SE", "z", "p"]]
        
        return result
    
    except Exception as e:
        print(f"因子载荷提取失败: {e}")
        return pd.DataFrame()


# ========================
# 7. 中介效应分析
# ========================
def test_mediation_effect(
    df: pd.DataFrame,
    x_col: str,
    m_col: str,
    y_col: str,
) -> Dict:
    """
    使用回归方法检验中介效应
    
    Args:
        df: 数据集
        x_col: 自变量
        m_col: 中介变量
        y_col: 因变量
    
    Returns:
        中介效应结果字典
    """
    try:
        from statsmodels.formula.api import ols
        
        # 路径 c: X -> Y
        model_c = ols(f"{y_col} ~ {x_col}", data=df).fit()
        c_coef = model_c.params[x_col]
        c_p = model_c.pvalues[x_col]
        
        # 路径 a: X -> M
        model_a = ols(f"{m_col} ~ {x_col}", data=df).fit()
        a_coef = model_a.params[x_col]
        a_p = model_a.pvalues[x_col]
        
        # 路径 b 和 c': M -> Y (控制 X)
        model_b = ols(f"{y_col} ~ {x_col} + {m_col}", data=df).fit()
        b_coef = model_b.params[m_col]
        b_p = model_b.pvalues[m_col]
        c_prime_coef = model_b.params[x_col]
        c_prime_p = model_b.pvalues[x_col]
        
        # 间接效应 = a * b
        indirect_effect = a_coef * b_coef
        
        # 直接效应 = c'
        direct_effect = c_prime_coef
        
        # 总效应 = c
        total_effect = c_coef
        
        return {
            "X": x_col,
            "M": m_col,
            "Y": y_col,
            "Total_Effect_c": total_effect,
            "Direct_Effect_c_prime": direct_effect,
            "Indirect_Effect_ab": indirect_effect,
            "Path_a": a_coef,
            "Path_a_p": a_p,
            "Path_b": b_coef,
            "Path_b_p": b_p,
            "Path_c_prime_p": c_prime_p,
        }
    
    except Exception as e:
        print(f"中介效应分析失败: {e}")
        return {}


# ========================
# 8. 生成 SEM 报告
# ========================
def generate_sem_report(
    model: Optional[semopy.Model],
    fit_indices: Dict,
    path_coefs: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    生成 SEM 完整报告文件
    
    Args:
        model: 拟合的模型对象
        fit_indices: 拟合指标字典
        path_coefs: 路径系数 DataFrame
        factor_loadings: 因子载荷 DataFrame
        output_path: 输出文件路径
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("结构方程模型 (SEM) 分析报告\n")
        f.write("Structural Equation Modeling Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        # 1. 拟合指标
        f.write("【1】模型拟合指标\n")
        f.write("-" * 80 + "\n")
        if fit_indices:
            for key, value in fit_indices.items():
                # 处理可能是Series的情况
                if hasattr(value, 'iloc'):
                    value = value.iloc[0] if len(value) > 0 else np.nan
                try:
                    f.write(f"{key:15s}: {value:.4f}\n")
                except (TypeError, ValueError):
                    f.write(f"{key:15s}: {value}\n")
        else:
            f.write("拟合指标提取失败\n")
        f.write("\n")
        
        # 2. 路径系数
        f.write("【2】标准化路径系数\n")
        f.write("-" * 80 + "\n")
        if not path_coefs.empty:
            f.write(path_coefs.to_string(index=False))
        else:
            f.write("路径系数提取失败\n")
        f.write("\n\n")
        
        # 3. 因子载荷
        f.write("【3】因子载荷（测量模型）\n")
        f.write("-" * 80 + "\n")
        if not factor_loadings.empty:
            f.write(factor_loadings.to_string(index=False))
        else:
            f.write("因子载荷提取失败\n")
        f.write("\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("报告生成完毕\n")
    
    print(f"SEM 报告已保存至: {output_path}")
