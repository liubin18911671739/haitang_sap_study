"""
统计分析工具函数 - 信效度、前后测、效应量等
Statistical Utilities - Reliability, validity, pre-post tests, effect sizes
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional, List
import pingouin as pg
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from sklearn.metrics import roc_auc_score, classification_report
import warnings

warnings.filterwarnings('ignore')


# ========================
# 1. 外部基线统计
# ========================
def compute_external_baseline(
    df: pd.DataFrame,
    var_cols: List[str],
) -> pd.DataFrame:
    """
    计算外部基线的描述统计
    
    Args:
        df: 外部数据集
        var_cols: 需要统计的变量列
    
    Returns:
        包含均值、标准差、样本量的 DataFrame
    """
    results = []
    for col in var_cols:
        if col in df.columns:
            results.append({
                "Variable": col,
                "Mean": df[col].mean(),
                "SD": df[col].std(),
                "N": df[col].notna().sum(),
            })
    return pd.DataFrame(results)


# ========================
# 2. 前后测配对 t 检验 + Cohen's d
# ========================
def paired_ttest_with_cohens_d(
    pre_scores: pd.Series,
    post_scores: pd.Series,
    dim_name: str,
) -> Dict:
    """
    计算配对样本 t 检验 + Cohen's d 效应量
    
    Args:
        pre_scores: 前测分数
        post_scores: 后测分数
        dim_name: 维度名称
    
    Returns:
        包含统计结果的字典
    """
    # 移除缺失值
    valid_idx = pre_scores.notna() & post_scores.notna()
    pre = pre_scores[valid_idx]
    post = post_scores[valid_idx]
    
    n = len(pre)
    if n < 2:
        return {
            "Dimension": dim_name,
            "Pre_Mean": np.nan,
            "Pre_SD": np.nan,
            "Post_Mean": np.nan,
            "Post_SD": np.nan,
            "t": np.nan,
            "df": 0,
            "p": np.nan,
            "Cohens_d": np.nan,
            "N": n,
        }
    
    # 使用 pingouin 计算配对 t 检验
    ttest_result = pg.ttest(post, pre, paired=True)
    
    # 提取结果
    t_val = ttest_result["T"].values[0]
    p_val = ttest_result["p-val"].values[0]
    cohens_d = ttest_result["cohen-d"].values[0]
    df = ttest_result["dof"].values[0]
    
    return {
        "Dimension": dim_name,
        "Pre_Mean": pre.mean(),
        "Pre_SD": pre.std(),
        "Post_Mean": post.mean(),
        "Post_SD": post.std(),
        "t": t_val,
        "df": df,
        "p": p_val,
        "Cohens_d": cohens_d,
        "N": n,
    }


# ========================
# 3. Cronbach's Alpha
# ========================
def compute_cronbach_alpha(df: pd.DataFrame, items: List[str]) -> float:
    """
    计算 Cronbach's Alpha 信度系数
    
    Args:
        df: 数据集
        items: 条目列名列表
    
    Returns:
        Cronbach's Alpha 值
    """
    # 筛选有效条目
    valid_items = [col for col in items if col in df.columns]
    if len(valid_items) < 2:
        return np.nan
    
    item_data = df[valid_items].dropna()
    if item_data.shape[0] < 2:
        return np.nan
    
    # 使用 pingouin 计算
    alpha = pg.cronbach_alpha(data=item_data)[0]
    return alpha


# ========================
# 4. KMO + Bartlett 球形检验
# ========================
def compute_kmo_bartlett(df: pd.DataFrame, items: List[str]) -> Dict:
    """
    计算 KMO 和 Bartlett 球形检验
    
    Args:
        df: 数据集
        items: 条目列名列表
    
    Returns:
        包含 KMO 和 Bartlett 结果的字典
    """
    valid_items = [col for col in items if col in df.columns]
    if len(valid_items) < 3:
        return {
            "KMO": np.nan,
            "Bartlett_chi2": np.nan,
            "Bartlett_p": np.nan,
        }
    
    item_data = df[valid_items].dropna()
    if item_data.shape[0] < len(valid_items):
        return {
            "KMO": np.nan,
            "Bartlett_chi2": np.nan,
            "Bartlett_p": np.nan,
        }
    
    try:
        # KMO
        kmo_all, kmo_model = calculate_kmo(item_data)
        
        # Bartlett
        chi2, p_val = calculate_bartlett_sphericity(item_data)
        
        return {
            "KMO": kmo_model,
            "Bartlett_chi2": chi2,
            "Bartlett_p": p_val,
        }
    except Exception as e:
        print(f"KMO/Bartlett 计算失败: {e}")
        return {
            "KMO": np.nan,
            "Bartlett_chi2": np.nan,
            "Bartlett_p": np.nan,
        }


# ========================
# 5. 探索性因子分析 (EFA)
# ========================
def run_efa(
    df: pd.DataFrame,
    items: List[str],
    n_factors: int = 4,
    rotation: str = "varimax",
) -> Tuple[Optional[FactorAnalyzer], pd.DataFrame]:
    """
    运行探索性因子分析
    
    Args:
        df: 数据集
        items: 条目列名列表
        n_factors: 因子数量
        rotation: 旋转方法
    
    Returns:
        (FactorAnalyzer 对象, 因子载荷矩阵)
    """
    valid_items = [col for col in items if col in df.columns]
    if len(valid_items) < n_factors:
        print(f"条目数 ({len(valid_items)}) 少于因子数 ({n_factors})，跳过 EFA")
        return None, pd.DataFrame()
    
    item_data = df[valid_items].dropna()
    if item_data.shape[0] < len(valid_items) * 2:
        print(f"样本量不足 EFA 要求，跳过")
        return None, pd.DataFrame()
    
    try:
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method="minres")
        fa.fit(item_data)
        
        # 获取因子载荷
        loadings = pd.DataFrame(
            fa.loadings_,
            index=valid_items,
            columns=[f"Factor{i+1}" for i in range(n_factors)],
        )
        
        return fa, loadings
    except Exception as e:
        print(f"EFA 运行失败: {e}")
        return None, pd.DataFrame()


# ========================
# 6. 计算维度总分
# ========================
def compute_dimension_scores(
    df: pd.DataFrame,
    dim_items: Dict[str, List[str]],
    method: str = "mean",
) -> pd.DataFrame:
    """
    计算各维度总分
    
    Args:
        df: 数据集
        dim_items: 维度-条目映射字典
        method: 计算方法 ("mean" 或 "sum")
    
    Returns:
        带有维度总分列的 DataFrame
    """
    df_copy = df.copy()
    for dim, items in dim_items.items():
        valid_items = [col for col in items if col in df.columns]
        if len(valid_items) > 0:
            if method == "mean":
                df_copy[f"{dim}_total"] = df[valid_items].mean(axis=1)
            elif method == "sum":
                df_copy[f"{dim}_total"] = df[valid_items].sum(axis=1)
    return df_copy


# ========================
# 7. 计算增量分数（Δ）
# ========================
def compute_delta_scores(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    score_cols: List[str],
    id_col: str = "user_id",
) -> pd.DataFrame:
    """
    计算前后测增量分数
    
    Args:
        pre_df: 前测数据
        post_df: 后测数据
        score_cols: 需要计算增量的分数列
        id_col: 用户ID列名
    
    Returns:
        包含增量分数的 DataFrame
    """
    # 合并前后测数据
    merged = pre_df[[id_col] + score_cols].merge(
        post_df[[id_col] + score_cols],
        on=id_col,
        suffixes=("_pre", "_post"),
        how="inner",
    )
    
    # 计算增量
    for col in score_cols:
        merged[f"delta_{col}"] = merged[f"{col}_post"] - merged[f"{col}_pre"]
    
    return merged


# ========================
# 8. 分类模型评估（AUC）
# ========================
def evaluate_classification(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray,
    label: str = "Model",
) -> Dict:
    """
    评估分类模型性能
    
    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        y_pred: 预测类别
        label: 模型标签
    
    Returns:
        包含 AUC 等指标的字典
    """
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except:
        auc = np.nan
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    return {
        "Model": label,
        "AUC": auc,
        "Accuracy": report.get("accuracy", np.nan),
        "Precision": report.get("1", {}).get("precision", np.nan),
        "Recall": report.get("1", {}).get("recall", np.nan),
        "F1": report.get("1", {}).get("f1-score", np.nan),
    }
