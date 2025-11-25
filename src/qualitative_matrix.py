"""
质性三角互证 - SaP 结果证据矩阵生成
Qualitative Triangulation - SaP Outcome Evidence Matrix Generation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


# ========================
# 1. 读取 NVivo 编码数据
# ========================
def load_qual_coded_data(file_path: Path) -> Optional[pd.DataFrame]:
    """
    读取 NVivo 导出的质性编码数据
    
    Args:
        file_path: Excel 文件路径
    
    Returns:
        编码数据 DataFrame (包含 case_id, dim, weight)
    """
    try:
        # 尝试读取 Excel
        df = pd.read_excel(file_path)
        
        # 检查必需列
        required_cols = ["case_id", "dim", "weight"]
        if not all(col in df.columns for col in required_cols):
            print(f"警告: 质性编码数据缺少必需列: {required_cols}")
            print(f"当前列: {df.columns.tolist()}")
            return None
        
        return df
    
    except FileNotFoundError:
        print(f"文件不存在: {file_path}")
        return None
    except Exception as e:
        print(f"读取质性编码数据失败: {e}")
        return None


# ========================
# 2. 按 SaP 维度聚合证据
# ========================
def aggregate_sap_evidence(
    df: pd.DataFrame,
    sap_dims: List[str],
) -> pd.DataFrame:
    """
    按 SaP 结果维度聚合编码证据
    
    Args:
        df: 质性编码数据 (case_id, dim, weight)
        sap_dims: SaP 结果维度列表
    
    Returns:
        证据矩阵 DataFrame (case_id × 各维度权重)
    """
    # 筛选 SaP 相关维度
    df_sap = df[df["dim"].isin(sap_dims)].copy()
    
    if df_sap.empty:
        print("警告: 未找到 SaP 相关编码维度")
        return pd.DataFrame()
    
    # 透视表: case_id × dim
    pivot = df_sap.pivot_table(
        index="case_id",
        columns="dim",
        values="weight",
        aggfunc="sum",
        fill_value=0,
    )
    
    # 确保所有维度都存在
    for dim in sap_dims:
        if dim not in pivot.columns:
            pivot[dim] = 0
    
    # 重新排序列
    pivot = pivot[sap_dims]
    
    # 重置索引
    pivot.reset_index(inplace=True)
    
    return pivot


# ========================
# 3. 计算维度统计摘要
# ========================
def compute_dimension_summary(matrix_df: pd.DataFrame, sap_dims: List[str]) -> pd.DataFrame:
    """
    计算各 SaP 维度的描述统计
    
    Args:
        matrix_df: 证据矩阵
        sap_dims: SaP 维度列表
    
    Returns:
        维度统计摘要 DataFrame
    """
    summary = []
    
    for dim in sap_dims:
        if dim in matrix_df.columns:
            dim_data = matrix_df[dim]
            summary.append({
                "Dimension": dim,
                "Total_Weight": dim_data.sum(),
                "Mean": dim_data.mean(),
                "SD": dim_data.std(),
                "Min": dim_data.min(),
                "Max": dim_data.max(),
                "N_Cases": (dim_data > 0).sum(),  # 有编码的案例数
            })
    
    return pd.DataFrame(summary)


# ========================
# 4. 识别典型案例
# ========================
def identify_exemplar_cases(
    matrix_df: pd.DataFrame,
    sap_dims: List[str],
    top_n: int = 3,
) -> Dict[str, List]:
    """
    识别每个维度的典型案例（权重最高）
    
    Args:
        matrix_df: 证据矩阵
        sap_dims: SaP 维度列表
        top_n: 每个维度取前 N 个案例
    
    Returns:
        维度 -> 典型案例列表的字典
    """
    exemplars = {}
    
    for dim in sap_dims:
        if dim in matrix_df.columns:
            # 按该维度排序,取 top N
            top_cases = matrix_df.nlargest(top_n, dim)[["case_id", dim]]
            exemplars[dim] = top_cases.to_dict("records")
    
    return exemplars


# ========================
# 5. 生成交叉表（案例×维度热力图数据）
# ========================
def generate_crosstab_heatmap_data(
    matrix_df: pd.DataFrame,
    sap_dims: List[str],
) -> pd.DataFrame:
    """
    生成可用于热力图的交叉表数据
    
    Args:
        matrix_df: 证据矩阵
        sap_dims: SaP 维度列表
    
    Returns:
        标准化后的矩阵（用于可视化）
    """
    # 提取维度列
    heatmap_data = matrix_df[["case_id"] + sap_dims].copy()
    
    # 标准化（0-1 范围）
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    heatmap_data[sap_dims] = scaler.fit_transform(heatmap_data[sap_dims])
    
    return heatmap_data


# ========================
# 6. 导出质性证据矩阵
# ========================
def export_sap_outcome_matrix(
    matrix_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    exemplars: Dict,
    output_dir: Path,
) -> None:
    """
    导出 SaP 结果证据矩阵及相关报告
    
    Args:
        matrix_df: 证据矩阵
        summary_df: 维度统计摘要
        exemplars: 典型案例字典
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    
    # 1. 导出证据矩阵
    matrix_df.to_csv(
        output_dir / "sap_outcome_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"已保存: sap_outcome_matrix.csv")
    
    # 2. 导出维度统计摘要
    summary_df.to_csv(
        output_dir / "sap_dimension_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"已保存: sap_dimension_summary.csv")
    
    # 3. 导出典型案例列表
    with open(output_dir / "sap_exemplar_cases.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SaP 结果维度典型案例\n")
        f.write("=" * 80 + "\n\n")
        
        for dim, cases in exemplars.items():
            f.write(f"\n【{dim}】\n")
            f.write("-" * 80 + "\n")
            for i, case in enumerate(cases, 1):
                f.write(f"  {i}. Case ID: {case['case_id']}, Weight: {case[dim]:.2f}\n")
            f.write("\n")
    
    print(f"已保存: sap_exemplar_cases.txt")


# ========================
# 7. 生成质性报告
# ========================
def generate_qualitative_report(
    matrix_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    exemplars: Dict,
    sap_dim_labels: Dict[str, str],
    output_path: Path,
) -> None:
    """
    生成完整的质性分析报告
    
    Args:
        matrix_df: 证据矩阵
        summary_df: 维度统计摘要
        exemplars: 典型案例字典
        sap_dim_labels: 维度中文标签
        output_path: 报告输出路径
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("质性三角互证分析报告 - SaP 结果证据矩阵\n")
        f.write("Qualitative Triangulation Report - SaP Outcome Evidence Matrix\n")
        f.write("=" * 80 + "\n\n")
        
        # 1. 概述
        f.write("【1】分析概述\n")
        f.write("-" * 80 + "\n")
        f.write(f"案例总数: {len(matrix_df)}\n")
        f.write(f"SaP 结果维度数: {len(summary_df)}\n")
        f.write("\n")
        
        # 2. 维度统计摘要
        f.write("【2】各维度统计摘要\n")
        f.write("-" * 80 + "\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")
        
        # 3. 典型案例
        f.write("【3】典型案例（Top 3 per dimension）\n")
        f.write("-" * 80 + "\n")
        for dim, cases in exemplars.items():
            dim_label = sap_dim_labels.get(dim, dim)
            f.write(f"\n{dim_label} ({dim}):\n")
            for i, case in enumerate(cases, 1):
                f.write(f"  {i}. Case {case['case_id']}: Weight = {case[dim]:.2f}\n")
        f.write("\n")
        
        # 4. 跨维度覆盖分析
        f.write("【4】跨维度覆盖分析\n")
        f.write("-" * 80 + "\n")
        sap_cols = [col for col in matrix_df.columns if col != "case_id"]
        # 计算每个案例在多少维度有编码
        if sap_cols:
            matrix_df_temp = matrix_df[sap_cols]
            coverage = (matrix_df_temp > 0).sum(axis=1)
            f.write(f"平均每案例覆盖维度数: {coverage.mean():.2f}\n")
            f.write(f"覆盖所有维度的案例数: {(coverage == len(sap_cols)).sum()}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("报告生成完毕\n")
    
    print(f"质性分析报告已保存至: {output_path}")


# ========================
# 8. 主流程函数
# ========================
def run_qualitative_analysis(
    qual_coded_path: Path,
    sap_dims: List[str],
    sap_dim_labels: Dict[str, str],
    output_dir: Path,
) -> None:
    """
    运行完整的质性分析流程
    
    Args:
        qual_coded_path: 质性编码数据文件路径
        sap_dims: SaP 结果维度列表
        sap_dim_labels: 维度中文标签映射
        output_dir: 输出目录
    """
    print("\n" + "="*60)
    print("质性三角互证分析 - SaP 结果证据矩阵")
    print("="*60)
    
    # 1. 读取数据
    df = load_qual_coded_data(qual_coded_path)
    if df is None or df.empty:
        print("跳过质性分析（数据不可用）")
        return
    
    print(f"成功读取质性编码数据: {len(df)} 条记录")
    
    # 2. 生成证据矩阵
    matrix_df = aggregate_sap_evidence(df, sap_dims)
    if matrix_df.empty:
        print("证据矩阵生成失败,跳过")
        return
    
    print(f"生成证据矩阵: {len(matrix_df)} 个案例 × {len(sap_dims)} 个维度")
    
    # 3. 计算统计摘要
    summary_df = compute_dimension_summary(matrix_df, sap_dims)
    
    # 4. 识别典型案例
    exemplars = identify_exemplar_cases(matrix_df, sap_dims, top_n=3)
    
    # 5. 导出结果
    export_sap_outcome_matrix(matrix_df, summary_df, exemplars, output_dir)
    
    # 6. 生成报告
    report_path = output_dir / "qualitative_analysis_report.txt"
    generate_qualitative_report(
        matrix_df,
        summary_df,
        exemplars,
        sap_dim_labels,
        report_path,
    )
    
    print("\n质性分析完成!")
