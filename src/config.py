"""
配置文件 - 集中管理所有路径和参数
Configuration - Centralized path and parameter management
"""
from pathlib import Path
from typing import Dict, List

# ========================
# 项目根目录路径
# ========================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# ========================
# 数据路径配置
# ========================
# 外部基线数据
EXTERNAL_AI_LITERACY = DATA_ROOT / "external" / "external_ai_literacy.csv"
EXTERNAL_AI_READINESS = DATA_ROOT / "external" / "external_ai_readiness.csv"

# 海棠杯本地数据
HAITANG_PRE = DATA_ROOT / "haitang_local" / "haitang_pre.csv"
HAITANG_POST = DATA_ROOT / "haitang_local" / "haitang_post.csv"
HAITANG_COCREATE = DATA_ROOT / "haitang_local" / "haitang_cocreate.csv"
HAITANG_ENGAGEMENT = DATA_ROOT / "haitang_local" / "haitang_engagement_ose.csv"
HAITANG_BEHAVIOR_LOG = DATA_ROOT / "haitang_local" / "haitang_behavior_log.csv"
HAITANG_QUAL_CODED = DATA_ROOT / "haitang_local" / "haitang_qual_coded.xlsx"

# ========================
# 输出路径配置
# ========================
OUTPUT_TABLES = OUTPUT_ROOT / "tables"
OUTPUT_FIGS = OUTPUT_ROOT / "figs"
OUTPUT_MODELS = OUTPUT_ROOT / "models"

# 确保输出目录存在
for output_dir in [OUTPUT_TABLES, OUTPUT_FIGS, OUTPUT_MODELS]:
    output_dir.mkdir(parents=True, exist_ok=True)

# ========================
# 统计分析参数
# ========================
ALPHA_LEVEL = 0.05  # 显著性水平
N_FACTORS_EFA = 4  # EFA 因子数量
RANDOM_STATE = 42  # 随机种子

# ========================
# 机器学习参数
# ========================
ML_PARAMS = {
    "kmeans_n_clusters": 4,  # 行为轨迹聚类数
    "rf_n_estimators": 100,  # 随机森林树数量
    "test_size": 0.3,  # 测试集比例
    "cv_folds": 5,  # 交叉验证折数
}

# ========================
# SEM 模型参数
# ========================
SEM_PARAMS = {
    "max_iter": 1000,  # 最大迭代次数
    "tol": 1e-6,  # 收敛容差
}

# ========================
# 可视化参数
# ========================
VIZ_PARAMS = {
    "figure_size": (10, 6),
    "dpi": 300,
    "style": "seaborn-v0_8-darkgrid",
}

# ========================
# 日志配置
# ========================
LOG_LEVEL = "INFO"
