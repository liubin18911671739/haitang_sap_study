"""
量表条目映射 - 集中管理所有问卷条目与维度
Scale Item Mapping - Centralized questionnaire item and dimension management
"""
from typing import Dict, List

# ========================
# GenAI 素养四维量表条目映射
# ========================
GENAI_LITERACY_ITEMS: Dict[str, List[str]] = {
    "ai_knowledge": [
        "AI基础知识1",
        "AI基础知识2",
        "AI基础知识3",
    ],
    "ai_skill": [
        "AI核心技能1",
        "AI核心技能2",
        "AI核心技能3",
    ],
    "ai_ethics": [
        "AI伦理意识1",
        "AI伦理意识2",
        "AI伦理意识3",
    ],
    "innovation_teamwork": [
        "创新思维1",
        "团队协作1",
        "问题解决1",
    ],
}

# 所有 GenAI 条目（用于信效度分析）
ALL_GENAI_ITEMS: List[str] = []
for items in GENAI_LITERACY_ITEMS.values():
    ALL_GENAI_ITEMS.extend(items)

# 维度名称映射（用于报告）
GENAI_DIM_LABELS: Dict[str, str] = {
    "ai_knowledge": "AI基础知识",
    "ai_skill": "AI核心技能",
    "ai_ethics": "AI伦理意识",
    "innovation_teamwork": "创新与团队协作",
}

# ========================
# 馆社共创过程七维量表
# ========================
COCREATE_ITEMS: List[str] = [
    "pos_interdep",      # 积极互依
    "indiv_account",     # 个体责任
    "collaboration",     # 协作互动
    "shared_mental",     # 共享心智模型
    "safe_env",          # 安全环境
    "creative_comm",     # 创造性沟通
    "group_reflect",     # 团队反思
]

COCREATE_DIM_LABELS: Dict[str, str] = {
    "pos_interdep": "积极互依",
    "indiv_account": "个体责任",
    "collaboration": "协作互动",
    "shared_mental": "共享心智模型",
    "safe_env": "安全环境",
    "creative_comm": "创造性沟通",
    "group_reflect": "团队反思",
}

# ========================
# OSE 参与度四维量表
# ========================
OSE_ITEMS: List[str] = [
    "ose_skill",      # 技能参与
    "ose_emotion",    # 情感参与
    "ose_particip",   # 行为参与
    "ose_perf",       # 绩效参与
]

OSE_DIM_LABELS: Dict[str, str] = {
    "ose_skill": "技能参与",
    "ose_emotion": "情感参与",
    "ose_particip": "行为参与",
    "ose_perf": "绩效参与",
}

# ========================
# 产出质量变量
# ========================
PRODUCT_QUALITY_VARS: List[str] = [
    "product_score",   # 作品评分（连续）
    "high_quality",    # 高质量标记（0/1）
]

# ========================
# 行为日志字段
# ========================
BEHAVIOR_LOG_COLS: List[str] = [
    "user_id",   # 用户ID
    "action",    # 行为类型
    "ts",        # 时间戳
    "duration",  # 持续时长
]

# 行为类型编码
BEHAVIOR_ACTIONS: List[str] = [
    "view",      # 观看
    "discuss",   # 讨论
    "cocreate",  # 共创
    "submit",    # 提交
    "revise",    # 修订
]

# ========================
# 质性编码维度（SaP 结果框架）
# ========================
SAP_OUTCOME_DIMS: List[str] = [
    "empowerment",       # 赋权增能
    "belonging",         # 归属感
    "learning_gain",     # 学习收获
    "relation_shift",    # 关系转变
    "org_improvement",   # 组织改善
]

SAP_OUTCOME_LABELS: Dict[str, str] = {
    "empowerment": "赋权增能",
    "belonging": "归属感",
    "learning_gain": "学习收获",
    "relation_shift": "关系转变",
    "org_improvement": "组织改善",
}

# ========================
# 质性数据字段
# ========================
QUAL_CODED_COLS: List[str] = [
    "case_id",  # 案例ID
    "dim",      # 维度
    "weight",   # 权重/频次
]

# ========================
# 辅助函数：获取维度总分列名
# ========================
def get_dimension_score_col(dim: str, suffix: str = "_total") -> str:
    """
    生成维度总分列名
    
    Args:
        dim: 维度名称
        suffix: 后缀（默认 _total）
    
    Returns:
        维度总分列名
    """
    return f"{dim}{suffix}"


def get_all_items_for_dimensions(dim_items: Dict[str, List[str]]) -> List[str]:
    """
    获取所有维度的所有条目
    
    Args:
        dim_items: 维度-条目映射字典
    
    Returns:
        所有条目列表
    """
    all_items = []
    for items in dim_items.values():
        all_items.extend(items)
    return all_items
