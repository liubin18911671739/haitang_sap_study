"""
学习分析模块 - 行为日志分析、轨迹聚类、产出预测
Learning Analytics - Behavior log analysis, trajectory clustering, outcome prediction
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')


# ========================
# 1. 行为日志特征工程
# ========================
def engineer_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从行为日志中构造特征
    
    Args:
        df: 行为日志数据，包含 user_id, action, ts, duration
    
    Returns:
        用户级别的特征 DataFrame
    """
    # 确保有必要的列
    required_cols = ["user_id", "action"]
    if not all(col in df.columns for col in required_cols):
        print(f"警告: 行为日志缺少必需列 {required_cols}")
        return pd.DataFrame()
    
    # 按用户聚合特征
    features = []
    
    for user_id, user_df in df.groupby("user_id"):
        user_feat = {"user_id": user_id}
        
        # 1. 各类行为频次
        action_counts = user_df["action"].value_counts().to_dict()
        user_feat["view_count"] = action_counts.get("view", 0)
        user_feat["discuss_count"] = action_counts.get("discuss", 0)
        user_feat["cocreate_count"] = action_counts.get("cocreate", 0)
        user_feat["submit_count"] = action_counts.get("submit", 0)
        user_feat["revise_count"] = action_counts.get("revise", 0)
        
        # 2. 总行为次数
        user_feat["total_actions"] = len(user_df)
        
        # 3. 活跃天数（如果有 ts 列）
        if "ts" in user_df.columns:
            try:
                user_df_copy = user_df.copy()
                user_df_copy["date"] = pd.to_datetime(user_df_copy["ts"]).dt.date
                user_feat["active_days"] = user_df_copy["date"].nunique()
            except:
                user_feat["active_days"] = 0
        else:
            user_feat["active_days"] = 0
        
        # 4. 平均持续时长（如果有 duration 列）
        if "duration" in user_df.columns:
            user_feat["avg_duration"] = user_df["duration"].mean()
            user_feat["total_duration"] = user_df["duration"].sum()
        else:
            user_feat["avg_duration"] = 0
            user_feat["total_duration"] = 0
        
        # 5. 行为多样性（Shannon entropy）
        from scipy.stats import entropy
        action_probs = user_df["action"].value_counts(normalize=True).values
        user_feat["action_diversity"] = entropy(action_probs)
        
        features.append(user_feat)
    
    return pd.DataFrame(features)


# ========================
# 2. 轨迹聚类（KMeans）
# ========================
def cluster_behavior_trajectories(
    features_df: pd.DataFrame,
    n_clusters: int = 4,
    feature_cols: Optional[List[str]] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """
    对用户行为轨迹进行聚类
    
    Args:
        features_df: 用户特征 DataFrame
        n_clusters: 聚类数量
        feature_cols: 用于聚类的特征列（默认为所有数值列）
        random_state: 随机种子
    
    Returns:
        (带有聚类标签的 DataFrame, 聚类评估指标字典)
    """
    # 如果未指定特征列，使用所有数值列（除了 user_id）
    if feature_cols is None:
        feature_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        if "user_id" in feature_cols:
            feature_cols.remove("user_id")
    
    # 筛选有效特征
    valid_cols = [col for col in feature_cols if col in features_df.columns]
    if len(valid_cols) == 0:
        print("警告: 没有可用的聚类特征")
        return features_df, {}
    
    X = features_df[valid_cols].fillna(0)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMeans 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # 计算轮廓系数
    if len(X) > n_clusters:
        silhouette = silhouette_score(X_scaled, cluster_labels)
    else:
        silhouette = np.nan
    
    # 添加聚类标签
    result_df = features_df.copy()
    result_df["cluster"] = cluster_labels
    
    # 聚类中心（逆标准化）
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(cluster_centers, columns=valid_cols)
    
    metrics = {
        "n_clusters": n_clusters,
        "silhouette_score": silhouette,
        "inertia": kmeans.inertia_,
        "cluster_sizes": pd.Series(cluster_labels).value_counts().to_dict(),
    }
    
    print(f"\n聚类完成:")
    print(f"  聚类数: {n_clusters}")
    print(f"  轮廓系数: {silhouette:.3f}")
    print(f"  各簇大小: {metrics['cluster_sizes']}")
    
    return result_df, metrics


# ========================
# 3. 产出预测（随机森林）
# ========================
def predict_outcome_with_rf(
    features_df: pd.DataFrame,
    outcome_col: str,
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.3,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, Dict]:
    """
    使用随机森林预测产出质量
    
    Args:
        features_df: 特征数据集
        outcome_col: 结果变量列名（二分类）
        feature_cols: 特征列（默认为所有数值列）
        test_size: 测试集比例
        n_estimators: 树的数量
        random_state: 随机种子
    
    Returns:
        (训练好的模型, 评估指标字典)
    """
    # 检查结果变量是否存在
    if outcome_col not in features_df.columns:
        print(f"警告: 结果变量 {outcome_col} 不存在")
        return None, {}
    
    # 如果未指定特征列,使用所有数值列
    if feature_cols is None:
        feature_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        if outcome_col in feature_cols:
            feature_cols.remove(outcome_col)
        if "user_id" in feature_cols:
            feature_cols.remove("user_id")
    
    valid_cols = [col for col in feature_cols if col in features_df.columns]
    if len(valid_cols) == 0:
        print("警告: 没有可用的预测特征")
        return None, {}
    
    # 准备数据
    df_clean = features_df[[outcome_col] + valid_cols].dropna()
    if len(df_clean) < 10:
        print(f"警告: 有效样本量不足 ({len(df_clean)})，跳过预测")
        return None, {}
    
    X = df_clean[valid_cols]
    y = df_clean[outcome_col]
    
    # 检查类别平衡
    if y.nunique() < 2:
        print("警告: 结果变量只有一个类别，无法训练分类器")
        return None, {}
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 训练随机森林
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=10,
        min_samples_split=5,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    
    # 预测
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    # 评估
    from stats_utils import evaluate_classification
    metrics = evaluate_classification(y_test, y_pred_proba, y_pred, label="RandomForest")
    
    # 交叉验证
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="roc_auc")
    metrics["CV_AUC_mean"] = cv_scores.mean()
    metrics["CV_AUC_std"] = cv_scores.std()
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        "Feature": valid_cols,
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=False)
    
    metrics["feature_importance"] = feature_importance
    
    print(f"\n随机森林预测完成:")
    print(f"  测试集 AUC: {metrics['AUC']:.3f}")
    print(f"  交叉验证 AUC: {metrics['CV_AUC_mean']:.3f} ± {metrics['CV_AUC_std']:.3f}")
    print(f"\n特征重要性 Top 5:")
    print(feature_importance.head().to_string(index=False))
    
    return rf, metrics


# ========================
# 4. OULAD 模板函数（可复现框架）
# ========================
def oulad_template_analysis(oulad_path: Optional[str] = None) -> Dict:
    """
    OULAD 数据分析模板（仅作为范式参考）
    
    本函数展示如何使用 OULAD 数据集进行学习分析:
    1. 加载学生行为日志 (studentVle)
    2. 加载学生成绩 (studentAssessment)
    3. 构造特征
    4. 预测最终成绩/退学风险
    
    Args:
        oulad_path: OULAD 数据集路径（可选）
    
    Returns:
        分析结果字典（模板）
    """
    print("\n" + "="*60)
    print("OULAD 学习分析模板")
    print("="*60)
    print("\n本模板展示 OULAD 数据集的标准分析流程:")
    print("  1. 加载 studentVle.csv (点击流日志)")
    print("  2. 加载 studentAssessment.csv (评估成绩)")
    print("  3. 加载 studentInfo.csv (学生元数据)")
    print("  4. 特征工程: 点击次数、活跃度、早期表现")
    print("  5. 预测: 最终成绩/退学风险")
    print("  6. 评估: AUC, 准确率, 特征重要性")
    
    if oulad_path:
        print(f"\n提示: 可从以下链接下载 OULAD 数据集:")
        print("  https://analyse.kmi.open.ac.uk/open_dataset")
    
    print("\n对于海棠杯数据,可直接迁移此流程:")
    print("  - 将 studentVle → haitang_behavior_log")
    print("  - 将 studentAssessment → haitang_product_score")
    print("  - 使用相同的特征工程和预测方法")
    
    return {
        "template": "OULAD",
        "status": "展示完毕",
        "note": "此为范式模板,实际运行需下载 OULAD 数据",
    }


# ========================
# 5. 导出行为分析结果
# ========================
def export_behavior_analysis_results(
    features_df: pd.DataFrame,
    clustering_metrics: Dict,
    prediction_metrics: Dict,
    output_dir: str,
) -> None:
    """
    导出行为分析的所有结果
    
    Args:
        features_df: 用户特征（含聚类标签）
        clustering_metrics: 聚类评估指标
        prediction_metrics: 预测评估指标
        output_dir: 输出目录
    """
    from pathlib import Path
    output_path = Path(output_dir)
    
    # 1. 导出用户特征 + 聚类标签
    features_df.to_csv(output_path / "behavior_features_clustered.csv", index=False, encoding="utf-8-sig")
    print(f"已保存: behavior_features_clustered.csv")
    
    # 2. 导出聚类评估指标
    cluster_summary = pd.DataFrame([clustering_metrics])
    cluster_summary.to_csv(output_path / "clustering_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"已保存: clustering_metrics.csv")
    
    # 3. 导出预测 AUC 表
    if prediction_metrics:
        auc_row = {
            "Model": prediction_metrics.get("Model", "RandomForest"),
            "AUC": prediction_metrics.get("AUC", np.nan),
            "Accuracy": prediction_metrics.get("Accuracy", np.nan),
            "Precision": prediction_metrics.get("Precision", np.nan),
            "Recall": prediction_metrics.get("Recall", np.nan),
            "F1": prediction_metrics.get("F1", np.nan),
            "CV_AUC_mean": prediction_metrics.get("CV_AUC_mean", np.nan),
            "CV_AUC_std": prediction_metrics.get("CV_AUC_std", np.nan),
        }
        auc_df = pd.DataFrame([auc_row])
        auc_df.to_csv(output_path / "behavior_auc.csv", index=False, encoding="utf-8-sig")
        print(f"已保存: behavior_auc.csv")
        
        # 4. 导出特征重要性
        if "feature_importance" in prediction_metrics:
            fi_df = prediction_metrics["feature_importance"]
            fi_df.to_csv(output_path / "feature_importance.csv", index=False, encoding="utf-8-sig")
            print(f"已保存: feature_importance.csv")
