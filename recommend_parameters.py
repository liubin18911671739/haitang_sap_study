"""
参数推荐工具 - 根据实际数据特征自动推荐最佳参数
Parameter Recommendation - Automatically suggest optimal parameters based on data
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))
import config
from scales import ALL_GENAI_ITEMS

def recommend_efa_factors(df, item_cols):
    """根据特征值推荐因子数"""
    from factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
    from sklearn.decomposition import PCA
    
    # KMO检验
    kmo_all, kmo_model = calculate_kmo(df[item_cols])
    
    # 特征值分析
    pca = PCA()
    pca.fit(df[item_cols])
    eigenvalues = pca.explained_variance_
    
    # Kaiser准则: 特征值>1
    kaiser = sum(eigenvalues > 1)
    
    # 累计解释方差>60%
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    var_60 = np.where(cum_var > 0.6)[0][0] + 1
    
    # 碎石图拐点 (简单判断: 相邻特征值变化率最大)
    diffs = np.diff(eigenvalues)
    scree = np.argmax(diffs) + 1
    
    return {
        'kmo': kmo_model,
        'kaiser': kaiser,
        'var60': var_60,
        'scree': scree,
        'recommend': int(np.median([kaiser, var_60, scree]))
    }

def recommend_clusters(df):
    """根据轮廓系数推荐聚类数"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    X = StandardScaler().fit_transform(df.select_dtypes(include=[np.number]))
    
    scores = {}
    for k in range(2, min(8, len(df)//5)):  # 至少每类5个样本
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    
    best_k = max(scores, key=scores.get)
    return {
        'scores': scores,
        'recommend': best_k,
        'best_score': scores[best_k]
    }

def recommend_rf_estimators(n_samples):
    """根据样本量推荐随机森林树数"""
    if n_samples < 50:
        return 50
    elif n_samples < 100:
        return 100
    elif n_samples < 500:
        return 200
    else:
        return 500

print("="*80)
print("参数推荐分析")
print("="*80)

# 读取数据
df_pre = pd.read_csv(config.HAITANG_PRE)
df_behavior = pd.read_csv(config.HAITANG_BEHAVIOR_LOG)

n_samples = len(df_pre)
print(f"\n样本量: {n_samples}")

# 1. EFA因子数推荐
print("\n【1】EFA 因子数推荐")
print(f"当前配置: N_FACTORS_EFA = {config.N_FACTORS_EFA}")

efa_result = recommend_efa_factors(df_pre, ALL_GENAI_ITEMS)
print(f"  KMO 适配度: {efa_result['kmo']:.3f}")
print(f"  Kaiser准则 (特征值>1): {efa_result['kaiser']} 因子")
print(f"  累计方差60%: {efa_result['var60']} 因子")
print(f"  碎石图拐点: {efa_result['scree']} 因子")
print(f"  ➡️  推荐值: {efa_result['recommend']} 因子")

if efa_result['recommend'] != config.N_FACTORS_EFA:
    print(f"  ⚠️  建议修改 config.py: N_FACTORS_EFA = {efa_result['recommend']}")

# 2. 聚类数推荐
print("\n【2】K-Means 聚类数推荐")
print(f"当前配置: kmeans_n_clusters = {config.ML_PARAMS['kmeans_n_clusters']}")

# 行为特征工程 (简化版)
df_agg = df_behavior.groupby('user_id').agg({
    'action': 'count',
    'duration': 'sum'
}).reset_index()
df_agg.columns = ['user_id', 'action_count', 'total_duration']

cluster_result = recommend_clusters(df_agg)
print(f"  轮廓系数评分:")
for k, score in sorted(cluster_result['scores'].items()):
    marker = "  ➡️ " if k == cluster_result['recommend'] else "     "
    print(f"{marker} k={k}: {score:.3f}")

if cluster_result['recommend'] != config.ML_PARAMS['kmeans_n_clusters']:
    print(f"  ⚠️  建议修改 config.py: kmeans_n_clusters = {cluster_result['recommend']}")

# 3. 随机森林树数推荐
print("\n【3】随机森林树数推荐")
print(f"当前配置: rf_n_estimators = {config.ML_PARAMS['rf_n_estimators']}")

rf_recommend = recommend_rf_estimators(n_samples)
print(f"  ➡️  推荐值: {rf_recommend}")

if rf_recommend != config.ML_PARAMS['rf_n_estimators']:
    print(f"  ⚠️  建议修改 config.py: rf_n_estimators = {rf_recommend}")

# 4. 生成配置建议
print("\n" + "="*80)
print("配置修改建议 (复制到 src/config.py):")
print("="*80)
print(f"""
# 探索性因子分析
N_FACTORS_EFA = {efa_result['recommend']}  # 推荐值 (当前: {config.N_FACTORS_EFA})

# 机器学习参数
ML_PARAMS = {{
    'kmeans_n_clusters': {cluster_result['recommend']},  # 推荐值 (当前: {config.ML_PARAMS['kmeans_n_clusters']})
    'rf_n_estimators': {rf_recommend},  # 推荐值 (当前: {config.ML_PARAMS['rf_n_estimators']})
    'test_size': {config.ML_PARAMS['test_size']},  # 保持不变
    'cv_folds': {config.ML_PARAMS['cv_folds']},  # 保持不变
}}
""")
print("="*80)
