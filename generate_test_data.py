"""
生成测试数据 - 用于验证系统功能
Generate Test Data - For system validation
"""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 50  # 样本量

print("生成测试数据...")

# 1. 生成前测数据
print("  - 生成前测数据 (haitang_pre.csv)")
pre_data = {
    'user_id': [f'S{i:03d}' for i in range(1, n+1)],
    'AI基础知识1': np.random.uniform(3, 4.5, n),
    'AI基础知识2': np.random.uniform(3, 4.5, n),
    'AI基础知识3': np.random.uniform(3, 4.5, n),
    'AI核心技能1': np.random.uniform(2.8, 4.3, n),
    'AI核心技能2': np.random.uniform(2.8, 4.3, n),
    'AI核心技能3': np.random.uniform(2.8, 4.3, n),
    'AI伦理意识1': np.random.uniform(3.2, 4.6, n),
    'AI伦理意识2': np.random.uniform(3.2, 4.6, n),
    'AI伦理意识3': np.random.uniform(3.2, 4.6, n),
    '创新思维1': np.random.uniform(3, 4.4, n),
    '团队协作1': np.random.uniform(3, 4.4, n),
    '问题解决1': np.random.uniform(3, 4.4, n),
}
df_pre = pd.DataFrame(pre_data)
df_pre.to_csv('data/haitang_local/haitang_pre.csv', index=False, encoding='utf-8-sig')

# 2. 生成后测数据 (略高于前测，模拟学习效果)
print("  - 生成后测数据 (haitang_post.csv)")
post_data = {col: df_pre[col].values.copy() for col in df_pre.columns}
for col in post_data:
    if col != 'user_id':
        post_data[col] = np.clip(post_data[col] + np.random.uniform(0.3, 0.8, n), 1, 5)
df_post = pd.DataFrame(post_data)
df_post.to_csv('data/haitang_local/haitang_post.csv', index=False, encoding='utf-8-sig')

# 3. 生成共创过程数据
print("  - 生成共创过程数据 (haitang_cocreate.csv)")
cocreate_data = {
    'user_id': [f'S{i:03d}' for i in range(1, n+1)],
    'pos_interdep': np.random.uniform(3.5, 5, n),
    'indiv_account': np.random.uniform(3.5, 5, n),
    'collaboration': np.random.uniform(3.5, 5, n),
    'shared_mental': np.random.uniform(3.3, 4.8, n),
    'safe_env': np.random.uniform(3.5, 5, n),
    'creative_comm': np.random.uniform(3.3, 4.8, n),
    'group_reflect': np.random.uniform(3.3, 4.8, n),
}
df_cocreate = pd.DataFrame(cocreate_data)
df_cocreate.to_csv('data/haitang_local/haitang_cocreate.csv', index=False, encoding='utf-8-sig')

# 4. 生成参与度数据
print("  - 生成参与度数据 (haitang_engagement_ose.csv)")
engagement_data = {
    'user_id': [f'S{i:03d}' for i in range(1, n+1)],
    'ose_skill': np.random.uniform(3.5, 5, n),
    'ose_emotion': np.random.uniform(3.5, 5, n),
    'ose_particip': np.random.uniform(3.5, 5, n),
    'ose_perf': np.random.uniform(3.3, 4.8, n),
    'product_score': np.random.uniform(70, 95, n),
    'high_quality': np.random.choice([0, 1], n, p=[0.3, 0.7]),
}
df_engagement = pd.DataFrame(engagement_data)
df_engagement.to_csv('data/haitang_local/haitang_engagement_ose.csv', index=False, encoding='utf-8-sig')

# 5. 生成行为日志数据
print("  - 生成行为日志数据 (haitang_behavior_log.csv)")
actions = ['view', 'discuss', 'cocreate', 'submit', 'revise']
behavior_logs = []
for i in range(1, n+1):
    user_id = f'S{i:03d}'
    n_actions = np.random.randint(10, 50)
    for j in range(n_actions):
        behavior_logs.append({
            'user_id': user_id,
            'action': np.random.choice(actions),
            'ts': f'2024-03-{np.random.randint(1, 30):02d} {np.random.randint(9, 18):02d}:{np.random.randint(0, 60):02d}:00',
            'duration': np.random.uniform(1, 30),
        })
df_behavior = pd.DataFrame(behavior_logs)
df_behavior.to_csv('data/haitang_local/haitang_behavior_log.csv', index=False, encoding='utf-8-sig')

# 6. 生成质性编码数据
print("  - 生成质性编码数据 (haitang_qual_coded.xlsx)")
qual_dims = ['empowerment', 'belonging', 'learning_gain', 'relation_shift', 'org_improvement']
qual_data = []
for i in range(1, 21):  # 20个案例
    case_id = f'C{i:03d}'
    for dim in qual_dims:
        if np.random.rand() > 0.3:  # 70% 概率有编码
            qual_data.append({
                'case_id': case_id,
                'dim': dim,
                'weight': np.random.randint(3, 20),
            })
df_qual = pd.DataFrame(qual_data)
df_qual.to_excel('data/haitang_local/haitang_qual_coded.xlsx', index=False)

# 7. 生成外部基线数据
print("  - 生成外部基线数据 (external_ai_literacy.csv)")
external_lit = pd.DataFrame({
    'country': ['USA', 'UK', 'China', 'Japan', 'Germany'],
    'sample_size': [250, 180, 300, 200, 220],
    'ai_knowledge_mean': [3.85, 3.91, 3.67, 3.75, 3.88],
    'ai_knowledge_sd': [0.62, 0.55, 0.70, 0.65, 0.58],
    'ai_skill_mean': [3.72, 3.88, 3.55, 3.68, 3.80],
    'ai_skill_sd': [0.58, 0.61, 0.65, 0.60, 0.56],
})
external_lit.to_csv('data/external/external_ai_literacy.csv', index=False, encoding='utf-8-sig')

print("  - 生成外部基线数据 (external_ai_readiness.csv)")
external_ready = pd.DataFrame({
    'institution': ['Univ_A', 'Univ_B', 'Univ_C', 'Univ_D', 'Univ_E'],
    'readiness_score': [75.2, 68.5, 82.1, 71.3, 78.8],
    'year': [2023] * 5,
    'n': [120, 95, 150, 110, 130],
})
external_ready.to_csv('data/external/external_ai_readiness.csv', index=False, encoding='utf-8-sig')

print("\n✅ 测试数据生成完成！")
print(f"\n生成的文件:")
print(f"  - data/haitang_local/haitang_pre.csv ({n} 样本)")
print(f"  - data/haitang_local/haitang_post.csv ({n} 样本)")
print(f"  - data/haitang_local/haitang_cocreate.csv ({n} 样本)")
print(f"  - data/haitang_local/haitang_engagement_ose.csv ({n} 样本)")
print(f"  - data/haitang_local/haitang_behavior_log.csv ({len(df_behavior)} 条日志)")
print(f"  - data/haitang_local/haitang_qual_coded.xlsx ({len(df_qual)} 条编码)")
print(f"  - data/external/external_ai_literacy.csv (5 个国家)")
print(f"  - data/external/external_ai_readiness.csv (5 个机构)")
print(f"\n现在可以运行: cd src && ../.venv/bin/python3 main.py")
