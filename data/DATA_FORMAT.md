# 数据格式示例说明

本文件提供各数据文件的格式示例，帮助您准备真实数据。

## 1. 前后测数据格式

### haitang_pre.csv / haitang_post.csv

```csv
user_id,AI基础知识1,AI基础知识2,AI基础知识3,AI核心技能1,AI核心技能2,AI核心技能3,AI伦理意识1,AI伦理意识2,AI伦理意识3,创新思维1,团队协作1,问题解决1
S001,4.5,4.2,4.0,3.8,4.1,3.9,4.3,4.0,4.2,3.7,4.0,3.8
S002,3.9,3.7,4.1,4.2,4.0,4.3,3.8,3.9,4.1,4.0,3.9,4.2
S003,4.0,4.1,3.8,3.9,4.2,4.0,4.1,4.3,3.9,4.2,4.1,4.0
```

**说明**:
- `user_id`: 学生唯一标识符
- 所有条目: Likert 5点量表 (1-5)
- 12个条目分属4个维度（每维度3个条目）

---

## 2. 共创过程数据

### haitang_cocreate.csv

```csv
user_id,pos_interdep,indiv_account,collaboration,shared_mental,safe_env,creative_comm,group_reflect
S001,4.2,4.0,4.3,3.9,4.1,4.0,3.8
S002,3.8,4.1,3.9,4.0,3.7,4.2,4.0
S003,4.0,3.9,4.1,4.2,4.0,3.9,4.1
```

**说明**:
- 7个维度对应馆社共创过程
- Likert 5点量表 (1-5)

---

## 3. OSE 参与度数据

### haitang_engagement_ose.csv

```csv
user_id,ose_skill,ose_emotion,ose_particip,ose_perf,product_score,high_quality
S001,4.1,4.0,3.9,4.2,85.5,1
S002,3.8,3.9,4.1,3.7,78.2,0
S003,4.3,4.2,4.0,4.1,92.0,1
```

**说明**:
- OSE 四维: Likert 5点量表
- `product_score`: 作品评分 (0-100)
- `high_quality`: 高质量标记 (0/1 二分类)

---

## 4. 行为日志数据

### haitang_behavior_log.csv

```csv
user_id,action,ts,duration
S001,view,2024-03-01 09:30:00,15.5
S001,discuss,2024-03-01 10:00:00,8.2
S001,cocreate,2024-03-01 14:20:00,45.0
S002,view,2024-03-01 09:35:00,12.0
S002,submit,2024-03-02 16:00:00,5.0
```

**说明**:
- `action`: 行为类型 (view/discuss/cocreate/submit/revise)
- `ts`: 时间戳 (YYYY-MM-DD HH:MM:SS)
- `duration`: 持续时长（分钟）

---

## 5. 质性编码数据

### haitang_qual_coded.xlsx

| case_id | dim | weight |
|---------|-----|--------|
| C001 | empowerment | 15 |
| C001 | belonging | 8 |
| C001 | learning_gain | 12 |
| C002 | empowerment | 10 |
| C002 | relation_shift | 6 |

**说明**:
- `case_id`: 案例标识
- `dim`: SaP 结果维度
- `weight`: 编码频次/权重

可能的 `dim` 取值:
- `empowerment`: 赋权增能
- `belonging`: 归属感
- `learning_gain`: 学习收获
- `relation_shift`: 关系转变
- `org_improvement`: 组织改善

---

## 6. 外部基线数据

### external_ai_literacy.csv

```csv
country,sample_size,ai_knowledge_mean,ai_knowledge_sd,ai_skill_mean,ai_skill_sd
USA,250,3.85,0.62,3.72,0.58
UK,180,3.91,0.55,3.88,0.61
China,300,3.67,0.70,3.55,0.65
```

### external_ai_readiness.csv

```csv
institution,readiness_score,year,n
Univ_A,75.2,2023,120
Univ_B,68.5,2023,95
Univ_C,82.1,2023,150
```

**说明**: 格式灵活，主要用于提取均值和标准差

---

## 数据准备提示

1. **编码一致性**: 确保所有 `user_id` 在各文件中一致
2. **缺失值**: 用空格表示缺失值，程序会自动处理
3. **数值类型**: 所有分数列应为数值型（float）
4. **日期格式**: 时间戳建议使用 ISO 格式
5. **字符编码**: 保存为 UTF-8 编码（带 BOM 更佳）

## 最小数据要求

要运行完整流程，至少需要：
- `haitang_pre.csv` (N ≥ 30)
- `haitang_post.csv` (N ≥ 30，与 pre 匹配)

其他数据文件可选，缺失时会跳过相应模块。
