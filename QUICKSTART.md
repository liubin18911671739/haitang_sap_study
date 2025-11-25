# 海棠杯 SaP 研究 - 真实数据替换与复跑清单

## 📋 快速替换清单（5步完成）

### ✅ 第1步: 准备数据文件

将您的真实数据文件重命名并放置到以下位置：

```
data/
├── external/
│   ├── external_ai_literacy.csv      ← 外部AI素养基线数据
│   └── external_ai_readiness.csv     ← 外部AI准备度数据
└── haitang_local/
    ├── haitang_pre.csv               ← 前测数据（必需）
    ├── haitang_post.csv              ← 后测数据（必需）
    ├── haitang_cocreate.csv          ← 共创过程数据
    ├── haitang_engagement_ose.csv    ← 参与度数据
    ├── haitang_behavior_log.csv      ← 行为日志
    └── haitang_qual_coded.xlsx       ← 质性编码数据
```

**最小要求**: 至少提供 `haitang_pre.csv` 和 `haitang_post.csv`

---

### ✅ 第2步: 更新列名映射

打开 `src/scales.py`，找到以下部分并替换为真实列名：

```python
# 第 11-31 行：GenAI 四维量表
GENAI_LITERACY_ITEMS: Dict[str, List[str]] = {
    "ai_knowledge": [
        "Q1_1",  # ← 替换为真实列名
        "Q1_2",  # ← 替换为真实列名
        "Q1_3",  # ← 替换为真实列名
    ],
    "ai_skill": [
        "Q2_1",  # ← 替换
        "Q2_2",
        "Q2_3",
    ],
    # ... 继续替换其他维度
}
```

**提示**: 用 Excel 打开数据文件，复制列名即可

---

### ✅ 第3步: 检查数据格式

运行数据检查（可选但推荐）：

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/haitang_local/haitang_pre.csv')
print('列名:', df.columns.tolist())
print('样本量:', len(df))
print('数据类型:', df.dtypes)
"
```

确认：
- [ ] 所有分数列为数值类型（float/int）
- [ ] 有 `user_id` 列
- [ ] 无明显异常值

---

### ✅ 第4步: 运行完整分析

```bash
# 1. 检查环境
python check_environment.py

# 2. 运行主程序
cd src
python main.py
```

预期运行时间：1-5 分钟（取决于数据量）

---

### ✅ 第5步: 查看结果

检查输出目录：

```bash
ls outputs/tables/
ls outputs/models/
```

关键输出文件：
- `pre_post_ai_lit.csv` - 前后测对比（论文表1）
- `ai_lit_alpha.csv` - 信度分析（论文3.2节）
- `sem_report.txt` - SEM完整报告（论文表2-3）
- `behavior_auc.csv` - 预测性能（论文表4）
- `sap_outcome_matrix.csv` - 质性证据（论文表5）

---

## 🔍 常见调整场景

### 场景1: 条目数量不同

如果您的量表有不同数量的条目，修改 `src/scales.py`:

```python
"ai_knowledge": [
    "Q1", "Q2", "Q3", "Q4", "Q5"  # 5个条目而非3个
],
```

### 场景2: 没有user_id列

如果数据没有 `user_id`，需要添加：

```python
import pandas as pd
df = pd.read_csv('haitang_pre.csv')
df.insert(0, 'user_id', range(1, len(df)+1))
df.to_csv('haitang_pre.csv', index=False)
```

### 场景3: 缺少某些数据文件

**没问题！** 程序会自动跳过缺失数据的模块。例如：
- 没有行为日志？跳过模块6
- 没有质性数据？跳过模块7
- 只要有前后测数据，核心分析即可完成

### 场景4: 需要修改统计参数

编辑 `src/config.py`:

```python
ALPHA_LEVEL = 0.01        # 改为更严格的显著性水平
N_FACTORS_EFA = 5         # 改为5因子模型
```

---

## 🚨 故障排查

### 问题1: KeyError: '某列名'

**原因**: 列名不匹配

**解决**: 检查 `src/scales.py` 中的列名是否与数据文件完全一致（区分大小写）

### 问题2: ValueError: could not convert string to float

**原因**: 数据中有非数值

**解决**: 
```python
df = pd.read_csv('your_file.csv')
df = df.apply(pd.to_numeric, errors='coerce')  # 转换为数值
df.to_csv('your_file.csv', index=False)
```

### 问题3: SEM 模型不收敛

**原因**: 样本量太小或数据质量问题

**解决**: 
1. 确保样本量 ≥ 100
2. 检查缺失值比例
3. 使用简化模型（程序默认已使用）

### 问题4: 内存错误

**原因**: 行为日志数据过大

**解决**: 抽样行为日志
```python
df_log = pd.read_csv('haitang_behavior_log.csv')
df_sample = df_log.sample(n=10000, random_state=42)
df_sample.to_csv('haitang_behavior_log.csv', index=False)
```

---

## ✨ 高级技巧

### 批量处理多个数据集

创建 `batch_run.py`:

```python
import os
import sys

datasets = ['dataset1', 'dataset2', 'dataset3']

for ds in datasets:
    print(f"\n处理 {ds}...")
    os.system(f"cp data_archive/{ds}/*.csv data/haitang_local/")
    os.system("python src/main.py")
    os.system(f"cp -r outputs results_{ds}/")
```

### 自动化报告生成

在 `src/main.py` 末尾添加：

```python
import subprocess
subprocess.run([
    "jupyter", "nbconvert", 
    "--to", "pdf", 
    "analysis_report.ipynb"
])
```

### 参数扫描

测试不同因子数：

```python
for n_factors in [3, 4, 5, 6]:
    config.N_FACTORS_EFA = n_factors
    module_reliability_validity(df_post)
```

---

## 📞 获取帮助

1. **查看详细文档**: `README.md`
2. **数据格式示例**: `data/DATA_FORMAT.md`
3. **检查环境**: `python check_environment.py`
4. **逐模块测试**: 在 Python 中单独运行各模块函数

---

## ✅ 完成确认

- [ ] 所有数据文件已放置到正确位置
- [ ] `src/scales.py` 中的列名已更新
- [ ] 运行 `check_environment.py` 通过
- [ ] 运行 `python src/main.py` 无错误
- [ ] 输出文件已生成在 `outputs/` 目录
- [ ] 数值结果与预期相符

**恭喜！您已成功完成数据替换与分析复现！** 🎉

---

*最后更新: 2025年11月25日*
