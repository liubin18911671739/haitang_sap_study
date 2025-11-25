你现在是“学术实验工程师 + 统计建模专家 + 可复现研究开发者”。请为以下论文实验生成一个可直接运行的完整 Python 项目（多文件 repo），并一次性创建所有源码文件与 README、requirements.txt。目标是：复现“Students as Partners 视角下资源约束高校图书馆生成式AI素养教育的馆社共创机制与效能研究——以‘海棠杯’为例”的整套实证闭环。

========================
【一、研究与功能总目标】
========================
项目需要实现以下 6 个模块的可复现管线（必须一键运行 main.py 完成所有统计与导出）：

1) 外部基线校准：
   - 读取 AI Literacy Questionnaire 外部数据（external_ai_literacy.csv）
   - 读取跨国 AI readiness 外部数据（external_ai_readiness.csv）
   - 输出外部基线均值/SD 表，供论文 3.1 节对照

2) 本地前后测效能评估：
   - 读取 haitang_pre.csv / haitang_post.csv
   - 对齐 GenAI literacy 四维结构：ai_knowledge, ai_skill, ai_ethics, innovation_teamwork
   - 计算每维及总分的 pre/post 均值、配对样本 t 检验、p 值、Cohen’s d
   - 输出 pre_post_ai_lit.csv（结构与论文表1一致）

3) 量表信效度：
   - Cronbach’s alpha（总体与各维度）
   - KMO + Bartlett
   - EFA（四因子结构）
   - 输出 ai_lit_alpha.csv、efa_report.txt
   - 代码必须允许我后续替换真实条目列名

4) SaP/馆社共创机制量化：
   - 读取 haitang_cocreate.csv（7维共创过程）
   - 读取 haitang_engagement_ose.csv（OSE 参与度四维）
   - 合并为 SEM 数据集
   - 生成过程变量总分、参与度总分

5) 机制—效能模型检验（SEM + 中介）：
   - 构建结构方程/回归路径：
     CoCreate(7维) → Engage(OSE) → ΔGenAI(四维增量) → ProductQuality/持续参与
   - 使用 semopy 跑 CFA/SEM
   - 输出 sem_report.txt（含拟合指标、路径系数、显著性）

6) 资源约束下低成本学习分析（复现→迁移模板）：
   - OULAD：读取行为日志模型范式（只要写“可复现模板”，不强制下载即跑）
   - 海棠杯迁移：读取 haitang_behavior_log.csv
   - 构造行为特征（观看/讨论/共创/提交/修订 等）
   - 做轨迹聚类（KMeans）
   - 做产出预测（RandomForest），输出 AUC 表 behavior_auc.csv

7) 质性三角互证（SaP 结果证据矩阵导出）：
   - 读取 haitang_qual_coded.xlsx（NVivo 导出的编码表）
   - 按 SaP outcome dims：empowerment, belonging, learning_gain, relation_shift, org_improvement
   - 生成 sap_outcome_matrix.csv

========================
【二、必须生成的项目文件树】
========================
请在当前工作区创建如下文件与目录（文件名必须完全一致）：

haitang_sap_study/
  data/
    external/
    haitang_local/
  outputs/
    tables/
    figs/
    models/
  src/
    main.py
    config.py
    scales.py
    stats_utils.py
    sem_models.py
    learning_analytics.py
    qualitative_matrix.py
  requirements.txt
  README.md

========================
【三、编码硬性要求】
========================
A. Python 3.10+，使用 pandas/numpy/scipy/pingouin/statsmodels/sklearn/factor_analyzer/semopy/lifelines
B. 所有模块必须写成可独立 import 的函数
C. main.py 一键跑全流程，并把结果全部保存到 outputs/tables、outputs/models
D. 代码需包含充分注释、类型标注、异常处理（比如文件不存在时跳过该模块并提示）
E. 不使用 GPU；保证资源约束笔记本可跑
F. 所有列名映射集中在 scales.py 顶部的 item_map 中，方便我换真实问卷字段
G. 生成 README.md：写清楚“数据放哪里、怎么跑、输出是什么、对应论文哪个表/图”
H. requirements.txt 写死版本（给我一份可 pip install 的锁定）

========================
【四、条目/维度默认列名（可后续替换）】
========================
1) GenAI 四维条目默认列名：
   AI基础知识1 AI基础知识2 AI基础知识3
   AI核心技能1 AI核心技能2 AI核心技能3
   AI伦理意识1 AI伦理意识2 AI伦理意识3
   创新思维1 团队协作1 问题解决1

2) 共创过程七维列名：
   pos_interdep indiv_account collaboration shared_mental safe_env creative_comm group_reflect

3) OSE 四维列名：
   ose_skill ose_emotion ose_particip ose_perf

4) 作品质量列名：
   product_score high_quality(0/1)

5) 行为日志列名：
   user_id action ts duration

6) 质性编码列名：
   case_id dim weight

========================
【五、生成策略（按顺序输出）】
========================
请按以下顺序逐文件生成，每生成一个文件就给出完整内容（用 Markdown code block）：
1. requirements.txt
2. src/config.py
3. src/scales.py
4. src/stats_utils.py
5. src/sem_models.py
6. src/learning_analytics.py
7. src/qualitative_matrix.py
8. src/main.py
9. README.md

生成完后，请自检：
- main.py 是否能在无外部数据时优雅跳过相关模块
- 各文件 import 是否一致
- 输出文件路径是否全部指向 outputs/

========================
【六、最终交付标准】
========================
当你输出完所有文件后，再给一个“如何用真实海棠杯数据替换并复跑”的最短 checklist。
不要问我问题，直接假设数据按上述结构存在即可。
开始生成。
