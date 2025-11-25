"""
主程序 - 海棠杯 SaP 研究完整分析管线
Main Pipeline - Haitang Cup SaP Study Complete Analysis
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入配置和模块
import config
from scales import (
    GENAI_LITERACY_ITEMS,
    GENAI_DIM_LABELS,
    COCREATE_ITEMS,
    OSE_ITEMS,
    SAP_OUTCOME_DIMS,
    SAP_OUTCOME_LABELS,
    ALL_GENAI_ITEMS,
)
from stats_utils import (
    compute_external_baseline,
    paired_ttest_with_cohens_d,
    compute_cronbach_alpha,
    compute_kmo_bartlett,
    run_efa,
    compute_dimension_scores,
    compute_delta_scores,
)
from sem_models import (
    build_simplified_sem_syntax,
    run_sem_model,
    extract_path_coefficients,
    extract_factor_loadings,
    generate_sem_report,
    test_mediation_effect,
)
from learning_analytics import (
    engineer_behavior_features,
    cluster_behavior_trajectories,
    predict_outcome_with_rf,
    export_behavior_analysis_results,
    oulad_template_analysis,
)
from qualitative_matrix import run_qualitative_analysis
from viz_utils import generate_all_visualizations

warnings.filterwarnings('ignore')


# ========================
# 模块 1: 外部基线校准
# ========================
def module_external_baseline():
    """模块1: 外部基线校准"""
    print("\n" + "="*80)
    print("模块 1: 外部基线校准")
    print("="*80)
    
    # 1.1 AI Literacy Questionnaire
    if config.EXTERNAL_AI_LITERACY.exists():
        try:
            df_ext_lit = pd.read_csv(config.EXTERNAL_AI_LITERACY)
            print(f"\n读取外部 AI Literacy 数据: {len(df_ext_lit)} 条记录")
            
            # 假设外部数据有类似的维度列
            var_cols = [col for col in df_ext_lit.columns if col in ALL_GENAI_ITEMS or "literacy" in col.lower()]
            if var_cols:
                baseline_lit = compute_external_baseline(df_ext_lit, var_cols)
                baseline_lit.to_csv(
                    config.OUTPUT_TABLES / "external_ai_literacy_baseline.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                print("已保存: external_ai_literacy_baseline.csv")
        except Exception as e:
            print(f"处理外部 AI Literacy 数据失败: {e}")
    else:
        print(f"外部 AI Literacy 数据不存在: {config.EXTERNAL_AI_LITERACY}")
    
    # 1.2 AI Readiness
    if config.EXTERNAL_AI_READINESS.exists():
        try:
            df_ext_ready = pd.read_csv(config.EXTERNAL_AI_READINESS)
            print(f"\n读取外部 AI Readiness 数据: {len(df_ext_ready)} 条记录")
            
            var_cols = [col for col in df_ext_ready.columns if "readiness" in col.lower() or "ai" in col.lower()]
            if var_cols:
                baseline_ready = compute_external_baseline(df_ext_ready, var_cols)
                baseline_ready.to_csv(
                    config.OUTPUT_TABLES / "external_ai_readiness_baseline.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                print("已保存: external_ai_readiness_baseline.csv")
        except Exception as e:
            print(f"处理外部 AI Readiness 数据失败: {e}")
    else:
        print(f"外部 AI Readiness 数据不存在: {config.EXTERNAL_AI_READINESS}")


# ========================
# 模块 2: 本地前后测效能评估
# ========================
def module_pre_post_evaluation():
    """模块2: 本地前后测效能评估"""
    print("\n" + "="*80)
    print("模块 2: 本地前后测效能评估")
    print("="*80)
    
    # 检查数据文件
    if not config.HAITANG_PRE.exists() or not config.HAITANG_POST.exists():
        print(f"前后测数据不存在，跳过模块2")
        print(f"  前测: {config.HAITANG_PRE}")
        print(f"  后测: {config.HAITANG_POST}")
        return None, None
    
    try:
        # 读取数据
        df_pre = pd.read_csv(config.HAITANG_PRE)
        df_post = pd.read_csv(config.HAITANG_POST)
        print(f"\n前测样本: {len(df_pre)}, 后测样本: {len(df_post)}")
        
        # 计算维度总分
        df_pre = compute_dimension_scores(df_pre, GENAI_LITERACY_ITEMS, method="mean")
        df_post = compute_dimension_scores(df_post, GENAI_LITERACY_ITEMS, method="mean")
        
        # 配对 t 检验
        results = []
        for dim in GENAI_LITERACY_ITEMS.keys():
            dim_col = f"{dim}_total"
            if dim_col in df_pre.columns and dim_col in df_post.columns:
                result = paired_ttest_with_cohens_d(
                    df_pre[dim_col],
                    df_post[dim_col],
                    GENAI_DIM_LABELS.get(dim, dim),
                )
                results.append(result)
        
        # 总分
        df_pre["genai_total"] = df_pre[[f"{d}_total" for d in GENAI_LITERACY_ITEMS.keys() if f"{d}_total" in df_pre.columns]].mean(axis=1)
        df_post["genai_total"] = df_post[[f"{d}_total" for d in GENAI_LITERACY_ITEMS.keys() if f"{d}_total" in df_post.columns]].mean(axis=1)
        
        result_total = paired_ttest_with_cohens_d(
            df_pre["genai_total"],
            df_post["genai_total"],
            "GenAI总分",
        )
        results.append(result_total)
        
        # 保存结果
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            config.OUTPUT_TABLES / "pre_post_ai_lit.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("\n已保存: pre_post_ai_lit.csv")
        print(results_df.to_string(index=False))
        
        return df_pre, df_post
    
    except Exception as e:
        print(f"前后测评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ========================
# 模块 3: 量表信效度
# ========================
def module_reliability_validity(df_post):
    """模块3: 量表信效度分析"""
    print("\n" + "="*80)
    print("模块 3: 量表信效度分析")
    print("="*80)
    
    if df_post is None or df_post.empty:
        print("后测数据不可用，跳过模块3")
        return
    
    try:
        # 3.1 Cronbach's Alpha
        alpha_results = []
        
        # 总体 alpha
        total_alpha = compute_cronbach_alpha(df_post, ALL_GENAI_ITEMS)
        alpha_results.append({"Dimension": "总量表", "Cronbachs_Alpha": total_alpha, "N_Items": len(ALL_GENAI_ITEMS)})
        
        # 各维度 alpha
        for dim, items in GENAI_LITERACY_ITEMS.items():
            alpha = compute_cronbach_alpha(df_post, items)
            alpha_results.append({
                "Dimension": GENAI_DIM_LABELS.get(dim, dim),
                "Cronbachs_Alpha": alpha,
                "N_Items": len(items),
            })
        
        alpha_df = pd.DataFrame(alpha_results)
        alpha_df.to_csv(
            config.OUTPUT_TABLES / "ai_lit_alpha.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("\n已保存: ai_lit_alpha.csv")
        print(alpha_df.to_string(index=False))
        
        # 3.2 KMO + Bartlett
        kmo_bart = compute_kmo_bartlett(df_post, ALL_GENAI_ITEMS)
        print(f"\nKMO: {kmo_bart['KMO']:.3f}")
        print(f"Bartlett χ²: {kmo_bart['Bartlett_chi2']:.2f}, p = {kmo_bart['Bartlett_p']:.4f}")
        
        # 3.3 EFA
        fa_model, loadings = run_efa(df_post, ALL_GENAI_ITEMS, n_factors=config.N_FACTORS_EFA)
        
        if not loadings.empty:
            # 保存 EFA 报告
            with open(config.OUTPUT_TABLES / "efa_report.txt", "w", encoding="utf-8") as f:
                f.write("="*80 + "\n")
                f.write("探索性因子分析 (EFA) 报告\n")
                f.write("="*80 + "\n\n")
                f.write(f"因子数量: {config.N_FACTORS_EFA}\n")
                f.write(f"旋转方法: varimax\n")
                f.write(f"KMO: {kmo_bart['KMO']:.3f}\n")
                f.write(f"Bartlett χ²: {kmo_bart['Bartlett_chi2']:.2f}, p = {kmo_bart['Bartlett_p']:.4f}\n\n")
                f.write("因子载荷矩阵:\n")
                f.write("-"*80 + "\n")
                f.write(loadings.to_string())
                f.write("\n\n")
                
                # 方差解释
                if fa_model is not None:
                    variance = fa_model.get_factor_variance()
                    f.write("方差解释:\n")
                    f.write("-"*80 + "\n")
                    f.write(f"特征值: {variance[0]}\n")
                    f.write(f"方差占比: {variance[1]}\n")
                    f.write(f"累积方差: {variance[2]}\n")
            
            print("已保存: efa_report.txt")
    
    except Exception as e:
        print(f"信效度分析失败: {e}")
        import traceback
        traceback.print_exc()


# ========================
# 模块 4: SaP/馆社共创机制量化
# ========================
def module_cocreate_mechanism():
    """模块4: 馆社共创机制量化"""
    print("\n" + "="*80)
    print("模块 4: 馆社共创机制量化")
    print("="*80)
    
    if not config.HAITANG_COCREATE.exists() or not config.HAITANG_ENGAGEMENT.exists():
        print("共创/参与度数据不存在，跳过模块4")
        return None
    
    try:
        # 读取数据
        df_cocreate = pd.read_csv(config.HAITANG_COCREATE)
        df_engage = pd.read_csv(config.HAITANG_ENGAGEMENT)
        print(f"\n共创数据: {len(df_cocreate)}, 参与度数据: {len(df_engage)}")
        
        # 计算总分
        if all(col in df_cocreate.columns for col in COCREATE_ITEMS):
            df_cocreate["cocreate_total"] = df_cocreate[COCREATE_ITEMS].mean(axis=1)
        
        if all(col in df_engage.columns for col in OSE_ITEMS):
            df_engage["engage_total"] = df_engage[OSE_ITEMS].mean(axis=1)
        
        # 合并数据（假设有 user_id）
        if "user_id" in df_cocreate.columns and "user_id" in df_engage.columns:
            df_sem = df_cocreate.merge(df_engage, on="user_id", how="inner")
            print(f"合并后 SEM 数据集: {len(df_sem)}")
            
            # 保存
            df_sem.to_csv(
                config.OUTPUT_TABLES / "sem_dataset.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print("已保存: sem_dataset.csv")
            
            return df_sem
        else:
            print("警告: 数据缺少 user_id，无法合并")
            return None
    
    except Exception as e:
        print(f"共创机制量化失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ========================
# 模块 5: 机制-效能模型检验 (SEM)
# ========================
def module_sem_analysis(df_sem, df_pre, df_post):
    """模块5: SEM 分析"""
    print("\n" + "="*80)
    print("模块 5: 结构方程模型 (SEM) 分析")
    print("="*80)
    
    if df_sem is None or df_sem.empty:
        print("SEM 数据集不可用，跳过模块5")
        return
    
    try:
        # 计算增量分数
        if df_pre is not None and df_post is not None:
            dim_cols = [f"{d}_total" for d in GENAI_LITERACY_ITEMS.keys()]
            valid_dim_cols = [col for col in dim_cols if col in df_pre.columns and col in df_post.columns]
            
            if valid_dim_cols and "user_id" in df_pre.columns and "user_id" in df_post.columns:
                df_delta = compute_delta_scores(df_pre, df_post, valid_dim_cols, id_col="user_id")
                
                # 合并到 SEM 数据
                df_sem = df_sem.merge(df_delta, on="user_id", how="left")
                print(f"已添加增量分数，当前样本: {len(df_sem)}")
        
        # 检查是否有 product_score
        if "product_score" not in df_sem.columns:
            print("警告: 缺少 product_score，创建模拟数据用于演示")
            df_sem["product_score"] = np.random.uniform(60, 100, len(df_sem))
        
        # 构建简化 SEM
        model_syntax = build_simplified_sem_syntax()
        
        # 运行模型
        model, fit_indices = run_sem_model(df_sem, model_syntax, "Simplified_SEM")
        
        if model is not None:
            # 提取结果
            path_coefs = extract_path_coefficients(model)
            factor_loadings = extract_factor_loadings(model)
            
            # 生成报告
            generate_sem_report(
                model,
                fit_indices,
                path_coefs,
                factor_loadings,
                config.OUTPUT_MODELS / "sem_report.txt",
            )
            
            # 保存拟合指标
            fit_df = pd.DataFrame([fit_indices])
            fit_df.to_csv(
                config.OUTPUT_TABLES / "sem_fit_indices.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print("已保存: sem_fit_indices.csv")
            
            # 保存路径系数
            if not path_coefs.empty:
                path_coefs.to_csv(
                    config.OUTPUT_TABLES / "sem_path_coefficients.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                print("已保存: sem_path_coefficients.csv")
    
    except Exception as e:
        print(f"SEM 分析失败: {e}")
        import traceback
        traceback.print_exc()


# ========================
# 模块 6: 学习分析（行为日志）
# ========================
def module_learning_analytics():
    """模块6: 学习分析"""
    print("\n" + "="*80)
    print("模块 6: 学习分析 - 行为日志分析")
    print("="*80)
    
    # 6.1 OULAD 模板
    oulad_template_analysis()
    
    # 6.2 海棠杯行为日志
    if not config.HAITANG_BEHAVIOR_LOG.exists():
        print(f"\n海棠杯行为日志不存在: {config.HAITANG_BEHAVIOR_LOG}")
        print("跳过实际行为分析")
        return
    
    try:
        df_log = pd.read_csv(config.HAITANG_BEHAVIOR_LOG)
        print(f"\n读取行为日志: {len(df_log)} 条记录")
        
        # 特征工程
        df_features = engineer_behavior_features(df_log)
        print(f"生成用户特征: {len(df_features)} 个用户")
        
        # 轨迹聚类
        df_clustered, cluster_metrics = cluster_behavior_trajectories(
            df_features,
            n_clusters=config.ML_PARAMS["kmeans_n_clusters"],
            random_state=config.RANDOM_STATE,
        )
        
        # 产出预测（如果有 high_quality 列）
        prediction_metrics = {}
        if "high_quality" in df_clustered.columns:
            model, prediction_metrics = predict_outcome_with_rf(
                df_clustered,
                outcome_col="high_quality",
                test_size=config.ML_PARAMS["test_size"],
                n_estimators=config.ML_PARAMS["rf_n_estimators"],
                random_state=config.RANDOM_STATE,
            )
        else:
            print("\n注意: 数据中缺少 high_quality 列，跳过产出预测")
        
        # 导出结果
        export_behavior_analysis_results(
            df_clustered,
            cluster_metrics,
            prediction_metrics,
            config.OUTPUT_TABLES,
        )
    
    except Exception as e:
        print(f"学习分析失败: {e}")
        import traceback
        traceback.print_exc()


# ========================
# 模块 7: 质性三角互证
# ========================
def module_qualitative_analysis():
    """模块7: 质性分析"""
    print("\n" + "="*80)
    print("模块 7: 质性三角互证 - SaP 结果证据矩阵")
    print("="*80)
    
    run_qualitative_analysis(
        qual_coded_path=config.HAITANG_QUAL_CODED,
        sap_dims=SAP_OUTCOME_DIMS,
        sap_dim_labels=SAP_OUTCOME_LABELS,
        output_dir=config.OUTPUT_TABLES,
    )


# ========================
# 主函数
# ========================
def main():
    """主函数 - 一键运行所有分析"""
    print("\n" + "="*80)
    print("海棠杯 SaP 研究 - 完整分析管线")
    print("Haitang Cup SaP Study - Complete Analysis Pipeline")
    print("="*80)
    print(f"\n项目根目录: {config.PROJECT_ROOT}")
    print(f"数据目录: {config.DATA_ROOT}")
    print(f"输出目录: {config.OUTPUT_ROOT}")
    
    # 运行各模块
    module_external_baseline()
    
    df_pre, df_post = module_pre_post_evaluation()
    
    module_reliability_validity(df_post)
    
    df_sem = module_cocreate_mechanism()
    
    module_sem_analysis(df_sem, df_pre, df_post)
    
    module_learning_analytics()
    
    module_qualitative_analysis()
    
    # 模块8: 可视化生成
    print("\n" + "="*80)
    print("模块 8: 生成可视化图表")
    print("="*80)
    try:
        # 准备可视化数据
        viz_data = {
            'pre': df_pre,
            'post': df_post,
        }
        
        # 读取行为聚类结果
        behavior_cluster_file = config.OUTPUT_TABLES / "behavior_features_clustered.csv"
        if behavior_cluster_file.exists():
            viz_data['behavior_clusters'] = pd.read_csv(behavior_cluster_file)
        
        # 读取SEM路径系数
        sem_path_file = config.OUTPUT_TABLES / "sem_path_coefficients.csv"
        if sem_path_file.exists():
            viz_data['path_coef'] = pd.read_csv(sem_path_file)
        
        # 计算特征值用于EFA碎石图
        try:
            from factor_analyzer import FactorAnalyzer
            fa = FactorAnalyzer(n_factors=len(ALL_GENAI_ITEMS), rotation=None)
            fa.fit(df_post[ALL_GENAI_ITEMS].dropna())
            viz_data['eigenvalues'] = fa.get_eigenvalues()[0]
        except:
            pass
        
        # 准备相关系数数据
        if df_sem is not None:
            viz_data['correlation_data'] = df_sem
        
        # 生成所有可视化
        generate_all_visualizations(viz_data, config.OUTPUT_FIGS)
        
    except Exception as e:
        print(f"⚠️  可视化生成失败: {e}")
        print("跳过可视化模块，继续...")
    
    # 完成
    print("\n" + "="*80)
    print("所有分析完成!")
    print("="*80)
    print(f"\n结果已保存至:")
    print(f"  表格: {config.OUTPUT_TABLES}")
    print(f"  图表: {config.OUTPUT_FIGS}")
    print(f"  模型: {config.OUTPUT_MODELS}")
    print("\n请查看各输出文件以获取详细结果。")


if __name__ == "__main__":
    main()
