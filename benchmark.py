"""
性能基准测试 - 测试各模块运行时间和资源占用
Performance Benchmark - Test module execution time and resource usage
"""
import time
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Callable
import tracemalloc

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import config
from stats_utils import (
    paired_ttest_with_cohens_d,
    compute_cronbach_alpha,
    compute_kmo_bartlett,
    run_efa,
    compute_dimension_scores
)
from learning_analytics import (
    engineer_behavior_features,
    cluster_behavior_trajectories
)


class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self):
        self.results = []
        self.process = psutil.Process()
    
    def measure_function(
        self,
        func: Callable,
        func_name: str,
        *args,
        **kwargs
    ) -> Dict:
        """
        测量函数性能
        
        Args:
            func: 要测试的函数
            func_name: 函数名称
            *args: 函数参数
            **kwargs: 函数关键字参数
        
        Returns:
            性能指标字典
        """
        # 开始内存追踪
        tracemalloc.start()
        
        # 记录初始状态
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # 测量执行时间
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        
        # 记录峰值内存
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # 记录最终内存
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # 计算指标
        execution_time = end_time - start_time
        memory_delta = final_memory - initial_memory
        peak_memory_mb = peak / 1024 / 1024
        
        metrics = {
            'function': func_name,
            'execution_time_sec': round(execution_time, 4),
            'memory_delta_mb': round(memory_delta, 2),
            'peak_memory_mb': round(peak_memory_mb, 2),
            'success': success,
            'error': error
        }
        
        self.results.append(metrics)
        return metrics
    
    def benchmark_stats_utils(self, df_pre: pd.DataFrame, df_post: pd.DataFrame):
        """测试统计工具函数"""
        print("\n【1】统计工具函数性能测试")
        print("-" * 60)
        
        # 配对t检验
        metrics = self.measure_function(
            paired_ttest_with_cohens_d,
            'paired_ttest_with_cohens_d',
            df_pre['AI基础知识1'],
            df_post['AI基础知识1'],
            'Test Dimension'
        )
        print(f"  配对t检验: {metrics['execution_time_sec']}秒 | {metrics['memory_delta_mb']}MB")
        
        # Cronbach's Alpha
        items = ['AI基础知识1', 'AI基础知识2', 'AI基础知识3']
        metrics = self.measure_function(
            compute_cronbach_alpha,
            'compute_cronbach_alpha',
            df_post,
            items
        )
        print(f"  Cronbach's Alpha: {metrics['execution_time_sec']}秒 | {metrics['memory_delta_mb']}MB")
        
        # KMO/Bartlett
        metrics = self.measure_function(
            compute_kmo_bartlett,
            'compute_kmo_bartlett',
            df_post,
            items
        )
        print(f"  KMO/Bartlett: {metrics['execution_time_sec']}秒 | {metrics['memory_delta_mb']}MB")
        
        # EFA
        from scales import ALL_GENAI_ITEMS
        metrics = self.measure_function(
            run_efa,
            'run_efa',
            df_post,
            ALL_GENAI_ITEMS,
            n_factors=4
        )
        print(f"  EFA因子分析: {metrics['execution_time_sec']}秒 | {metrics['memory_delta_mb']}MB")
    
    def benchmark_data_loading(self):
        """测试数据加载性能"""
        print("\n【2】数据加载性能测试")
        print("-" * 60)
        
        # CSV加载
        def load_csv():
            return pd.read_csv(config.HAITANG_PRE)
        
        metrics = self.measure_function(load_csv, 'load_csv')
        print(f"  CSV加载: {metrics['execution_time_sec']}秒 | {metrics['memory_delta_mb']}MB")
        
        # Excel加载
        def load_excel():
            return pd.read_excel(config.HAITANG_QUAL_CODED)
        
        metrics = self.measure_function(load_excel, 'load_excel')
        print(f"  Excel加载: {metrics['execution_time_sec']}秒 | {metrics['memory_delta_mb']}MB")
    
    def benchmark_learning_analytics(self, df_behavior: pd.DataFrame):
        """测试学习分析性能"""
        print("\n【3】学习分析性能测试")
        print("-" * 60)
        
        # 特征工程
        metrics = self.measure_function(
            engineer_behavior_features,
            'engineer_behavior_features',
            df_behavior
        )
        print(f"  特征工程: {metrics['execution_time_sec']}秒 | {metrics['memory_delta_mb']}MB")
        
        # 聚类分析
        if metrics['success']:
            df_features = engineer_behavior_features(df_behavior)
            metrics = self.measure_function(
                cluster_behavior_trajectories,
                'cluster_behavior_trajectories',
                df_features,
                n_clusters=4
            )
            print(f"  聚类分析: {metrics['execution_time_sec']}秒 | {metrics['memory_delta_mb']}MB")
    
    def benchmark_scalability(self):
        """测试可扩展性（不同数据规模）"""
        print("\n【4】可扩展性测试")
        print("-" * 60)
        
        sample_sizes = [10, 50, 100, 500, 1000]
        
        for n in sample_sizes:
            # 生成测试数据
            df = pd.DataFrame({
                'v1': np.random.randn(n),
                'v2': np.random.randn(n),
                'v3': np.random.randn(n)
            })
            items = ['v1', 'v2', 'v3']
            
            metrics = self.measure_function(
                compute_cronbach_alpha,
                f'alpha_n{n}',
                df,
                items
            )
            
            if metrics['success']:
                print(f"  n={n:4d}: {metrics['execution_time_sec']:.4f}秒 | {metrics['peak_memory_mb']:.2f}MB")
    
    def benchmark_data_operations(self):
        """测试数据操作性能"""
        print("\n【5】数据操作性能测试")
        print("-" * 60)
        
        # 创建测试数据
        n = 10000
        df1 = pd.DataFrame({
            'user_id': [f'S{i:05d}' for i in range(n)],
            'score1': np.random.randn(n)
        })
        df2 = pd.DataFrame({
            'user_id': [f'S{i:05d}' for i in range(n)],
            'score2': np.random.randn(n)
        })
        
        # 数据合并
        def merge_data():
            return pd.merge(df1, df2, on='user_id')
        
        metrics = self.measure_function(merge_data, 'merge_10k_rows')
        print(f"  合并(10k行): {metrics['execution_time_sec']}秒 | {metrics['memory_delta_mb']}MB")
        
        # 分组聚合
        def groupby_agg():
            df = df1.copy()
            df['group'] = np.random.randint(0, 10, n)
            return df.groupby('group').agg({'score1': ['mean', 'std', 'count']})
        
        metrics = self.measure_function(groupby_agg, 'groupby_agg')
        print(f"  分组聚合: {metrics['execution_time_sec']}秒 | {metrics['memory_delta_mb']}MB")
    
    def generate_report(self):
        """生成基准测试报告"""
        print("\n" + "="*80)
        print("性能基准测试报告")
        print("="*80)
        
        df_results = pd.DataFrame(self.results)
        
        # 按执行时间排序
        df_sorted = df_results.sort_values('execution_time_sec', ascending=False)
        
        print("\n【最耗时的5个操作】")
        print(df_sorted[['function', 'execution_time_sec', 'memory_delta_mb']].head(5).to_string(index=False))
        
        print("\n【最耗内存的5个操作】")
        df_sorted_mem = df_results.sort_values('peak_memory_mb', ascending=False)
        print(df_sorted_mem[['function', 'peak_memory_mb', 'execution_time_sec']].head(5).to_string(index=False))
        
        # 统计信息
        print("\n【总体统计】")
        print(f"  总测试数: {len(df_results)}")
        print(f"  成功数: {df_results['success'].sum()}")
        print(f"  失败数: {(~df_results['success']).sum()}")
        print(f"  总耗时: {df_results['execution_time_sec'].sum():.2f}秒")
        print(f"  平均耗时: {df_results['execution_time_sec'].mean():.4f}秒")
        print(f"  中位耗时: {df_results['execution_time_sec'].median():.4f}秒")
        
        # 保存结果
        output_path = config.OUTPUT_TABLES / 'benchmark_results.csv'
        df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 基准测试结果已保存: {output_path}")
        
        return df_results


def main():
    """主函数"""
    print("="*80)
    print("海棠杯 SaP 研究 - 性能基准测试")
    print("="*80)
    
    benchmark = PerformanceBenchmark()
    
    # 加载数据
    print("\n加载测试数据...")
    df_pre = pd.read_csv(config.HAITANG_PRE)
    df_post = pd.read_csv(config.HAITANG_POST)
    df_behavior = pd.read_csv(config.HAITANG_BEHAVIOR_LOG)
    
    print(f"  前测样本: {len(df_pre)}")
    print(f"  后测样本: {len(df_post)}")
    print(f"  行为日志: {len(df_behavior)}")
    
    # 运行基准测试
    benchmark.benchmark_data_loading()
    benchmark.benchmark_stats_utils(df_pre, df_post)
    benchmark.benchmark_learning_analytics(df_behavior)
    benchmark.benchmark_data_operations()
    benchmark.benchmark_scalability()
    
    # 生成报告
    df_results = benchmark.generate_report()
    
    print("\n" + "="*80)
    print("基准测试完成！")
    print("="*80)


if __name__ == '__main__':
    main()
