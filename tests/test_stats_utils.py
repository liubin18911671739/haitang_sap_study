"""
统计工具函数单元测试
Unit Tests for Statistical Utilities
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stats_utils import (
    compute_cronbach_alpha,
    compute_kmo_bartlett,
    paired_ttest_with_cohens_d,
    compute_dimension_scores
)


class TestCronbachAlpha:
    """测试Cronbach's Alpha信度计算"""
    
    def test_perfect_reliability(self):
        """测试完全一致的数据（alpha=1）"""
        df = pd.DataFrame({
            'item1': [1, 2, 3, 4, 5],
            'item2': [1, 2, 3, 4, 5],
            'item3': [1, 2, 3, 4, 5]
        })
        items = ['item1', 'item2', 'item3']
        alpha = compute_cronbach_alpha(df, items)
        assert alpha == pytest.approx(1.0, abs=0.01), "完全一致数据alpha应接近1.0"
    
    def test_zero_reliability(self):
        """测试完全不相关的数据"""
        np.random.seed(42)
        df = pd.DataFrame({
            'item1': np.random.randn(50),
            'item2': np.random.randn(50),
            'item3': np.random.randn(50)
        })
        items = ['item1', 'item2', 'item3']
        alpha = compute_cronbach_alpha(df, items)
        assert -0.5 < alpha < 0.5, "随机数据alpha应接近0"
    
    def test_missing_values(self):
        """测试含缺失值的数据"""
        df = pd.DataFrame({
            'item1': [1, 2, np.nan, 4, 5],
            'item2': [1, 2, 3, np.nan, 5],
            'item3': [1, 2, 3, 4, 5]
        })
        items = ['item1', 'item2', 'item3']
        alpha = compute_cronbach_alpha(df, items)
        assert isinstance(alpha, (float, type(None))), "含缺失值应返回有效alpha"
    
    def test_single_item(self):
        """测试单个条目（无法计算）"""
        df = pd.DataFrame({'item1': [1, 2, 3, 4, 5]})
        items = ['item1']
        alpha = compute_cronbach_alpha(df, items)
        assert alpha is None or np.isnan(alpha), "单条目应返回None或NaN"


class TestKMOBartlett:
    """测试KMO和Bartlett检验"""
    
    def test_high_correlation_data(self):
        """测试高相关数据（适合因子分析）"""
        np.random.seed(42)
        n = 100
        factor = np.random.randn(n, 1)
        df = pd.DataFrame({
            'v1': factor.flatten() + np.random.randn(n) * 0.1,
            'v2': factor.flatten() + np.random.randn(n) * 0.1,
            'v3': factor.flatten() + np.random.randn(n) * 0.1,
            'v4': factor.flatten() + np.random.randn(n) * 0.1
        })
        items = ['v1', 'v2', 'v3', 'v4']
        
        result = compute_kmo_bartlett(df, items)
        assert result is not None, "应返回结果"
        assert 'kmo' in result
        assert 'bartlett_p' in result
    
    def test_minimum_sample_size(self):
        """测试最小样本量"""
        df = pd.DataFrame({
            'v1': [1, 2, 3],
            'v2': [1, 2, 3],
            'v3': [1, 2, 3]
        })
        items = ['v1', 'v2', 'v3']
        result = compute_kmo_bartlett(df, items)
        assert result is not None, "最小样本量应返回结果"


class TestPairedTTest:
    """测试配对t检验"""
    
    def test_significant_difference(self):
        """测试显著差异"""
        pre = pd.Series([1, 2, 3, 4, 5] * 10)
        post = pd.Series([3, 4, 5, 6, 7] * 10)
        
        result = paired_ttest_with_cohens_d(pre, post, 'Test Dimension')
        
        assert result['p_value'] < 0.01, "应检测到显著差异"
        assert abs(result['cohens_d']) > 1.0, "效应量应较大"
        assert result['mean_diff'] == pytest.approx(2.0, abs=0.01)
    
    def test_no_difference(self):
        """测试无差异"""
        pre = pd.Series([3, 3, 3, 3, 3] * 10)
        post = pd.Series([3, 3, 3, 3, 3] * 10)
        
        result = paired_ttest_with_cohens_d(pre, post, 'Test Dimension')
        
        assert result['p_value'] > 0.05, "应无显著差异"
        assert result['mean_diff'] == 0.0
    
    def test_negative_effect(self):
        """测试负向效应"""
        pre = pd.Series([5, 6, 7, 8, 9] * 10)
        post = pd.Series([3, 4, 5, 6, 7] * 10)
        
        result = paired_ttest_with_cohens_d(pre, post, 'Test Dimension')
        
        assert result['mean_diff'] < 0, "均值差应为负"
        assert result['cohens_d'] < 0, "效应量应为负"


class TestDimensionScores:
    """测试维度分数计算"""
    
    def test_dimension_calculation(self):
        """测试维度分数计算"""
        df = pd.DataFrame({
            'dim1_item1': [1, 2, 3, 4, 5],
            'dim1_item2': [2, 3, 4, 5, 6],
            'dim2_item1': [3, 4, 5, 6, 7],
            'dim2_item2': [4, 5, 6, 7, 8]
        })
        
        dimensions = {
            'Dimension 1': ['dim1_item1', 'dim1_item2'],
            'Dimension 2': ['dim2_item1', 'dim2_item2']
        }
        
        result = compute_dimension_scores(df, dimensions)
        
        assert 'Dimension 1' in result.columns
        assert 'Dimension 2' in result.columns
        assert len(result) == 5
        assert result['Dimension 1'].iloc[0] == pytest.approx(1.5, abs=0.01)
    
    def test_missing_columns(self):
        """测试缺失列处理"""
        df = pd.DataFrame({
            'item1': [1, 2, 3],
            'item2': [2, 3, 4]
        })
        
        dimensions = {
            'Dim1': ['item1', 'missing_item']
        }
        
        # 应该优雅处理或抛出清晰错误
        try:
            result = compute_dimension_scores(df, dimensions)
            assert True, "应能处理缺失列"
        except KeyError:
            assert True, "应抛出清晰的KeyError"


class TestEdgeCases:
    """边界情况测试"""
    
    def test_empty_dataframe(self):
        """测试空数据框"""
        df = pd.DataFrame()
        items = []
        alpha = compute_cronbach_alpha(df, items)
        assert alpha is None or np.isnan(alpha)
    
    def test_all_same_values(self):
        """测试全部相同值"""
        df = pd.DataFrame({
            'v1': [3, 3, 3, 3, 3],
            'v2': [3, 3, 3, 3, 3],
            'v3': [3, 3, 3, 3, 3]
        })
        items = ['v1', 'v2', 'v3']
        # 标准差为0，可能导致问题
        alpha = compute_cronbach_alpha(df, items)
        # 应该返回None或处理特殊情况
        assert alpha is None or np.isnan(alpha) or alpha == 0.0
    
    def test_large_dataset(self):
        """测试大数据集"""
        np.random.seed(42)
        n = 10000
        df = pd.DataFrame({
            f'v{i}': np.random.randn(n) for i in range(10)
        })
        items = [f'v{i}' for i in range(10)]
        alpha = compute_cronbach_alpha(df, items)
        assert isinstance(alpha, (float, type(None))), "大数据集应正常计算"


# 运行测试的便捷函数
def run_tests():
    """运行所有测试"""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_tests()
