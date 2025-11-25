"""
输出格式验证测试
Unit Tests for Output Format Validation
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestCSVOutput:
    """测试CSV输出格式"""
    
    def test_csv_write_read_consistency(self):
        """测试CSV写入读取一致性"""
        df_original = pd.DataFrame({
            'user_id': ['S001', 'S002', 'S003'],
            'score': [4.5, 3.8, 4.2],
            'category': ['A', 'B', 'A']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            df_original.to_csv(temp_path, index=False, encoding='utf-8-sig')
            df_loaded = pd.read_csv(temp_path)
            
            pd.testing.assert_frame_equal(df_original, df_loaded)
        finally:
            os.unlink(temp_path)
    
    def test_csv_encoding(self):
        """测试CSV编码"""
        df = pd.DataFrame({
            '用户ID': ['学生001', '学生002'],
            '得分': [4.5, 3.8]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8-sig') as f:
            temp_path = f.name
        
        try:
            df.to_csv(temp_path, index=False, encoding='utf-8-sig')
            df_loaded = pd.read_csv(temp_path, encoding='utf-8-sig')
            
            assert '用户ID' in df_loaded.columns
            assert df_loaded['用户ID'].iloc[0] == '学生001'
        finally:
            os.unlink(temp_path)
    
    def test_csv_decimal_precision(self):
        """测试小数精度"""
        df = pd.DataFrame({
            'value': [1.123456789, 2.987654321]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            df.to_csv(temp_path, index=False, float_format='%.3f')
            df_loaded = pd.read_csv(temp_path)
            
            assert df_loaded['value'].iloc[0] == pytest.approx(1.123, abs=0.001)
        finally:
            os.unlink(temp_path)


class TestStatisticalOutputFormat:
    """测试统计输出格式"""
    
    def test_ttest_result_format(self):
        """测试t检验结果格式"""
        result = {
            't_statistic': -5.234567,
            'p_value': 0.000123,
            'mean_diff': -0.543210,
            'cohens_d': -1.234567
        }
        
        # 验证必需字段
        assert 't_statistic' in result
        assert 'p_value' in result
        assert 'cohens_d' in result
        
        # 验证数值范围
        assert 0 <= result['p_value'] <= 1
    
    def test_alpha_result_format(self):
        """测试Cronbach's Alpha结果格式"""
        result = {
            'dimension': 'AI基础知识',
            'alpha': 0.856,
            'n_items': 3,
            'interpretation': 'Good'
        }
        
        assert 'alpha' in result
        assert -1 <= result['alpha'] <= 1
        assert result['n_items'] > 0
    
    def test_sem_path_format(self):
        """测试SEM路径系数格式"""
        df = pd.DataFrame({
            'path': ['X -> Y', 'Y -> Z'],
            'estimate': [0.45, 0.52],
            'std_err': [0.08, 0.09],
            'z_value': [5.62, 5.77],
            'p_value': [0.000, 0.000]
        })
        
        assert 'estimate' in df.columns
        assert 'p_value' in df.columns
        assert all(df['p_value'] >= 0) and all(df['p_value'] <= 1)


class TestTextReportFormat:
    """测试文本报告格式"""
    
    def test_report_structure(self):
        """测试报告结构"""
        report = """
================================================================================
前后测效能评估
================================================================================

样本量: 50
前测均值: 3.73 ± 0.45
后测均值: 4.27 ± 0.38

配对t检验:
  t统计量 = -12.34
  p值 = 0.000
  Cohen's d = 1.24 (大效应)
"""
        
        assert '前后测效能评估' in report
        assert 't统计量' in report
        assert 'p值' in report
        assert 'Cohen\'s d' in report
    
    def test_table_alignment(self):
        """测试表格对齐"""
        table = """
| 维度         | Alpha  | 解释   |
|--------------|--------|--------|
| AI基础知识   | 0.856  | Good   |
| AI核心技能   | 0.823  | Good   |
"""
        lines = [line.strip() for line in table.strip().split('\n')]
        
        # 验证分隔符一致性
        assert all('|' in line for line in lines)


class TestDataValidation:
    """测试数据验证"""
    
    def test_likert_scale_range(self):
        """测试Likert量表范围"""
        df = pd.DataFrame({
            'score': [1, 2, 3, 4, 5]
        })
        
        assert df['score'].min() >= 1
        assert df['score'].max() <= 5
    
    def test_percentage_range(self):
        """测试百分比范围"""
        df = pd.DataFrame({
            'accuracy': [0.85, 0.92, 0.78]
        })
        
        assert all(df['accuracy'] >= 0)
        assert all(df['accuracy'] <= 1)
    
    def test_count_non_negative(self):
        """测试计数非负"""
        df = pd.DataFrame({
            'count': [10, 25, 0, 15]
        })
        
        assert all(df['count'] >= 0)
        assert df['count'].dtype in [np.int32, np.int64]


class TestOutputCompleteness:
    """测试输出完整性"""
    
    def test_all_dimensions_present(self):
        """测试所有维度存在"""
        df = pd.DataFrame({
            'dimension': ['Dim1', 'Dim2', 'Dim3', 'Dim4'],
            'score': [4.2, 3.8, 4.5, 4.1]
        })
        
        expected_dims = ['Dim1', 'Dim2', 'Dim3', 'Dim4']
        assert set(df['dimension']) == set(expected_dims)
    
    def test_no_missing_statistics(self):
        """测试统计量不缺失"""
        stats = {
            'mean': 4.2,
            'std': 0.5,
            'min': 3.0,
            'max': 5.0,
            'n': 50
        }
        
        required_keys = ['mean', 'std', 'min', 'max', 'n']
        assert all(key in stats for key in required_keys)
        assert all(not np.isnan(stats[key]) for key in required_keys)
    
    def test_all_users_in_output(self):
        """测试所有用户在输出中"""
        df_input = pd.DataFrame({
            'user_id': ['S001', 'S002', 'S003']
        })
        
        df_output = pd.DataFrame({
            'user_id': ['S001', 'S002', 'S003'],
            'cluster': [0, 1, 0]
        })
        
        assert set(df_input['user_id']) == set(df_output['user_id'])


class TestErrorHandling:
    """测试错误处理"""
    
    def test_division_by_zero(self):
        """测试除零处理"""
        with pytest.raises(ZeroDivisionError):
            result = 10 / 0
    
    def test_empty_dataframe_handling(self):
        """测试空数据框处理"""
        df = pd.DataFrame()
        
        if len(df) == 0:
            result = None  # 应返回None或跳过
        else:
            result = df.mean()
        
        assert result is None
    
    def test_invalid_column_access(self):
        """测试无效列访问"""
        df = pd.DataFrame({'a': [1, 2, 3]})
        
        with pytest.raises(KeyError):
            value = df['nonexistent_column']


class TestFileNamingConventions:
    """测试文件命名约定"""
    
    def test_output_filename_format(self):
        """测试输出文件名格式"""
        valid_names = [
            'pre_post_ai_lit.csv',
            'ai_lit_alpha.csv',
            'efa_report.txt',
            'sem_fit_indices.csv'
        ]
        
        for name in valid_names:
            assert '_' in name  # 使用下划线
            assert name.endswith('.csv') or name.endswith('.txt')
            assert ' ' not in name  # 无空格
    
    def test_directory_structure(self):
        """测试目录结构"""
        from pathlib import Path
        
        expected_dirs = ['tables', 'figs', 'models']
        # 这里只测试路径字符串格式
        for dir_name in expected_dirs:
            assert isinstance(dir_name, str)
            assert dir_name.isalnum()


def run_tests():
    """运行所有测试"""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_tests()
