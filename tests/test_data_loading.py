"""
数据加载容错性测试
Unit Tests for Data Loading Robustness
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
import config


class TestDataFileReading:
    """测试数据文件读取"""
    
    def test_csv_encoding(self):
        """测试不同编码的CSV"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write('user_id,score\n')
            f.write('S001,4.5\n')
            f.write('S002,3.8\n')
            temp_path = f.name
        
        try:
            df = pd.read_csv(temp_path)
            assert len(df) == 2
            assert 'user_id' in df.columns
        finally:
            os.unlink(temp_path)
    
    def test_missing_file(self):
        """测试文件不存在"""
        with pytest.raises(FileNotFoundError):
            pd.read_csv('nonexistent_file.csv')
    
    def test_empty_file(self):
        """测试空文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(pd.errors.EmptyDataError):
                pd.read_csv(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_malformed_csv(self):
        """测试格式错误的CSV"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('col1,col2\n')
            f.write('val1,val2,val3\n')  # 列数不匹配
            temp_path = f.name
        
        try:
            df = pd.read_csv(temp_path, on_bad_lines='skip')
            assert len(df) == 0 or len(df) == 1
        finally:
            os.unlink(temp_path)


class TestMissingValueHandling:
    """测试缺失值处理"""
    
    def test_na_detection(self):
        """测试缺失值检测"""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [1, None, 3, 4],
            'col3': [1, 2, 3, '']
        })
        
        assert df['col1'].isna().sum() == 1
        assert df['col2'].isna().sum() == 1
    
    def test_dropna_behavior(self):
        """测试dropna行为"""
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [1, np.nan, 3, 4]
        })
        
        df_dropped = df.dropna()
        assert len(df_dropped) == 2
        
        df_dropped_any = df.dropna(how='any')
        assert len(df_dropped_any) == 2
        
        df_dropped_all = df.dropna(how='all')
        assert len(df_dropped_all) == 4
    
    def test_fillna_strategies(self):
        """测试填充策略"""
        df = pd.DataFrame({
            'score': [1.0, 2.0, np.nan, 4.0, 5.0]
        })
        
        # 均值填充
        df_mean = df.fillna(df['score'].mean())
        assert df_mean['score'].isna().sum() == 0
        assert df_mean['score'].iloc[2] == pytest.approx(3.0, abs=0.01)
        
        # 前向填充
        df_ffill = df.fillna(method='ffill')
        assert df_ffill['score'].iloc[2] == 2.0
        
        # 后向填充
        df_bfill = df.fillna(method='bfill')
        assert df_bfill['score'].iloc[2] == 4.0


class TestDataTypeValidation:
    """测试数据类型验证"""
    
    def test_numeric_conversion(self):
        """测试数值转换"""
        df = pd.DataFrame({
            'score': ['1', '2', '3.5', '4']
        })
        
        df['score'] = pd.to_numeric(df['score'])
        assert df['score'].dtype in [np.float64, np.int64]
    
    def test_invalid_numeric(self):
        """测试无效数值"""
        df = pd.DataFrame({
            'score': ['1', '2', 'invalid', '4']
        })
        
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        assert df['score'].isna().sum() == 1
    
    def test_date_parsing(self):
        """测试日期解析"""
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', 'invalid']
        })
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        assert df['date'].isna().sum() == 1
        assert pd.api.types.is_datetime64_any_dtype(df['date'])


class TestColumnNameHandling:
    """测试列名处理"""
    
    def test_duplicate_columns(self):
        """测试重复列名"""
        data = {
            'col': [1, 2, 3],
            'col': [4, 5, 6]  # 重复列名
        }
        df = pd.DataFrame(data)
        # pandas会自动处理为 'col' 和 'col.1'
        assert len(df.columns) >= 1
    
    def test_special_characters(self):
        """测试特殊字符列名"""
        df = pd.DataFrame({
            'col with spaces': [1, 2, 3],
            'col-with-dash': [4, 5, 6],
            'col.with.dot': [7, 8, 9]
        })
        
        assert 'col with spaces' in df.columns
        assert df['col with spaces'].iloc[0] == 1
    
    def test_chinese_columns(self):
        """测试中文列名"""
        df = pd.DataFrame({
            '用户ID': ['S001', 'S002'],
            '得分': [4.5, 3.8]
        })
        
        assert '用户ID' in df.columns
        assert df['得分'].mean() == pytest.approx(4.15, abs=0.01)


class TestDataMerging:
    """测试数据合并"""
    
    def test_inner_join(self):
        """测试内连接"""
        df1 = pd.DataFrame({
            'user_id': ['S001', 'S002', 'S003'],
            'score_pre': [3.0, 4.0, 5.0]
        })
        df2 = pd.DataFrame({
            'user_id': ['S002', 'S003', 'S004'],
            'score_post': [4.5, 5.5, 6.0]
        })
        
        merged = pd.merge(df1, df2, on='user_id', how='inner')
        assert len(merged) == 2
        assert 'S001' not in merged['user_id'].values
    
    def test_outer_join(self):
        """测试外连接"""
        df1 = pd.DataFrame({
            'user_id': ['S001', 'S002'],
            'score': [3.0, 4.0]
        })
        df2 = pd.DataFrame({
            'user_id': ['S002', 'S003'],
            'score': [4.5, 5.5]
        })
        
        merged = pd.merge(df1, df2, on='user_id', how='outer', suffixes=('_pre', '_post'))
        assert len(merged) == 3
        assert merged['score_pre'].isna().sum() == 1
    
    def test_duplicate_keys(self):
        """测试重复键"""
        df1 = pd.DataFrame({
            'user_id': ['S001', 'S001', 'S002'],
            'action': ['login', 'submit', 'login']
        })
        df2 = pd.DataFrame({
            'user_id': ['S001', 'S002'],
            'name': ['Alice', 'Bob']
        })
        
        merged = pd.merge(df1, df2, on='user_id', how='left')
        assert len(merged) == 3
        assert merged['name'].iloc[0] == 'Alice'


class TestRobustnessChecks:
    """鲁棒性检查"""
    
    def test_sample_size_validation(self):
        """测试样本量验证"""
        df_small = pd.DataFrame({'v1': [1, 2], 'v2': [3, 4]})
        df_large = pd.DataFrame({'v1': range(100), 'v2': range(100)})
        
        assert len(df_small) < 30, "小样本应被识别"
        assert len(df_large) >= 30, "大样本应满足要求"
    
    def test_outlier_detection(self):
        """测试异常值检测"""
        df = pd.DataFrame({
            'score': [1, 2, 3, 4, 100]  # 100是异常值
        })
        
        # IQR方法
        Q1 = df['score'].quantile(0.25)
        Q3 = df['score'].quantile(0.75)
        IQR = Q3 - Q1
        
        outliers = df[(df['score'] < Q1 - 1.5*IQR) | (df['score'] > Q3 + 1.5*IQR)]
        assert len(outliers) > 0
        assert 100 in outliers['score'].values
    
    def test_data_range_validation(self):
        """测试数据范围验证"""
        df = pd.DataFrame({
            'likert_score': [1, 2, 3, 6, 4]  # 6超出1-5范围
        })
        
        invalid = df[(df['likert_score'] < 1) | (df['likert_score'] > 5)]
        assert len(invalid) == 1


def run_tests():
    """运行所有测试"""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_tests()
