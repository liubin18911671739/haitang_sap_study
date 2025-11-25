"""
列名验证工具 - 检查数据文件列名与scales.py中的映射是否匹配
Column Name Validator - Check if data file columns match scales.py mapping
"""
import pandas as pd
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from scales import (
    GENAI_LITERACY_ITEMS,
    COCREATE_ITEMS,
    OSE_ITEMS,
    ALL_GENAI_ITEMS,
)
import config

def validate_columns(df, expected_cols, data_name):
    """验证数据框列名"""
    missing = [col for col in expected_cols if col not in df.columns]
    extra = [col for col in df.columns if col not in expected_cols and col != 'user_id']
    
    if not missing and not extra:
        print(f"  ✅ {data_name}: 所有列名匹配")
        return True
    else:
        print(f"  ⚠️  {data_name}:")
        if missing:
            print(f"     缺失列: {missing}")
        if extra:
            print(f"     额外列: {extra}")
        return False

print("="*80)
print("列名映射验证")
print("="*80)

all_valid = True

# 1. 验证前测数据
print("\n【1】前测/后测数据 (GenAI 四维)")
if config.HAITANG_PRE.exists():
    df_pre = pd.read_csv(config.HAITANG_PRE)
    expected = ['user_id'] + ALL_GENAI_ITEMS
    all_valid &= validate_columns(df_pre, expected, "haitang_pre.csv")
else:
    print("  ⚠️  文件不存在: haitang_pre.csv")

# 2. 验证共创数据
print("\n【2】共创过程数据 (七维)")
if config.HAITANG_COCREATE.exists():
    df_cocreate = pd.read_csv(config.HAITANG_COCREATE)
    expected = ['user_id'] + COCREATE_ITEMS
    all_valid &= validate_columns(df_cocreate, expected, "haitang_cocreate.csv")
else:
    print("  ⚠️  文件不存在: haitang_cocreate.csv")

# 3. 验证参与度数据
print("\n【3】OSE 参与度数据 (四维)")
if config.HAITANG_ENGAGEMENT.exists():
    df_engage = pd.read_csv(config.HAITANG_ENGAGEMENT)
    expected = ['user_id'] + OSE_ITEMS
    # 参与度数据可能还包含product_score和high_quality
    expected_full = expected + ['product_score', 'high_quality']
    all_valid &= validate_columns(df_engage, expected_full, "haitang_engagement_ose.csv")
else:
    print("  ⚠️  文件不存在: haitang_engagement_ose.csv")

print("\n" + "="*80)
if all_valid:
    print("✅ 所有列名映射验证通过！")
    print("可以直接运行: .venv/bin/python3 src/main.py")
else:
    print("⚠️  发现列名不匹配，请:")
    print("1. 检查数据文件的实际列名")
    print("2. 修改 src/scales.py 中的映射配置")
    print("3. 重新运行此验证脚本")
print("="*80)
