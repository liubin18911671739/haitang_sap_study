"""
环境检查脚本 - 验证所有依赖是否正确安装
Environment Check - Verify all dependencies are correctly installed
"""
import sys
from pathlib import Path

print("=" * 80)
print("海棠杯 SaP 研究 - 环境检查")
print("=" * 80)
print(f"\nPython 版本: {sys.version}")
print(f"Python 路径: {sys.executable}")

# 检查依赖包
required_packages = [
    "pandas",
    "numpy",
    "scipy",
    "pingouin",
    "statsmodels",
    "sklearn",
    "factor_analyzer",
    "semopy",
    "lifelines",
    "matplotlib",
    "seaborn",
    "openpyxl",
]

print("\n" + "=" * 80)
print("检查依赖包:")
print("=" * 80)

all_ok = True
for package in required_packages:
    try:
        if package == "sklearn":
            import sklearn
            version = sklearn.__version__
        elif package == "factor_analyzer":
            import factor_analyzer
            version = factor_analyzer.__version__
        else:
            mod = __import__(package)
            version = getattr(mod, "__version__", "未知")
        
        print(f"✓ {package:20s} {version}")
    except ImportError as e:
        print(f"✗ {package:20s} 未安装")
        all_ok = False

# 检查目录结构
print("\n" + "=" * 80)
print("检查目录结构:")
print("=" * 80)

project_root = Path(__file__).parent.parent
required_dirs = [
    "data/external",
    "data/haitang_local",
    "outputs/tables",
    "outputs/figs",
    "outputs/models",
    "src",
]

for dir_path in required_dirs:
    full_path = project_root / dir_path
    if full_path.exists():
        print(f"✓ {dir_path}")
    else:
        print(f"✗ {dir_path} (不存在)")

# 检查源文件
print("\n" + "=" * 80)
print("检查源文件:")
print("=" * 80)

src_files = [
    "src/main.py",
    "src/config.py",
    "src/scales.py",
    "src/stats_utils.py",
    "src/sem_models.py",
    "src/learning_analytics.py",
    "src/qualitative_matrix.py",
]

for file_path in src_files:
    full_path = project_root / file_path
    if full_path.exists():
        print(f"✓ {file_path}")
    else:
        print(f"✗ {file_path} (不存在)")
        all_ok = False

# 总结
print("\n" + "=" * 80)
if all_ok:
    print("✓ 环境检查通过! 可以运行 python src/main.py")
else:
    print("✗ 环境检查失败! 请安装缺失的依赖或创建缺失的文件")
print("=" * 80)
