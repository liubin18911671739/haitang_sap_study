#!/usr/bin/env python3
"""
测试运行器 - 执行所有单元测试并生成覆盖率报告
Test Runner - Execute all unit tests and generate coverage report
"""
import subprocess
import sys
from pathlib import Path

def run_tests():
    """运行所有测试"""
    project_root = Path(__file__).parent
    python_bin = project_root / '.venv' / 'bin' / 'python'
    pytest_bin = project_root / '.venv' / 'bin' / 'pytest'
    
    print("="*80)
    print("运行单元测试")
    print("="*80)
    
    # 运行测试并生成覆盖率报告
    cmd = [
        str(pytest_bin),
        'tests/',
        '-v',
        '--tb=short',
        '--cov=src',
        '--cov-report=term-missing',
        '--cov-report=html:outputs/coverage'
    ]
    
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode == 0:
        print("\n✅ 所有测试通过！")
    else:
        print(f"\n⚠️  部分测试失败 (退出码: {result.returncode})")
    
    print(f"\n覆盖率报告已生成: {project_root}/outputs/coverage/index.html")
    print("在浏览器中打开查看详细覆盖率信息")
    
    return result.returncode

if __name__ == '__main__':
    sys.exit(run_tests())
