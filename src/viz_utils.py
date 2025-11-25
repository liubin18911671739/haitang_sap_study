"""
可视化工具模块 - 生成高质量论文级图表
Visualization Utilities - Generate publication-quality figures
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def plot_pre_post_comparison(
    df_pre: pd.DataFrame,
    df_post: pd.DataFrame,
    dimensions: Dict[str, List[str]],
    output_path: Path,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 300
) -> None:
    """
    绘制前后测对比箱线图
    Plot pre-post comparison boxplots
    
    Args:
        df_pre: 前测数据
        df_post: 后测数据
        dimensions: 维度及其条目映射
        output_path: 输出路径
        figsize: 图表大小
        dpi: 分辨率
    """
    # 计算各维度总分
    pre_scores = {}
    post_scores = {}
    
    for dim_name, items in dimensions.items():
        pre_scores[dim_name] = df_pre[items].mean(axis=1)
        post_scores[dim_name] = df_post[items].mean(axis=1)
    
    # 准备数据
    plot_data = []
    for dim_name in dimensions.keys():
        for score, time in [(pre_scores[dim_name], '前测'), (post_scores[dim_name], '后测')]:
            for val in score:
                plot_data.append({
                    '维度': dim_name,
                    '时间': time,
                    '得分': val
                })
    
    df_plot = pd.DataFrame(plot_data)
    
    # 绘图
    fig, ax = plt.subplots(figsize=figsize)
    
    # 箱线图
    sns.boxplot(
        data=df_plot,
        x='维度',
        y='得分',
        hue='时间',
        palette=['#3498db', '#e74c3c'],
        ax=ax
    )
    
    # 添加均值点
    sns.stripplot(
        data=df_plot,
        x='维度',
        y='得分',
        hue='时间',
        dodge=True,
        alpha=0.3,
        size=3,
        ax=ax,
        legend=False
    )
    
    ax.set_title('GenAI素养前后测对比分析', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('维度', fontsize=12)
    ax.set_ylabel('平均得分', fontsize=12)
    ax.legend(title='测试阶段', loc='upper left', fontsize=10)
    ax.set_ylim(1, 5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 前后测对比图已保存: {output_path}")


def plot_scree_plot(
    eigenvalues: np.ndarray,
    output_path: Path,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300
) -> None:
    """
    绘制EFA碎石图
    Plot scree plot for EFA
    
    Args:
        eigenvalues: 特征值数组
        output_path: 输出路径
        figsize: 图表大小
        dpi: 分辨率
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_factors = len(eigenvalues)
    x = np.arange(1, n_factors + 1)
    
    # 绘制特征值折线图
    ax.plot(x, eigenvalues, 'bo-', linewidth=2, markersize=8, label='特征值')
    
    # Kaiser准则线 (特征值=1)
    ax.axhline(y=1, color='r', linestyle='--', linewidth=1.5, label='Kaiser准则 (λ=1)')
    
    # 标注特征值
    for i, val in enumerate(eigenvalues):
        if i < 8:  # 只标注前8个
            ax.annotate(f'{val:.2f}', 
                       xy=(i+1, val),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=9)
    
    ax.set_title('探索性因子分析碎石图', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('因子序号', fontsize=12)
    ax.set_ylabel('特征值', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✅ EFA碎石图已保存: {output_path}")


def plot_sem_path_diagram(
    path_coefficients: pd.DataFrame,
    output_path: Path,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300
) -> None:
    """
    绘制SEM路径图 (简化版网络图)
    Plot SEM path diagram (simplified network)
    
    Args:
        path_coefficients: 路径系数数据框
        output_path: 输出路径
        figsize: 图表大小
        dpi: 分辨率
    """
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 定义节点位置
    nodes = {
        '共创过程': (2, 6),
        'OSE参与度': (5, 6),
        'GenAI素养': (8, 6),
        '社区归属': (2, 3),
        '创新思维': (5, 3),
        '问题解决': (8, 3)
    }
    
    # 绘制节点
    for node_name, (x, y) in nodes.items():
        # 潜变量用椭圆
        if node_name in ['共创过程', 'OSE参与度', 'GenAI素养']:
            circle = mpatches.Ellipse((x, y), 1.5, 0.8, 
                                     facecolor='#3498db', 
                                     edgecolor='black', 
                                     linewidth=2,
                                     alpha=0.7)
        else:  # 观测变量用矩形
            circle = mpatches.Rectangle((x-0.6, y-0.3), 1.2, 0.6,
                                       facecolor='#95a5a6',
                                       edgecolor='black',
                                       linewidth=1.5,
                                       alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, y, node_name, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')
    
    # 绘制路径 (示例路径)
    paths = [
        ('共创过程', 'OSE参与度', 0.45),
        ('OSE参与度', 'GenAI素养', 0.52),
        ('共创过程', '社区归属', 0.38),
        ('OSE参与度', '创新思维', 0.41),
        ('GenAI素养', '问题解决', 0.48)
    ]
    
    for start, end, coef in paths:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        
        # 路径系数绝对值决定箭头粗细和颜色深度
        linewidth = abs(coef) * 5
        alpha = min(abs(coef) + 0.3, 1.0)
        color = '#e74c3c' if coef > 0 else '#34495e'
        
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', 
            mutation_scale=20,
            linewidth=linewidth,
            color=color,
            alpha=alpha,
            connectionstyle="arc3,rad=0.1"
        )
        ax.add_patch(arrow)
        
        # 标注路径系数
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, f'{coef:.2f}*',
               ha='center', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='yellow', alpha=0.7))
    
    ax.set_title('SEM路径分析图 (简化示意)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(facecolor='#3498db', alpha=0.7, label='潜变量'),
        mpatches.Patch(facecolor='#95a5a6', alpha=0.7, label='观测变量'),
        mpatches.Patch(facecolor='#e74c3c', alpha=0.7, label='正向路径'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✅ SEM路径图已保存: {output_path}")


def plot_behavior_clusters(
    df_features: pd.DataFrame,
    cluster_col: str,
    output_path: Path,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300
) -> None:
    """
    绘制行为轨迹聚类散点图 (PCA降维)
    Plot behavior clustering scatter plot with PCA
    
    Args:
        df_features: 特征数据框
        cluster_col: 聚类标签列名
        output_path: 输出路径
        figsize: 图表大小
        dpi: 分辨率
    """
    # 提取数值特征
    feature_cols = df_features.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in feature_cols if col != cluster_col and 'user_id' not in col.lower()]
    
    X = df_features[feature_cols].fillna(0)
    labels = df_features[cluster_col]
    
    # PCA降维到2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # 绘图
    fig, ax = plt.subplots(figsize=figsize)
    
    # 为每个聚类分配颜色
    n_clusters = len(labels.unique())
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    for i, cluster_id in enumerate(sorted(labels.unique())):
        mask = labels == cluster_id
        ax.scatter(
            X_pca[mask, 0], 
            X_pca[mask, 1],
            c=[colors[i]],
            label=f'聚类 {cluster_id}',
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
    
    # 绘制聚类中心
    for cluster_id in sorted(labels.unique()):
        mask = labels == cluster_id
        center = X_pca[mask].mean(axis=0)
        ax.scatter(
            center[0], center[1],
            c='red',
            marker='X',
            s=300,
            edgecolors='black',
            linewidth=2,
            zorder=10
        )
    
    # 解释方差
    var_exp = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)', fontsize=12)
    ax.set_title('学习行为轨迹聚类分析 (PCA降维)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 行为聚类散点图已保存: {output_path}")


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: Path,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 300
) -> None:
    """
    绘制相关系数热力图
    Plot correlation heatmap
    
    Args:
        df: 数据框
        output_path: 输出路径
        figsize: 图表大小
        dpi: 分辨率
    """
    # 计算相关系数
    corr = df.select_dtypes(include=[np.number]).corr()
    
    # 绘图
    fig, ax = plt.subplots(figsize=figsize)
    
    # 热力图
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': '相关系数'},
        ax=ax
    )
    
    ax.set_title('变量相关系数热力图', fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 相关系数热力图已保存: {output_path}")


def generate_all_visualizations(
    data_dict: Dict[str, pd.DataFrame],
    output_dir: Path,
    dpi: int = 300
) -> None:
    """
    生成所有可视化图表
    Generate all visualizations
    
    Args:
        data_dict: 数据字典 {'pre': df_pre, 'post': df_post, ...}
        output_dir: 输出目录
        dpi: 分辨率
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("开始生成可视化图表...")
    print("="*60)
    
    try:
        # 1. 前后测对比图
        if 'pre' in data_dict and 'post' in data_dict:
            from scales import GENAI_LITERACY_ITEMS
            plot_pre_post_comparison(
                data_dict['pre'],
                data_dict['post'],
                GENAI_LITERACY_ITEMS,
                output_dir / 'pre_post_comparison.png',
                dpi=dpi
            )
    except Exception as e:
        print(f"⚠️  前后测对比图生成失败: {e}")
    
    try:
        # 2. EFA碎石图 (需要特征值)
        if 'eigenvalues' in data_dict:
            plot_scree_plot(
                data_dict['eigenvalues'],
                output_dir / 'efa_scree_plot.png',
                dpi=dpi
            )
    except Exception as e:
        print(f"⚠️  EFA碎石图生成失败: {e}")
    
    try:
        # 3. SEM路径图
        if 'path_coef' in data_dict:
            plot_sem_path_diagram(
                data_dict['path_coef'],
                output_dir / 'sem_path_diagram.png',
                dpi=dpi
            )
    except Exception as e:
        print(f"⚠️  SEM路径图生成失败: {e}")
    
    try:
        # 4. 行为聚类散点图
        if 'behavior_clusters' in data_dict:
            plot_behavior_clusters(
                data_dict['behavior_clusters'],
                'cluster',
                output_dir / 'behavior_clusters.png',
                dpi=dpi
            )
    except Exception as e:
        print(f"⚠️  行为聚类图生成失败: {e}")
    
    try:
        # 5. 相关系数热力图
        if 'correlation_data' in data_dict:
            plot_correlation_heatmap(
                data_dict['correlation_data'],
                output_dir / 'correlation_heatmap.png',
                dpi=dpi
            )
    except Exception as e:
        print(f"⚠️  相关系数热力图生成失败: {e}")
    
    print("="*60)
    print("可视化图表生成完成！")
    print(f"输出目录: {output_dir}")
    print("="*60 + "\n")
