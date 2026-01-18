import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import seaborn as sns
from pathlib import Path
import numpy as np
from jinja2 import Template

class Visualizer:
    """可视化报告生成器"""
    
    def __init__(self):
        self.output_dir = Path("reports")
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_complexity_distribution(self, df: pd.DataFrame):
        """绘制复杂度分布图表"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 圈复杂度分布
        ax = axes[0, 0]
        complexity_data = df['cyclomatic_complexity'].dropna()
        ax.hist(complexity_data, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('圈复杂度')
        ax.set_ylabel('文件数量')
        ax.set_title('圈复杂度分布')
        ax.axvline(complexity_data.mean(), color='red', linestyle='--', 
                  label=f'平均值: {complexity_data.mean():.2f}')
        ax.legend()
        
        # 2. 可维护性指数分布
        ax = axes[0, 1]
        mi_data = df['maintainability_index'].dropna()
        ax.hist(mi_data, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel('可维护性指数')
        ax.set_ylabel('文件数量')
        ax.set_title('可维护性指数分布')
        ax.axvline(mi_data.mean(), color='red', linestyle='--',
                  label=f'平均值: {mi_data.mean():.2f}')
        ax.legend()
        
        # 3. 函数长度分布
        ax = axes[1, 0]
        func_lengths = df['avg_lines_per_function'].dropna()
        ax.hist(func_lengths[func_lengths < 100], bins=30, 
                edgecolor='black', alpha=0.7, color='orange')
        ax.set_xlabel('平均函数长度（行数）')
        ax.set_ylabel('文件数量')
        ax.set_title('函数长度分布（<100行）')
        
        # 4. 代码风格问题分布
        ax = axes[1, 1]
        style_issues = df['style_issues'].dropna()
        ax.boxplot(style_issues)
        ax.set_ylabel('代码风格问题数量')
        ax.set_title('代码风格问题分布')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'complexity_distribution.png', dpi=150)
        plt.close()
        
        # Plotly交互式图表
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('圈复杂度分布', 
                           '可维护性指数分布', 
                           '函数长度分布', 
                           '代码风格问题分布')
        )
        
        fig.add_trace(go.Histogram(x=complexity_data, name='圈复杂度'), row=1, col=1)
        fig.add_trace(go.Histogram(x=mi_data, name='可维护性指数'), row=1, col=2)
        fig.add_trace(go.Histogram(x=func_lengths, name='函数长度'), row=2, col=1)
        fig.add_trace(go.Box(y=style_issues, name='代码风格问题'), row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False)
        fig.write_html(self.output_dir / 'complexity_interactive.html')
    
    def plot_style_issues_by_module(self, df: pd.DataFrame):
        """按模块分析代码风格问题"""
        # 提取模块名称
        df['module'] = df['file_path'].apply(
            lambda x: '/'.join(Path(x).parts[:2]) if '/' in x else 'root'
        )
        
        # 按模块分组
        module_stats = df.groupby('module').agg({
            'style_issues': 'sum',
            'file_path': 'count',
            'cyclomatic_complexity': 'mean'
        }).rename(columns={'file_path': 'file_count'})
        
        # 问题最多的15个模块
        top_modules = module_stats.nlargest(15, 'style_issues')
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 问题数量条形图
        ax = axes[0]
        bars = ax.barh(range(len(top_modules)), top_modules['style_issues'])
        ax.set_yticks(range(len(top_modules)))
        ax.set_yticklabels(top_modules.index)
        ax.set_xlabel('代码风格问题数量')
        ax.set_title('按模块统计代码风格问题（Top 15）')
        ax.invert_yaxis()
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 3, bar.get_y() + bar.get_height()/2,
                   f'{int(width)}', ha='left', va='center')
        
        # 问题密度散点图
        ax = axes[1]
        scatter = ax.scatter(
            module_stats['file_count'],
            module_stats['style_issues'] / module_stats['file_count'],
            c=module_stats['cyclomatic_complexity'],
            s=100,
            alpha=0.6,
            cmap='viridis'
        )
        ax.set_xlabel('文件数量')
        ax.set_ylabel('平均每个文件问题数')
        ax.set_title('模块问题密度 vs 文件数量')
        
        plt.colorbar(scatter, ax=ax, label='平均圈复杂度')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'style_issues_by_module.png', dpi=150)
        plt.close()
    
    def plot_function_metrics(self, df: pd.DataFrame):
        """函数级别指标分析"""
        fig = plt.figure(figsize=(14, 10))
        
        # 相关性热力图
        numeric_cols = [
            'cyclomatic_complexity', 'maintainability_index',
            'avg_lines_per_function', 'avg_args_per_function',
            'style_issues', 'function_count'
        ]
        
        # 过滤存在的列
        existing_cols = [col for col in numeric_cols if col in df.columns]
        corr_matrix = df[existing_cols].corr()
        
        ax = fig.add_subplot(2, 2, 1)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                   center=0, ax=ax, square=True)
        ax.set_title('指标相关性热力图')
        
        # 复杂度 vs 可维护性
        ax = fig.add_subplot(2, 2, 2)
        scatter = ax.scatter(
            df['cyclomatic_complexity'],
            df['maintainability_index'],
            c=df['style_issues'],
            s=50,
            alpha=0.6,
            cmap='RdYlBu_r'
        )
        ax.set_xlabel('圈复杂度')
        ax.set_ylabel('可维护性指数')
        ax.set_title('复杂度 vs 可维护性指数')
        plt.colorbar(scatter, ax=ax, label='代码风格问题数量')
        
        # 按文件类型分组函数长度分布
        ax = fig.add_subplot(2, 2, 3)
        
        # 按文件类型分组（示例：基于路径）
        df['file_type'] = df['file_path'].apply(
            lambda x: '测试文件' if 'test' in x.lower() else 
                     '工具文件' if 'util' in x.lower() else '核心文件'
        )
        
        data_to_plot = []
        labels = []
        for file_type in ['核心文件', '工具文件', '测试文件']:
            if file_type in df['file_type'].values:
                data = df[df['file_type'] == file_type]['avg_lines_per_function']
                data_to_plot.append(data[data < 100].values)  # 移除异常值
                labels.append(file_type)
        
        ax.boxplot(data_to_plot, labels=labels)
        ax.set_ylabel('平均函数长度（行数）')
        ax.set_title('按文件类型分组函数长度分布')
        
        # 代码风格问题类型分布
        ax = fig.add_subplot(2, 2, 4)
        
        # 模拟问题类型分布
        issue_types = {
            'E2': 25,  # 空格相关问题
            'E3': 18,  # 缩进问题
            'E5': 12,  # 行长度问题
            'W1': 8,   # 警告
            'C9': 15,  # 复杂度问题
            '其他': 22
        }
        
        ax.pie(issue_types.values(), labels=issue_types.keys(),
              autopct='%1.1f%%', startangle=90)
        ax.set_title('代码风格问题类型分布')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'function_metrics_analysis.png', dpi=150)
        plt.close()
    
    def generate_html_report(self, df: pd.DataFrame):
        """生成详细HTML报告"""
        # 计算总体统计
        total_files = len(df)
        avg_complexity = df['cyclomatic_complexity'].mean()
        avg_maintainability = df['maintainability_index'].mean()
        total_style_issues = df['style_issues'].sum()
        
        # 查找最复杂的文件
        most_complex = df.nlargest(5, 'cyclomatic_complexity')[['file_path', 'cyclomatic_complexity']]
        most_issues = df.nlargest(5, 'style_issues')[['file_path', 'style_issues']]
        
        # 按模块分组
        df['module'] = df['file_path'].apply(
            lambda x: str(Path(x).parent)
        )
        module_stats = df.groupby('module').agg({
            'cyclomatic_complexity': 'mean',
            'style_issues': 'sum',
            'file_path': 'count'
        }).rename(columns={'file_path': 'file_count'})
        
        # HTML模板（中文版）
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>pandas 代码质量分析报告</title>
            <meta charset="UTF-8">
            <style>
                body { font-family: "Microsoft YaHei", Arial, sans-serif; margin: 40px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
                .metric-card { background: #f8f9fa; padding: 20px; margin: 10px 0; border-radius: 5px; }
                .metric-value { font-size: 2em; font-weight: bold; color: #3498db; }
                .table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                .table th, .table td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                .table th { background-color: #4CAF50; color: white; }
                .table tr:nth-child(even) { background-color: #f2f2f2; }
                .highlight { background-color: #fff3cd; padding: 10px; border-radius: 5px; }
                .chart-container { margin: 30px 0; }
                img { max-width: 100%; height: auto; }
                .improvement-list { background: #e7f3fe; padding: 15px; border-left: 4px solid #2196F3; }
                .critical { color: #e74c3c; font-weight: bold; }
                .warning { color: #f39c12; }
                .good { color: #27ae60; }
                .metric-grid { display: flex; justify-content: space-around; text-align: center; flex-wrap: wrap; }
                .metric-item { flex: 1; min-width: 200px; margin: 10px; }
                .chart-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
                @media (max-width: 768px) {
                    .chart-grid { grid-template-columns: 1fr; }
                    .metric-grid { flex-direction: column; }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 pandas 代码质量分析报告</h1>
                    <p>生成时间: {{ timestamp }}</p>
                </div>
                
                <div class="metric-card">
                    <h2>📈 总体统计</h2>
                    <div class="metric-grid">
                        <div class="metric-item">
                            <div class="metric-value">{{ total_files }}</div>
                            <div>分析文件总数</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value {% if avg_complexity > 15 %}critical{% elif avg_complexity > 10 %}warning{% else %}good{% endif %}">
                                {{ avg_complexity | round(2) }}
                            </div>
                            <div>平均圈复杂度</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value {% if avg_maintainability < 65 %}critical{% elif avg_maintainability < 85 %}warning{% else %}good{% endif %}">
                                {{ avg_maintainability | round(2) }}
                            </div>
                            <div>平均可维护性指数</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value {% if total_style_issues > 100 %}critical{% elif total_style_issues > 50 %}warning{% else %}good{% endif %}">
                                {{ total_style_issues }}
                            </div>
                            <div>代码风格问题总数</div>
                        </div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h2>📊 可视化图表</h2>
                    <div class="chart-grid">
                        <div>
                            <h3>复杂度分布</h3>
                            <img src="complexity_distribution.png" alt="复杂度分布">
                        </div>
                        <div>
                            <h3>按模块统计代码风格问题</h3>
                            <img src="style_issues_by_module.png" alt="按模块统计代码风格问题">
                        </div>
                        <div style="grid-column: span 2;">
                            <h3>函数指标分析</h3>
                            <img src="function_metrics_analysis.png" alt="函数指标分析">
                        </div>
                    </div>
                </div>
                
                <div class="highlight">
                    <h2>⚠️ 需要重点关注的关键区域</h2>
                    <p>以下文件在代码质量和复杂度方面需要特别关注：</p>
                </div>
                
                <h3>🔴 最复杂文件（Top 5）</h3>
                <table class="table">
                    <tr>
                        <th>文件路径</th>
                        <th>圈复杂度</th>
                        <th>状态</th>
                    </tr>
                    {% for row in most_complex %}
                    <tr>
                        <td>{{ row.file_path }}</td>
                        <td>{{ row.cyclomatic_complexity | round(2) }}</td>
                        <td>
                            {% if row.cyclomatic_complexity > 20 %}
                                <span class="critical">严重</span>
                            {% elif row.cyclomatic_complexity > 15 %}
                                <span class="warning">警告</span>
                            {% else %}
                                <span class="good">良好</span>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </table>
                
                <h3>⚠️ 代码风格问题最多文件（Top 5）</h3>
                <table class="table">
                    <tr>
                        <th>文件路径</th>
                        <th>代码风格问题数量</th>
                        <th>状态</th>
                    </tr>
                    {% for row in most_issues %}
                    <tr>
                        <td>{{ row.file_path }}</td>
                        <td>{{ row.style_issues }}</td>
                        <td>
                            {% if row.style_issues > 20 %}
                                <span class="critical">严重</span>
                            {% elif row.style_issues > 10 %}
                                <span class="warning">警告</span>
                            {% else %}
                                <span class="good">良好</span>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </table>
                
                <h3>📁 模块统计（Top 10）</h3>
                <table class="table">
                    <tr>
                        <th>模块</th>
                        <th>文件数量</th>
                        <th>平均复杂度</th>
                        <th>问题总数</th>
                        <th>平均每个文件问题数</th>
                    </tr>
                    {% for module, stats in module_stats.head(10).iterrows() %}
                    <tr>
                        <td>{{ module }}</td>
                        <td>{{ stats.file_count }}</td>
                        <td {% if stats.cyclomatic_complexity > 15 %}class="critical"{% elif stats.cyclomatic_complexity > 10 %}class="warning"{% endif %}>
                            {{ stats.cyclomatic_complexity | round(2) }}
                        </td>
                        <td>{{ stats.style_issues }}</td>
                        <td>{{ (stats.style_issues / stats.file_count) | round(2) }}</td>
                    </tr>
                    {% endfor %}
                </table>
                
                <div class="metric-card improvement-list">
                    <h2>💡 改进建议</h2>
                    <ul>
                        <li><strong>重构</strong> 圈复杂度大于20的函数</li>
                        <li><strong>添加文档字符串</strong> 到缺少文档的函数</li>
                        <li><strong>遵循PEP 8规范</strong> 并修复代码风格违规</li>
                        <li><strong>拆分长函数</strong> 为更小、更专注的单元</li>
                        <li><strong>提高测试覆盖率</strong> 针对关键模块</li>
                        <li><strong>减少函数参数</strong> 以提高可读性</li>
                        <li><strong>添加类型提示</strong> 以提高代码清晰度</li>
                        <li><strong>审查复杂模块</strong> 针对高问题密度的模块</li>
                    </ul>
                    
                    <h3>🏆 质量评估标准</h3>
                    <table class="table">
                        <tr>
                            <th>指标</th>
                            <th>优秀</th>
                            <th>良好</th>
                            <th>需要改进</th>
                            <th>严重</th>
                        </tr>
                        <tr>
                            <td>圈复杂度</td>
                            <td class="good">&lt; 10</td>
                            <td class="good">10-15</td>
                            <td class="warning">15-20</td>
                            <td class="critical">&gt; 20</td>
                        </tr>
                        <tr>
                            <td>可维护性指数</td>
                            <td class="good">&gt; 85</td>
                            <td class="good">65-85</td>
                            <td class="warning">50-65</td>
                            <td class="critical">&lt; 50</td>
                        </tr>
                        <tr>
                            <td>每个文件代码风格问题数</td>
                            <td class="good">&lt; 5</td>
                            <td class="good">5-10</td>
                            <td class="warning">10-20</td>
                            <td class="critical">&gt; 20</td>
                        </tr>
                        <tr>
                            <td>函数长度（行数）</td>
                            <td class="good">&lt; 20</td>
                            <td class="good">20-50</td>
                            <td class="warning">50-100</td>
                            <td class="critical">&gt; 100</td>
                        </tr>
                    </table>
                </div>
                
                <div style="margin-top: 30px; padding: 15px; background: #f8f9fa; border-radius: 5px;">
                    <h3>📋 分析总结</h3>
                    <p><strong>总体质量评级：</strong> 
                        {% if avg_complexity < 10 and avg_maintainability > 85 and total_style_issues/total_files < 5 %}
                        <span class="good">优秀</span> 🏆
                        {% elif avg_complexity < 15 and avg_maintainability > 65 and total_style_issues/total_files < 10 %}
                        <span class="good">良好</span> 👍
                        {% elif avg_complexity < 20 and avg_maintainability > 50 %}
                        <span class="warning">需要改进</span> ⚠️
                        {% else %}
                        <span class="critical">需要紧急处理</span> 🚨
                        {% endif %}
                    </p>
                    <p><strong>关键发现：</strong></p>
                    <ul>
                        <li>分析文件总数：{{ total_files }}</li>
                        <li>平均圈复杂度：{{ avg_complexity | round(2) }}</li>
                        <li>平均可维护性指数：{{ avg_maintainability | round(2) }}</li>
                        <li>发现代码风格问题总数：{{ total_style_issues }}</li>
                        <li>平均每个文件问题数：{{ (total_style_issues/total_files) | round(2) if total_files > 0 else 0 }}</li>
                    </ul>
                </div>
                
                <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #666;">
                    <p>报告由 pandas 代码质量分析器生成</p>
                    <p>交互式图表请查看：<code>reports/complexity_interactive.html</code></p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        # 渲染模板
        from datetime import datetime
        template = Template(html_template)
        html_content = template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_files=total_files,
            avg_complexity=avg_complexity,
            avg_maintainability=avg_maintainability,
            total_style_issues=total_style_issues,
            most_complex=most_complex.to_dict('records'),
            most_issues=most_issues.to_dict('records'),
            module_stats=module_stats
        )
        
        # 保存HTML文件
        report_path = self.output_dir / 'analysis_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML报告已生成: {report_path}")
        print(f"交互式图表: {self.output_dir}/complexity_interactive.html")