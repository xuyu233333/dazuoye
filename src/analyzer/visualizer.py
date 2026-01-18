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
    """Visualization Report Generator"""
    
    def __init__(self):
        self.output_dir = Path("reports")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_complexity_distribution(self, df: pd.DataFrame):
        """Plot complexity distribution charts"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Cyclomatic complexity distribution
        ax = axes[0, 0]
        complexity_data = df['cyclomatic_complexity'].dropna()
        ax.hist(complexity_data, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Cyclomatic Complexity')
        ax.set_ylabel('Number of Files')
        ax.set_title('Cyclomatic Complexity Distribution')
        ax.axvline(complexity_data.mean(), color='red', linestyle='--', 
                  label=f'Mean: {complexity_data.mean():.2f}')
        ax.legend()
        
        # 2. Maintainability index distribution
        ax = axes[0, 1]
        mi_data = df['maintainability_index'].dropna()
        ax.hist(mi_data, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel('Maintainability Index')
        ax.set_ylabel('Number of Files')
        ax.set_title('Maintainability Index Distribution')
        ax.axvline(mi_data.mean(), color='red', linestyle='--',
                  label=f'Mean: {mi_data.mean():.2f}')
        ax.legend()
        
        # 3. Function length distribution
        ax = axes[1, 0]
        func_lengths = df['avg_lines_per_function'].dropna()
        ax.hist(func_lengths[func_lengths < 100], bins=30, 
                edgecolor='black', alpha=0.7, color='orange')
        ax.set_xlabel('Average Function Length (Lines)')
        ax.set_ylabel('Number of Files')
        ax.set_title('Function Length Distribution (<100 lines)')
        
        # 4. Style issues distribution
        ax = axes[1, 1]
        style_issues = df['style_issues'].dropna()
        ax.boxplot(style_issues)
        ax.set_ylabel('Number of Style Issues')
        ax.set_title('Code Style Issues Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'complexity_distribution.png', dpi=150)
        plt.close()
        
        # Plotly interactive charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cyclomatic Complexity Distribution', 
                           'Maintainability Index Distribution', 
                           'Function Length Distribution', 
                           'Style Issues Distribution')
        )
        
        fig.add_trace(go.Histogram(x=complexity_data, name='Cyclomatic Complexity'), row=1, col=1)
        fig.add_trace(go.Histogram(x=mi_data, name='Maintainability Index'), row=1, col=2)
        fig.add_trace(go.Histogram(x=func_lengths, name='Function Length'), row=2, col=1)
        fig.add_trace(go.Box(y=style_issues, name='Style Issues'), row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False)
        fig.write_html(self.output_dir / 'complexity_interactive.html')
    
    def plot_style_issues_by_module(self, df: pd.DataFrame):
        """Analyze style issues by module"""
        # Extract module name
        df['module'] = df['file_path'].apply(
            lambda x: '/'.join(Path(x).parts[:2]) if '/' in x else 'root'
        )
        
        # Group by module
        module_stats = df.groupby('module').agg({
            'style_issues': 'sum',
            'file_path': 'count',
            'cyclomatic_complexity': 'mean'
        }).rename(columns={'file_path': 'file_count'})
        
        # Top 15 modules with most issues
        top_modules = module_stats.nlargest(15, 'style_issues')
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart for issue counts
        ax = axes[0]
        bars = ax.barh(range(len(top_modules)), top_modules['style_issues'])
        ax.set_yticks(range(len(top_modules)))
        ax.set_yticklabels(top_modules.index)
        ax.set_xlabel('Number of Style Issues')
        ax.set_title('Style Issues by Module (Top 15)')
        ax.invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 3, bar.get_y() + bar.get_height()/2,
                   f'{int(width)}', ha='left', va='center')
        
        # Scatter plot for issue density
        ax = axes[1]
        scatter = ax.scatter(
            module_stats['file_count'],
            module_stats['style_issues'] / module_stats['file_count'],
            c=module_stats['cyclomatic_complexity'],
            s=100,
            alpha=0.6,
            cmap='viridis'
        )
        ax.set_xlabel('Number of Files')
        ax.set_ylabel('Average Issues per File')
        ax.set_title('Issue Density vs File Count by Module')
        
        plt.colorbar(scatter, ax=ax, label='Average Cyclomatic Complexity')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'style_issues_by_module.png', dpi=150)
        plt.close()
    
    def plot_function_metrics(self, df: pd.DataFrame):
        """Function-level metrics analysis"""
        fig = plt.figure(figsize=(14, 10))
        
        # Correlation heatmap
        numeric_cols = [
            'cyclomatic_complexity', 'maintainability_index',
            'avg_lines_per_function', 'avg_args_per_function',
            'style_issues', 'function_count'
        ]
        
        # Filter existing columns
        existing_cols = [col for col in numeric_cols if col in df.columns]
        corr_matrix = df[existing_cols].corr()
        
        ax = fig.add_subplot(2, 2, 1)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                   center=0, ax=ax, square=True)
        ax.set_title('Metrics Correlation Heatmap')
        
        # Complexity vs Maintainability
        ax = fig.add_subplot(2, 2, 2)
        scatter = ax.scatter(
            df['cyclomatic_complexity'],
            df['maintainability_index'],
            c=df['style_issues'],
            s=50,
            alpha=0.6,
            cmap='RdYlBu_r'
        )
        ax.set_xlabel('Cyclomatic Complexity')
        ax.set_ylabel('Maintainability Index')
        ax.set_title('Complexity vs Maintainability Index')
        plt.colorbar(scatter, ax=ax, label='Number of Style Issues')
        
        # Function length distribution by file type
        ax = fig.add_subplot(2, 2, 3)
        
        # Group by file type (example: based on path)
        df['file_type'] = df['file_path'].apply(
            lambda x: 'test' if 'test' in x.lower() else 
                     'util' if 'util' in x.lower() else 'core'
        )
        
        data_to_plot = []
        labels = []
        for file_type in ['core', 'util', 'test']:
            if file_type in df['file_type'].values:
                data = df[df['file_type'] == file_type]['avg_lines_per_function']
                data_to_plot.append(data[data < 100].values)  # Remove outliers
                labels.append(file_type)
        
        ax.boxplot(data_to_plot, labels=labels)
        ax.set_ylabel('Average Function Length (Lines)')
        ax.set_title('Function Length Distribution by File Type')
        
        # Style issue types distribution
        ax = fig.add_subplot(2, 2, 4)
        
        # Mock issue type distribution
        issue_types = {
            'E2': 25,  # Whitespace related
            'E3': 18,  # Indentation
            'E5': 12,  # Line length
            'W1': 8,   # Warnings
            'C9': 15,  # Complexity
            'Other': 22
        }
        
        ax.pie(issue_types.values(), labels=issue_types.keys(),
              autopct='%1.1f%%', startangle=90)
        ax.set_title('Style Issue Type Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'function_metrics_analysis.png', dpi=150)
        plt.close()
    
    def generate_html_report(self, df: pd.DataFrame):
        """Generate detailed HTML report"""
        # Calculate overall statistics
        total_files = len(df)
        avg_complexity = df['cyclomatic_complexity'].mean()
        avg_maintainability = df['maintainability_index'].mean()
        total_style_issues = df['style_issues'].sum()
        
        # Find most complex files
        most_complex = df.nlargest(5, 'cyclomatic_complexity')[['file_path', 'cyclomatic_complexity']]
        most_issues = df.nlargest(5, 'style_issues')[['file_path', 'style_issues']]
        
        # Group by module
        df['module'] = df['file_path'].apply(
            lambda x: str(Path(x).parent)
        )
        module_stats = df.groupby('module').agg({
            'cyclomatic_complexity': 'mean',
            'style_issues': 'sum',
            'file_path': 'count'
        }).rename(columns={'file_path': 'file_count'})
        
        # HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>pandas Code Quality Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
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
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 pandas Code Quality Analysis Report</h1>
                    <p>Generated at: {{ timestamp }}</p>
                </div>
                
                <div class="metric-card">
                    <h2>📈 Overall Statistics</h2>
                    <div style="display: flex; justify-content: space-around; text-align: center;">
                        <div>
                            <div class="metric-value">{{ total_files }}</div>
                            <div>Files Analyzed</div>
                        </div>
                        <div>
                            <div class="metric-value {% if avg_complexity > 15 %}critical{% elif avg_complexity > 10 %}warning{% else %}good{% endif %}">
                                {{ avg_complexity | round(2) }}
                            </div>
                            <div>Avg Cyclomatic Complexity</div>
                        </div>
                        <div>
                            <div class="metric-value {% if avg_maintainability < 65 %}critical{% elif avg_maintainability < 85 %}warning{% else %}good{% endif %}">
                                {{ avg_maintainability | round(2) }}
                            </div>
                            <div>Avg Maintainability Index</div>
                        </div>
                        <div>
                            <div class="metric-value {% if total_style_issues > 100 %}critical{% elif total_style_issues > 50 %}warning{% else %}good{% endif %}">
                                {{ total_style_issues }}
                            </div>
                            <div>Total Style Issues</div>
                        </div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h2>📊 Visualizations</h2>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
                        <div>
                            <h3>Complexity Distribution</h3>
                            <img src="complexity_distribution.png" alt="Complexity Distribution">
                        </div>
                        <div>
                            <h3>Style Issues by Module</h3>
                            <img src="style_issues_by_module.png" alt="Style Issues by Module">
                        </div>
                        <div style="grid-column: span 2;">
                            <h3>Function Metrics Analysis</h3>
                            <img src="function_metrics_analysis.png" alt="Function Metrics Analysis">
                        </div>
                    </div>
                </div>
                
                <div class="highlight">
                    <h2>⚠️ Critical Areas Requiring Attention</h2>
                    <p>The following files require special attention for code quality and complexity:</p>
                </div>
                
                <h3>🔴 Most Complex Files (Top 5)</h3>
                <table class="table">
                    <tr>
                        <th>File Path</th>
                        <th>Cyclomatic Complexity</th>
                        <th>Status</th>
                    </tr>
                    {% for row in most_complex %}
                    <tr>
                        <td>{{ row.file_path }}</td>
                        <td>{{ row.cyclomatic_complexity | round(2) }}</td>
                        <td>
                            {% if row.cyclomatic_complexity > 20 %}
                                <span class="critical">CRITICAL</span>
                            {% elif row.cyclomatic_complexity > 15 %}
                                <span class="warning">WARNING</span>
                            {% else %}
                                <span class="good">OK</span>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </table>
                
                <h3>⚠️ Files with Most Style Issues (Top 5)</h3>
                <table class="table">
                    <tr>
                        <th>File Path</th>
                        <th>Style Issues Count</th>
                        <th>Status</th>
                    </tr>
                    {% for row in most_issues %}
                    <tr>
                        <td>{{ row.file_path }}</td>
                        <td>{{ row.style_issues }}</td>
                        <td>
                            {% if row.style_issues > 20 %}
                                <span class="critical">CRITICAL</span>
                            {% elif row.style_issues > 10 %}
                                <span class="warning">WARNING</span>
                            {% else %}
                                <span class="good">OK</span>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </table>
                
                <h3>📁 Module Statistics (Top 10)</h3>
                <table class="table">
                    <tr>
                        <th>Module</th>
                        <th>File Count</th>
                        <th>Avg Complexity</th>
                        <th>Total Issues</th>
                        <th>Issues per File</th>
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
                    <h2>💡 Improvement Recommendations</h2>
                    <ul>
                        <li><strong>Refactor</strong> functions with cyclomatic complexity > 20</li>
                        <li><strong>Add docstrings</strong> to functions missing documentation</li>
                        <li><strong>Follow PEP 8</strong> guidelines and fix style violations</li>
                        <li><strong>Split long functions</strong> into smaller, focused units</li>
                        <li><strong>Increase test coverage</strong> for critical modules</li>
                        <li><strong>Reduce function arguments</strong> to improve readability</li>
                        <li><strong>Add type hints</strong> to improve code clarity</li>
                        <li><strong>Review complex modules</strong> with high issue density</li>
                    </ul>
                    
                    <h3>🏆 Quality Benchmarks</h3>
                    <table class="table">
                        <tr>
                            <th>Metric</th>
                            <th>Excellent</th>
                            <th>Good</th>
                            <th>Needs Improvement</th>
                            <th>Critical</th>
                        </tr>
                        <tr>
                            <td>Cyclomatic Complexity</td>
                            <td class="good">&lt; 10</td>
                            <td class="good">10-15</td>
                            <td class="warning">15-20</td>
                            <td class="critical">&gt; 20</td>
                        </tr>
                        <tr>
                            <td>Maintainability Index</td>
                            <td class="good">&gt; 85</td>
                            <td class="good">65-85</td>
                            <td class="warning">50-65</td>
                            <td class="critical">&lt; 50</td>
                        </tr>
                        <tr>
                            <td>Style Issues per File</td>
                            <td class="good">&lt; 5</td>
                            <td class="good">5-10</td>
                            <td class="warning">10-20</td>
                            <td class="critical">&gt; 20</td>
                        </tr>
                        <tr>
                            <td>Function Length (Lines)</td>
                            <td class="good">&lt; 20</td>
                            <td class="good">20-50</td>
                            <td class="warning">50-100</td>
                            <td class="critical">&gt; 100</td>
                        </tr>
                    </table>
                </div>
                
                <div style="margin-top: 30px; padding: 15px; background: #f8f9fa; border-radius: 5px;">
                    <h3>📋 Analysis Summary</h3>
                    <p><strong>Overall Quality Rating:</strong> 
                        {% if avg_complexity < 10 and avg_maintainability > 85 and total_style_issues/total_files < 5 %}
                        <span class="good">EXCELLENT</span> 🏆
                        {% elif avg_complexity < 15 and avg_maintainability > 65 and total_style_issues/total_files < 10 %}
                        <span class="good">GOOD</span> 👍
                        {% elif avg_complexity < 20 and avg_maintainability > 50 %}
                        <span class="warning">NEEDS IMPROVEMENT</span> ⚠️
                        {% else %}
                        <span class="critical">REQUIRES URGENT ATTENTION</span> 🚨
                        {% endif %}
                    </p>
                    <p><strong>Key Findings:</strong></p>
                    <ul>
                        <li>Total files analyzed: {{ total_files }}</li>
                        <li>Average cyclomatic complexity: {{ avg_complexity | round(2) }}</li>
                        <li>Average maintainability index: {{ avg_maintainability | round(2) }}</li>
                        <li>Total style issues found: {{ total_style_issues }}</li>
                        <li>Average issues per file: {{ (total_style_issues/total_files) | round(2) if total_files > 0 else 0 }}</li>
                    </ul>
                </div>
                
                <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #666;">
                    <p>Report generated by pandas Code Quality Analyzer</p>
                    <p>Interactive charts available in: <code>reports/complexity_interactive.html</code></p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        # Render template
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
        
        # Save HTML file
        report_path = self.output_dir / 'analysis_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {report_path}")
        print(f"Interactive charts: {self.output_dir}/complexity_interactive.html")