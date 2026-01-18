import os
import sys
import json
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from tqdm import tqdm  # 进度条库
import argparse

# 导入自定义分析器
from analyzer.ast_parser import ASTParser
from analyzer.metrics_calculator import MetricsCalculator
from analyzer.style_checker import StyleChecker
from analyzer.visualizer import Visualizer
from analyzer.complexity_analyzer import AdvancedComplexityAnalyzer
from analyzer.dependency_analyzer import DependencyAnalyzer
from analyzer.security_analyzer import SecurityAnalyzer
from analyzer.test_coverage_analyzer import TestCoverageAnalyzer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Config:
    """配置文件类"""
    
    DEFAULT_CONFIG = {
        'analysis': {
            'max_files': 200,
            'skip_tests': True,
            'skip_large_files': True,
            'max_file_size_mb': 1,
            'exclude_patterns': ['__pycache__', '.git', '.svn', '.hg'],
            'include_extensions': ['.py', '.pyx']
        },
        'metrics': {
            'enable_complexity': True,
            'enable_security': True,
            'enable_dependency': True,
            'enable_coverage': False,
            'min_lines_for_analysis': 5
        },
        'output': {
            'reports_dir': 'reports',
            'save_raw_data': True,
            'generate_html': True,
            'generate_pdf': False,
            'verbose': False
        }
    }
    
    @classmethod
    def load(cls, config_path: Optional[str] = None):
        """加载配置文件"""
        config = cls.DEFAULT_CONFIG.copy()
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # 深度合并配置
                    cls._deep_update(config, user_config)
                    print(f"✅ 已加载配置文件: {config_path}")
            except Exception as e:
                print(f"⚠️  配置文件加载失败: {e}，使用默认配置")
        
        return config
    
    @staticmethod
    def _deep_update(original: Dict, update: Dict):
        """深度更新字典"""
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                Config._deep_update(original[key], value)
            else:
                original[key] = value

class FileCollector:
    """文件收集器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.analysis_config = config['analysis']
        self.metrics_config = config['metrics']
    
    def collect_files(self, base_path: Path) -> List[Path]:
        """收集要分析的文件"""
        all_files = []
        include_exts = self.analysis_config['include_extensions']
        exclude_patterns = self.analysis_config['exclude_patterns']
        
        print(f"📁 扫描目录: {base_path}")
        
        for ext in include_exts:
            for file_path in base_path.rglob(f"*{ext}"):
                # 检查排除模式
                if any(pattern in str(file_path) for pattern in exclude_patterns):
                    continue
                
                # 检查是否跳过测试文件
                if self.analysis_config['skip_tests'] and self._is_test_file(file_path):
                    continue
                
                # 检查文件大小
                if self.analysis_config['skip_large_files']:
                    try:
                        file_size_mb = file_path.stat().st_size / (1024 * 1024)
                        if file_size_mb > self.analysis_config['max_file_size_mb']:
                            continue
                    except:
                        pass
                
                all_files.append(file_path)
        
        # 限制文件数量
        max_files = self.analysis_config['max_files']
        if max_files and len(all_files) > max_files:
            print(f"📊 文件数量限制: {max_files} (共找到 {len(all_files)} 个文件)")
            all_files = all_files[:max_files]
        
        print(f"✅ 找到 {len(all_files)} 个文件用于分析")
        return all_files
    
    def _is_test_file(self, file_path: Path) -> bool:
        """判断是否为测试文件"""
        path_str = str(file_path).lower()
        name = file_path.name.lower()
        
        # 检查文件名模式
        test_patterns = [
            'test_', '_test.', 'test.', '_test_',
            'tests/', 'test/', '__test__'
        ]
        
        return any(pattern in path_str for pattern in test_patterns)

class PandasCodeAnalyzer:
    def __init__(self, config: Dict):
        """初始化分析器"""
        self.config = config
        self.output_config = config['output']
        self.metrics_config = config['metrics']
        
        # 创建输出目录
        self.reports_dir = Path(self.output_config['reports_dir'])
        self.reports_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化分析器
        self.ast_parser = ASTParser()
        self.metrics_calc = MetricsCalculator()
        self.style_checker = StyleChecker()
        self.visualizer = Visualizer()
        
        # 高级分析器
        self.complexity_analyzer = AdvancedComplexityAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
        self.security_analyzer = SecurityAnalyzer()
        self.test_analyzer = TestCoverageAnalyzer()
        
        # 结果存储
        self.results_df = pd.DataFrame()
        self.complexity_results = []
        self.security_results = []
        self.dependency_results = []
        self.test_results = []
        
        print("🔧 分析器初始化完成")
    
    def analyze_directory(self, source_path: Path) -> Dict[str, Any]:
        """分析整个目录"""
        print(f"🚀 开始分析: {source_path}")
        start_time = time.time()
        
        # 收集文件
        collector = FileCollector(self.config)
        files_to_analyze = collector.collect_files(source_path)
        
        if not files_to_analyze:
            print("❌ 未找到可分析的文件")
            return {}
        
        # 分析文件
        analysis_results = self._analyze_files(files_to_analyze)
        
        # 执行高级分析
        self._perform_advanced_analysis(analysis_results, files_to_analyze)
        
        # 生成报告
        self._generate_comprehensive_report(analysis_results)
        
        # 计算统计信息
        elapsed_time = time.time() - start_time
        stats = self._calculate_statistics(analysis_results, elapsed_time)
        
        return stats
    
    def _analyze_files(self, files: List[Path]) -> List[Dict]:
        """分析文件列表"""
        results = []
        errors = []
        
        print("📊 开始文件分析...")
        
        for file_path in tqdm(files, desc="分析进度", unit="文件"):
            try:
                result = self._analyze_single_file(file_path)
                if result:  # 只添加有效结果
                    results.append(result)
            except Exception as e:
                error_info = {
                    'file': str(file_path),
                    'error': str(e)
                }
                errors.append(error_info)
                
                if self.output_config['verbose']:
                    print(f"❌ 分析失败: {file_path} - {e}")
        
        # 保存错误信息
        if errors:
            self._save_errors(errors)
        
        print(f"✅ 成功分析 {len(results)}/{len(files)} 个文件")
        return results
    
    def _analyze_single_file(self, file_path: Path) -> Optional[Dict]:
        """分析单个文件"""
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            
            # 跳过过小的文件
            if len(code.strip().split('\n')) < self.metrics_config['min_lines_for_analysis']:
                return None
            
            # 解析AST
            ast_tree = self.ast_parser.parse_code(code)
            
            # 基础指标
            metrics = self.metrics_calc.calculate_file_metrics(code, ast_tree)
            
            # 文件信息
            metrics['file_path'] = str(file_path)
            metrics['file_name'] = file_path.name
            metrics['file_size_kb'] = len(code) / 1024
            metrics['lines_of_code'] = len(code.split('\n'))
            
            # 代码规范检查
            try:
                style_issues = self.style_checker.check_file(str(file_path))
                metrics['style_issues'] = len(style_issues)
                if style_issues:
                    metrics['style_issue_details'] = style_issues[:10]  # 只保留前10个
            except:
                metrics['style_issues'] = 0
            
            # 函数信息
            functions = self.ast_parser.extract_functions(ast_tree)
            metrics['function_count'] = len(functions)
            if functions:
                metrics['avg_function_lines'] = sum(f['lines'] for f in functions) / len(functions)
                metrics['max_function_lines'] = max(f['lines'] for f in functions)
                metrics['avg_function_args'] = sum(f['arg_count'] for f in functions) / len(functions)
                metrics['functions_with_docstring'] = sum(1 for f in functions if f['has_docstring'])
            
            # 导入信息
            imports = self.ast_parser.extract_imports(ast_tree)
            metrics['import_count'] = len(imports)
            
            # 高级复杂度分析
            if self.metrics_config['enable_complexity']:
                try:
                    complexity = self.complexity_analyzer.analyze_complexity(code, ast_tree)
                    metrics.update(complexity)
                except Exception as e:
                    if self.output_config['verbose']:
                        print(f"复杂度分析失败 {file_path}: {e}")
            
            # 安全分析
            if self.metrics_config['enable_security']:
                try:
                    security = self.security_analyzer.analyze_file_security(str(file_path), code)
                    metrics.update({
                        'security_score': security['security_score'],
                        'security_issues': security['total_issues'],
                        'high_risk_issues': security['high_risk_issues']
                    })
                except Exception as e:
                    if self.output_config['verbose']:
                        print(f"安全分析失败 {file_path}: {e}")
            
            return metrics
            
        except Exception as e:
            if self.output_config['verbose']:
                print(f"分析文件失败 {file_path}: {e}")
            return None
    
    def _perform_advanced_analysis(self, analysis_results: List[Dict], files: List[Path]):
        """执行高级分析"""
        print("🔍 执行高级分析...")
        
        # 依赖分析
        if self.metrics_config['enable_dependency']:
            print("  📦 分析依赖关系...")
            self.dependency_results = []
            for result in analysis_results:
                try:
                    with open(result['file_path'], 'r', encoding='utf-8') as f:
                        code = f.read()
                    deps = self.dependency_analyzer.analyze_file(result['file_path'], code)
                    self.dependency_results.append(deps)
                except Exception as e:
                    if self.output_config['verbose']:
                        print(f"依赖分析失败 {result.get('file_path', 'unknown')}: {e}")
                    continue
            
            if self.dependency_results:
                try:
                    # 修复：不依赖可视化方法，只生成报告
                    dep_df = pd.DataFrame(self.dependency_results)
                    dep_output_path = self.reports_dir / 'dependency_analysis.csv'
                    dep_df.to_csv(dep_output_path, index=False)
                    print(f"  ✅ 依赖分析结果保存至: {dep_output_path}")
                    
                    # 尝试构建依赖图但不强制可视化
                    try:
                        # 构建依赖图
                        import networkx as nx
                        from collections import defaultdict
                        
                        G = nx.DiGraph()
                        for result in self.dependency_results:
                            file_name = Path(result['file']).stem
                            G.add_node(file_name, **result)
                            
                            for imp in result.get('import_details', []):
                                if imp['type'] == 'from_import' and imp['module']:
                                    source_module = Path(imp['module']).stem
                                    if source_module in G.nodes:
                                        G.add_edge(file_name, source_module)
                        
                        # 保存依赖图数据
                        graph_data_path = self.reports_dir / 'dependency_graph.gpickle'
                        nx.write_gpickle(G, graph_data_path)
                        print(f"  ✅ 依赖图数据保存至: {graph_data_path}")
                        
                        # 尝试生成简化可视化（不强制依赖matplotlib）
                        self._generate_simple_dependency_report(G)
                        
                    except Exception as graph_error:
                        print(f"  ⚠️  依赖图生成失败: {graph_error}")
                        
                except Exception as e:
                    print(f"  ❌ 依赖报告生成失败: {e}")
        
        # 测试覆盖率分析
        if self.metrics_config['enable_coverage']:
            print("  ✅ 分析测试覆盖率...")
            try:
                test_files, non_test_files = self.test_analyzer.identify_test_files(files)
                test_stats = self.test_analyzer.analyze_test_structure(test_files)
                
                # 运行覆盖率工具
                try:
                    coverage_data = self.test_analyzer.run_coverage_tool()
                    test_df = self.test_analyzer.generate_test_report(test_stats, coverage_data)
                    self.test_results = test_stats
                except Exception as cov_error:
                    print(f"  ⚠️  覆盖率工具运行失败: {cov_error}")
                    test_df = self.test_analyzer.generate_test_report(test_stats)
                    self.test_results = test_stats
            except Exception as test_error:
                print(f"  ❌ 测试分析失败: {test_error}")
    
    def _generate_simple_dependency_report(self, graph):
        """生成简化的依赖报告（不依赖matplotlib）"""
        try:
            import networkx as nx
            
            # 计算基本指标
            metrics = {
                'total_nodes': len(graph.nodes()),
                'total_edges': len(graph.edges()),
                'density': nx.density(graph) if len(graph.nodes()) > 1 else 0,
            }
            
            # 计算入度和出度
            in_degrees = dict(graph.in_degree())
            out_degrees = dict(graph.out_degree())
            
            if in_degrees:
                metrics['avg_in_degree'] = sum(in_degrees.values()) / len(in_degrees)
                metrics['max_in_degree'] = max(in_degrees.values())
                # 最高入度的模块
                top_in_degree = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
                metrics['top_in_dependent_modules'] = top_in_degree
            
            if out_degrees:
                metrics['avg_out_degree'] = sum(out_degrees.values()) / len(out_degrees)
                metrics['max_out_degree'] = max(out_degrees.values())
                # 最高出度的模块
                top_out_degree = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
                metrics['top_dependent_modules'] = top_out_degree
            
            # 检测循环依赖
            try:
                cycles = list(nx.simple_cycles(graph))
                metrics['circular_dependencies'] = len(cycles)
                if cycles:
                    metrics['cycle_details'] = [', '.join(cycle) for cycle in cycles[:5]]  # 只显示前5个
            except:
                metrics['circular_dependencies'] = 0
            
            # 保存指标
            metrics_path = self.reports_dir / 'dependency_metrics.json'
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            
            print(f"  ✅ 依赖指标保存至: {metrics_path}")
            
        except Exception as e:
            print(f"  ⚠️  生成依赖报告失败: {e}")
    
    def _generate_comprehensive_report(self, analysis_results: List[Dict]):
        """生成综合报告"""
        print("📄 生成报告...")
        
        # 保存原始数据
        if self.output_config['save_raw_data'] and analysis_results:
            self.results_df = pd.DataFrame(analysis_results)
            self.results_df.to_csv(self.reports_dir / 'raw_analysis_data.csv', index=False)
            print(f"  📊 原始数据保存至: {self.reports_dir}/raw_analysis_data.csv")
        else:
            print("  ⚠️  无分析结果，跳过原始数据保存")
            return
        
        # 生成可视化报告
        if self.output_config['generate_html'] and not self.results_df.empty:
            try:
                self.visualizer.plot_complexity_distribution(self.results_df)
                self.visualizer.plot_style_issues_by_module(self.results_df)
                self.visualizer.plot_function_metrics(self.results_df)
                self.visualizer.generate_html_report(self.results_df)
                print("  ✅ 可视化报告生成完成")
            except Exception as e:
                print(f"  ❌ 可视化报告生成失败: {e}")
        
        # 生成文本摘要
        self._generate_text_summary(analysis_results)
        
        # 生成配置文件
        self._generate_config_summary()
    
    def _generate_text_summary(self, analysis_results: List[Dict]):
        """生成文本摘要"""
        if not analysis_results:
            return
        
        summary = {
            "analysis_summary": self._calculate_statistics(analysis_results, 0),
            "top_complex_files": self._get_top_complex_files(analysis_results),
            "security_alerts": self._get_security_alerts(analysis_results),
            "recommendations": self._generate_recommendations(analysis_results)
        }
        
        # 保存JSON格式
        json_path = self.reports_dir / 'analysis_summary.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 保存Markdown格式
        md_path = self.reports_dir / 'README.md'
        self._generate_markdown_report(summary, md_path)
    
    def _generate_markdown_report(self, summary: Dict, output_path: Path):
        """生成Markdown报告"""
        content = f"""# 代码质量分析报告

## 📊 分析概览
- **分析文件数**: {summary['analysis_summary']['total_files']}
- **平均复杂度**: {summary['analysis_summary']['avg_complexity']:.2f}
- **平均维护指数**: {summary['analysis_summary']['avg_maintainability']:.2f}
- **安全分数**: {summary['analysis_summary']['avg_security_score']:.2f}

## ⚠️ 复杂度最高的文件
"""
        
        for i, file_info in enumerate(summary['top_complex_files'][:5], 1):
            content += f"{i}. **{file_info['file_name']}** - 复杂度: {file_info['cyclomatic_complexity']:.2f}\n"
        
        content += "\n## 🔒 安全警报\n"
        if summary['security_alerts']:
            for alert in summary['security_alerts'][:3]:
                content += f"- **{alert['file_name']}**: {alert['high_risk_issues']} 个高风险问题\n"
        else:
            content += "✅ 未发现高风险安全问题\n"
        
        content += "\n## 💡 改进建议\n"
        for rec in summary['recommendations']:
            content += f"- {rec}\n"
        
        content += f"""
## 📁 文件详情
完整分析结果见: `{self.reports_dir}/raw_analysis_data.csv`

## 📈 可视化报告
HTML报告: `{self.reports_dir}/analysis_report.html`
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_config_summary(self):
        """生成配置文件摘要"""
        config_summary = {
            'analysis_config': self.config['analysis'],
            'metrics_config': self.config['metrics'],
            'output_config': self.config['output'],
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        config_path = self.reports_dir / 'config_summary.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_summary, f, indent=2)
    
    def _save_errors(self, errors: List[Dict]):
        """保存错误信息"""
        if errors:
            errors_path = self.reports_dir / 'analysis_errors.json'
            with open(errors_path, 'w', encoding='utf-8') as f:
                json.dump(errors, f, indent=2)
            print(f"⚠️  分析错误保存至: {errors_path}")
    
    def _calculate_statistics(self, results: List[Dict], elapsed_time: float) -> Dict:
        """计算统计信息"""
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        
        stats = {
            'total_files': len(results),
            'total_lines_of_code': int(df['lines_of_code'].sum()) if 'lines_of_code' in df.columns else 0,
            'analysis_time_seconds': round(elapsed_time, 2),
            'files_per_second': round(len(results) / elapsed_time, 2) if elapsed_time > 0 else 0
        }
        
        # 可选指标
        if 'cyclomatic_complexity' in df.columns:
            stats['avg_complexity'] = float(df['cyclomatic_complexity'].mean())
        
        if 'maintainability_index' in df.columns:
            stats['avg_maintainability'] = float(df['maintainability_index'].mean())
        
        if 'security_score' in df.columns:
            stats['avg_security_score'] = float(df['security_score'].mean())
        
        if 'style_issues' in df.columns:
            stats['total_style_issues'] = int(df['style_issues'].sum())
        
        if 'function_count' in df.columns:
            stats['avg_function_count'] = float(df['function_count'].mean())
        
        return stats
    
    def _get_top_complex_files(self, results: List[Dict]) -> List[Dict]:
        """获取复杂度最高的文件"""
        if not results:
            return []
        
        df = pd.DataFrame(results)
        if 'cyclomatic_complexity' not in df.columns:
            return []
        
        top_files = df.nlargest(10, 'cyclomatic_complexity')
        return top_files[['file_path', 'file_name', 'cyclomatic_complexity']].to_dict('records')
    
    def _get_security_alerts(self, results: List[Dict]) -> List[Dict]:
        """获取安全警报"""
        alerts = []
        
        for result in results:
            high_risk = result.get('high_risk_issues', 0)
            if high_risk > 0:
                alerts.append({
                    'file_path': result['file_path'],
                    'file_name': result.get('file_name', 'Unknown'),
                    'high_risk_issues': high_risk,
                    'security_score': result.get('security_score', 100)
                })
        
        return sorted(alerts, key=lambda x: x['high_risk_issues'], reverse=True)
    
    def _generate_recommendations(self, results: List[Dict]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if not results:
            return ["无法生成建议：没有分析结果"]
        
        df = pd.DataFrame(results)
        
        # 复杂度建议
        if 'cyclomatic_complexity' in df.columns:
            complex_count = len(df[df['cyclomatic_complexity'] > 20])
            if complex_count > 0:
                recommendations.append(f"重构 {complex_count} 个高复杂度函数（复杂度 > 20）")
        
        # 安全建议
        if 'high_risk_issues' in df.columns:
            high_risk_count = df['high_risk_issues'].sum()
            if high_risk_count > 0:
                recommendations.append(f"立即修复 {high_risk_count} 个高风险安全问题")
        
        # 文档建议
        if 'functions_with_docstring' in df.columns and 'function_count' in df.columns:
            total_functions = df['function_count'].sum()
            documented_functions = df['functions_with_docstring'].sum()
            if total_functions > 0 and documented_functions / total_functions < 0.5:
                recommendations.append("为超过50%的函数添加文档字符串")
        
        # 代码规范建议
        if 'style_issues' in df.columns:
            total_issues = df['style_issues'].sum()
            if total_issues > 100:
                recommendations.append(f"修复 {total_issues} 个代码规范问题")
        
        # 通用建议
        recommendations.extend([
            "使用类型注解提高代码可读性",
            "保持函数单一职责原则",
            "添加单元测试覆盖关键功能",
            "定期进行代码审查",
            "使用自动化代码质量检查工具"
        ])
        
        return recommendations

def find_pandas_source() -> Path:
    """查找pandas源代码路径"""
    try:
        # 尝试导入已安装的pandas
        import pandas
        pandas_path = Path(pandas.__file__).parent.parent  # 获取包根目录
        print(f"✅ 发现已安装的pandas: {pandas_path}")
        
        # 检查是否有Python文件
        py_files = list(pandas_path.rglob("*.py"))
        if len(py_files) > 10:  # 至少要有一些Python文件
            return pandas_path
        
    except ImportError:
        print("❌ 未安装pandas，请先安装: pip install pandas")
    
    # 返回当前目录
    current_dir = Path(".").absolute()
    print(f"⚠️  使用当前目录: {current_dir}")
    return current_dir

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='高级Python代码质量分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s  # 分析当前目录
  %(prog)s --path /path/to/code --max-files 100
  %(prog)s --config config.json --verbose
        """
    )
    
    parser.add_argument('--path', type=str, help='源代码路径')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--max-files', type=int, help='最大分析文件数')
    parser.add_argument('--output', type=str, default='reports', help='输出目录')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    parser.add_argument('--skip-tests', action='store_true', help='跳过测试文件')
    parser.add_argument('--skip-security', action='store_true', help='跳过安全分析')
    parser.add_argument('--skip-dependency', action='store_true', help='跳过依赖分析')
    
    args = parser.parse_args()
    
    # 加载配置
    config = Config.load(args.config)
    
    # 更新配置参数
    if args.max_files:
        config['analysis']['max_files'] = args.max_files
    
    if args.output:
        config['output']['reports_dir'] = args.output
    
    if args.verbose:
        config['output']['verbose'] = True
    
    if args.skip_tests:
        config['analysis']['skip_tests'] = True
    
    if args.skip_security:
        config['metrics']['enable_security'] = False
    
    if args.skip_dependency:
        config['metrics']['enable_dependency'] = False
    
    # 确定分析路径
    if args.path:
        source_path = Path(args.path)
        if not source_path.exists():
            print(f"❌ 路径不存在: {source_path}")
            sys.exit(1)
    else:
        source_path = find_pandas_source()
    
    print("=" * 60)
    print("🚀 Python代码质量分析工具")
    print("=" * 60)
    print(f"📁 分析路径: {source_path}")
    print(f"📊 最大文件数: {config['analysis']['max_files']}")
    print(f"📁 输出目录: {config['output']['reports_dir']}")
    print("=" * 60)
    
    # 创建分析器并执行分析
    analyzer = PandasCodeAnalyzer(config)
    stats = analyzer.analyze_directory(source_path)
    
    # 输出摘要
    if stats:
        print("\n" + "=" * 60)
        print("📋 分析摘要")
        print("=" * 60)
        print(f"📁 分析文件数: {stats['total_files']}")
        print(f"📝 总代码行数: {stats['total_lines_of_code']:,}")
        
        if 'avg_complexity' in stats:
            print(f"⚡ 平均复杂度: {stats['avg_complexity']:.2f}")
        
        if 'avg_maintainability' in stats:
            print(f"🛠️  平均维护指数: {stats['avg_maintainability']:.2f}")
        
        if 'avg_security_score' in stats:
            print(f"🔒 平均安全分数: {stats['avg_security_score']:.2f}")
        
        if 'total_style_issues' in stats:
            print(f"📝 规范问题总数: {stats['total_style_issues']}")
        
        print(f"⏱️  分析用时: {stats['analysis_time_seconds']:.2f}秒")
        print(f"🚀 处理速度: {stats['files_per_second']:.2f} 文件/秒")
        print("=" * 60)
        print(f"📄 报告已生成至: {config['output']['reports_dir']}/")
        print("=" * 60)
    else:
        print("❌ 分析失败，未生成结果")

if __name__ == "__main__":
    main()