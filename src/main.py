import os
import sys
from pathlib import Path
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from analyzer.ast_parser import ASTParser
from analyzer.metrics_calculator import MetricsCalculator
from analyzer.style_checker import StyleChecker
from analyzer.visualizer import Visualizer

# 设置中文字体（需要在Visualizer初始化之前设置）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class PandasCodeAnalyzer:
    def __init__(self, pandas_path=None):
        """初始化分析器"""
        if pandas_path is None:
            # 自动克隆pandas仓库（如不存在）
            self.pandas_path = self._clone_pandas_repo()
        else:
            self.pandas_path = Path(pandas_path)
        
        self.results = {}
        
    def _clone_pandas_repo(self):
        """使用已安装的pandas库，不进行网络克隆"""
        import pandas
        from pathlib import Path
        
        try:
            # 尝试导入已安装的pandas
            import pandas as pd
            pandas_path = Path(pd.__file__).parent
            print(f"✅ 成功找到已安装的pandas库: {pandas_path}")
            
            # 检查是否有足够的Python文件
            py_files = list(pandas_path.rglob("*.py"))
            if len(py_files) > 0:
                print(f"📁 找到 {len(py_files)} 个Python文件，使用此路径进行分析")
                return pandas_path
            else:
                print("⚠️  pandas安装目录中没有Python文件")
                # 返回pandas包根目录
                return pandas_path.parent
        except ImportError:
            print("❌ pandas未安装")
        
        # 如果pandas未安装，使用备用方案
        print("⚠️  使用当前目录作为分析路径")
        return Path(".")
    
    def analyze(self, max_files=None):
        """执行完整分析"""
        print(f"开始分析pandas代码库: {self.pandas_path}")
        
        # 1. 收集Python文件
        python_files = []
        for ext in ["*.py", "*.pyx"]:
            python_files.extend(list(self.pandas_path.rglob(ext)))
        
        # 排除测试文件和大文件以加快速度
        filtered_files = []
        for file_path in python_files:
            # 跳过测试文件（可选）
            if "test" in str(file_path).lower():
                continue
            
            # 跳过太大的文件
            try:
                if file_path.stat().st_size > 1024 * 1024:  # 1MB
                    continue
            except:
                pass
            
            filtered_files.append(file_path)
        
        python_files = filtered_files
        
        if max_files and max_files > 0:
            python_files = python_files[:max_files]
        
        print(f"找到 {len(python_files)} 个Python文件")
        
        if len(python_files) == 0:
            print("警告：未找到Python文件，检查路径是否正确")
            print(f"当前路径: {self.pandas_path.absolute()}")
            print(f"路径存在: {self.pandas_path.exists()}")
            # 尝试在当前目录下查找
            self.pandas_path = Path(".")
            python_files = list(Path(".").rglob("*.py"))[:max_files] if max_files else list(Path(".").rglob("*.py"))
            print(f"当前目录找到 {len(python_files)} 个文件")
        
        # 2. 初始化分析器
        ast_parser = ASTParser()
        metrics_calc = MetricsCalculator()
        style_checker = StyleChecker()
        
        all_metrics = []
        
        # 3. 分析每个文件
        for i, file_path in enumerate(python_files):
            if i % 10 == 0 and i > 0:
                print(f"进度: {i}/{len(python_files)}")
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                
                if not code.strip():
                    continue
                
                # 解析AST
                ast_tree = ast_parser.parse_code(code)
                
                # 计算指标
                metrics = metrics_calc.calculate_file_metrics(code, ast_tree)
                metrics['file_path'] = str(file_path.relative_to(self.pandas_path) 
                                          if self.pandas_path.exists() else str(file_path))
                metrics['file_size'] = len(code)
                
                # 检查代码规范
                try:
                    style_issues = style_checker.check_file(str(file_path))
                    metrics['style_issues'] = len(style_issues)
                except Exception as e:
                    print(f"检查文件规范时出错 {file_path}: {e}")
                    metrics['style_issues'] = 0
                
                # 提取函数信息
                try:
                    functions = ast_parser.extract_functions(ast_tree)
                    metrics['function_count'] = len(functions)
                    if functions:
                        metrics['avg_lines_per_function'] = sum(f['lines'] for f in functions) / len(functions)
                        metrics['avg_args_per_function'] = sum(f['arg_count'] for f in functions) / len(functions)
                    else:
                        metrics['avg_lines_per_function'] = 0
                        metrics['avg_args_per_function'] = 0
                except Exception as e:
                    print(f"提取函数信息时出错 {file_path}: {e}")
                    metrics['function_count'] = 0
                    metrics['avg_lines_per_function'] = 0
                    metrics['avg_args_per_function'] = 0
                
                all_metrics.append(metrics)
                
            except Exception as e:
                print(f"分析文件 {file_path} 时出错: {e}")
                continue
        
        # 4. 保存结果
        if all_metrics:
            self.results_df = pd.DataFrame(all_metrics)
            
            # 确保所有必需的列都存在
            required_columns = ['cyclomatic_complexity', 'maintainability_index', 
                               'style_issues', 'avg_lines_per_function']
            
            for col in required_columns:
                if col not in self.results_df.columns:
                    self.results_df[col] = 0
            
            self._save_results()
        else:
            print("警告：未成功分析任何文件")
            # 创建空DataFrame
            self.results_df = pd.DataFrame(columns=[
                'file_path', 'file_size', 'cyclomatic_complexity',
                'maintainability_index', 'style_issues', 
                'avg_lines_per_function', 'avg_args_per_function',
                'function_count'
            ])
        
        return self.results_df
    
    def _save_results(self):
        """保存分析结果"""
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        
        # 保存为CSV
        csv_path = output_dir / "pandas_analysis.csv"
        self.results_df.to_csv(csv_path, index=False)
        print(f"详细结果已保存到: {csv_path}")
        
        # 计算摘要统计
        summary = {
            "total_files": len(self.results_df),
            "avg_complexity": float(self.results_df['cyclomatic_complexity'].mean()) if len(self.results_df) > 0 else 0,
            "avg_maintainability": float(self.results_df['maintainability_index'].mean()) if len(self.results_df) > 0 else 0,
            "total_style_issues": int(self.results_df['style_issues'].sum()) if len(self.results_df) > 0 else 0,
            "avg_lines_per_function": float(self.results_df['avg_lines_per_function'].mean()) if len(self.results_df) > 0 else 0
        }
        
        # 保存为JSON
        import json
        json_path = output_dir / "summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"摘要统计已保存到: {json_path}")
        
        print(f"\n分析完成！共分析 {summary['total_files']} 个文件")
        print(f"平均圈复杂度: {summary['avg_complexity']:.2f}")
        print(f"平均维护性指数: {summary['avg_maintainability']:.2f}")
        print(f"规范问题总数: {summary['total_style_issues']}")
    
    def generate_report(self):
        """生成可视化报告"""
        if self.results_df.empty:
            print("没有数据可生成报告")
            return
        
        visualizer = Visualizer()
        
        # 生成各种图表
        print("生成可视化图表...")
        visualizer.plot_complexity_distribution(self.results_df)
        visualizer.plot_style_issues_by_module(self.results_df)
        visualizer.plot_function_metrics(self.results_df)
        visualizer.generate_html_report(self.results_df)
        
        print("报告生成完成！查看 reports/ 目录")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='pandas代码质量分析工具')
    parser.add_argument('--pandas-path', type=str, help='pandas代码库路径')
    parser.add_argument('--max-files', type=int, default=50, help='最大分析文件数')
    parser.add_argument('--output', type=str, default='reports/', help='输出目录')
    parser.add_argument('--skip-clone', action='store_true', help='跳过克隆仓库')
    
    args = parser.parse_args()
    
    if args.skip_clone and not args.pandas_path:
        print("警告：跳过克隆但未指定路径，将使用当前目录")
        analyzer = PandasCodeAnalyzer(pandas_path=".")
    else:
        analyzer = PandasCodeAnalyzer(pandas_path=args.pandas_path)
    
    # 分析代码
    df = analyzer.analyze(max_files=args.max_files)
    
    # 生成报告
    if not df.empty:
        analyzer.generate_report()
    else:
        print("无法生成报告：没有分析结果")

if __name__ == "__main__":
    main()