# [file name]: test_coverage_analyzer.py
# [file content begin]
import ast
import re
from typing import Dict, List, Tuple, Set
import pandas as pd
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import json

class TestCoverageAnalyzer:
    """分析测试覆盖率和测试质量"""
    
    def __init__(self):
        self.test_patterns = [
            r'^test_.*\.py$',      # test_开头的文件
            r'^.*_test\.py$',      # _test结尾的文件
            r'^.*test.*\.py$',     # 包含test的文件
        ]
        self.test_functions = set()
        
    def identify_test_files(self, file_paths: List[Path]) -> Tuple[List[Path], List[Path]]:
        """识别测试文件和非测试文件"""
        test_files = []
        non_test_files = []
        
        for file_path in file_paths:
            file_name = file_path.name
            is_test_file = False
            
            # 检查文件名模式
            for pattern in self.test_patterns:
                if re.match(pattern, file_name, re.IGNORECASE):
                    is_test_file = True
                    break
            
            # 检查路径中是否包含test
            if not is_test_file and 'test' in str(file_path).lower():
                is_test_file = True
            
            if is_test_file:
                test_files.append(file_path)
            else:
                non_test_files.append(file_path)
        
        return test_files, non_test_files
    
    def analyze_test_structure(self, test_files: List[Path]) -> Dict:
        """分析测试文件结构"""
        test_stats = {
            'total_test_files': len(test_files),
            'test_classes': 0,
            'test_methods': 0,
            'assertions': 0,
            'test_fixtures': 0,
            'file_details': []
        }
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                file_stats = self._analyze_test_file(code, test_file)
                file_stats['file_name'] = test_file.name
                file_stats['file_path'] = str(test_file)
                
                test_stats['file_details'].append(file_stats)
                test_stats['test_classes'] += file_stats['test_classes']
                test_stats['test_methods'] += file_stats['test_methods']
                test_stats['assertions'] += file_stats['assertions']
                test_stats['test_fixtures'] += file_stats['test_fixtures']
                
            except Exception as e:
                print(f"分析测试文件 {test_file} 时出错: {e}")
                continue
        
        return test_stats
    
    def _analyze_test_file(self, code: str, file_path: Path) -> Dict:
        """分析单个测试文件"""
        stats = {
            'test_classes': 0,
            'test_methods': 0,
            'assertions': 0,
            'test_fixtures': 0,
            'imports': [],
            'test_cases': []
        }
        
        try:
            tree = ast.parse(code)
            
            # 查找测试类（通常继承自TestCase或使用@pytest标记）
            for node in ast.walk(tree):
                # 统计assert语句
                if isinstance(node, ast.Assert):
                    stats['assertions'] += 1
                
                # 查找类定义
                if isinstance(node, ast.ClassDef):
                    # 检查是否是测试类
                    if self._is_test_class(node):
                        stats['test_classes'] += 1
                        
                        # 统计测试方法
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                if self._is_test_method(item):
                                    stats['test_methods'] += 1
                                    stats['test_cases'].append({
                                        'name': item.name,
                                        'class': node.name,
                                        'line': item.lineno if hasattr(item, 'lineno') else 0
                                    })
                
                # 查找测试函数（不在类中的）
                elif isinstance(node, ast.FunctionDef):
                    if self._is_test_function(node):
                        stats['test_methods'] += 1
                        stats['test_cases'].append({
                            'name': node.name,
                            'class': None,
                            'line': node.lineno if hasattr(node, 'lineno') else 0
                        })
                
                # 查找fixture装饰器
                if isinstance(node, ast.FunctionDef):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name):
                            if decorator.id == 'fixture' or decorator.id == 'pytest.fixture':
                                stats['test_fixtures'] += 1
                
                # 提取导入
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    stats['imports'].append(ast.unparse(node))
        
        except SyntaxError as e:
            print(f"解析测试文件 {file_path} 时出错: {e}")
        
        return stats
    
    def _is_test_class(self, class_node: ast.ClassDef) -> bool:
        """判断是否是测试类"""
        # 检查类名
        class_name = class_node.name.lower()
        if 'test' in class_name:
            return True
        
        # 检查基类
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                base_name = base.id.lower()
                if 'test' in base_name or 'case' in base_name:
                    return True
        
        return False
    
    def _is_test_method(self, func_node: ast.FunctionDef) -> bool:
        """判断是否是测试方法"""
        func_name = func_node.name.lower()
        return func_name.startswith('test_')
    
    def _is_test_function(self, func_node: ast.FunctionDef) -> bool:
        """判断是否是测试函数"""
        func_name = func_node.name.lower()
        return func_name.startswith('test_')
    
    def calculate_test_metrics(self, test_stats: Dict, total_files: int) -> Dict:
        """计算测试相关指标"""
        if total_files == 0:
            return {}
        
        total_test_files = test_stats['total_test_files']
        total_test_methods = test_stats['test_methods']
        
        metrics = {
            'test_file_ratio': total_test_files / total_files if total_files > 0 else 0,
            'tests_per_file': total_test_methods / total_test_files if total_test_files > 0 else 0,
            'assertions_per_test': test_stats['assertions'] / total_test_methods if total_test_methods > 0 else 0,
            'test_fixtures_per_file': test_stats['test_fixtures'] / total_test_files if total_test_files > 0 else 0,
            'test_coverage_score': self._calculate_test_coverage_score(test_stats),
            'test_quality_score': self._calculate_test_quality_score(test_stats)
        }
        
        return metrics
    
    def _calculate_test_coverage_score(self, test_stats: Dict) -> float:
        """计算测试覆盖率得分（简化版）"""
        score = 0.0
        
        # 基于测试文件比例
        if test_stats['total_test_files'] > 0:
            score += 30 * min(1.0, test_stats['total_test_files'] / 100)
        
        # 基于测试方法数量
        if test_stats['test_methods'] > 0:
            score += 40 * min(1.0, test_stats['test_methods'] / 500)
        
        # 基于断言数量
        if test_stats['assertions'] > 0:
            assertions_per_test = test_stats['assertions'] / test_stats['test_methods']
            if assertions_per_test >= 2:
                score += 30
            elif assertions_per_test >= 1:
                score += 20
            else:
                score += 10
        
        return min(100.0, score)
    
    def _calculate_test_quality_score(self, test_stats: Dict) -> float:
        """计算测试质量得分"""
        score = 0.0
        
        if test_stats['total_test_files'] == 0:
            return 0.0
        
        # 检查是否有fixture
        if test_stats['test_fixtures'] > 0:
            score += 20
        
        # 检查测试结构
        if test_stats['test_classes'] > 0:
            score += 20
        
        # 检查断言密度
        if test_stats['assertions'] > 0:
            density = test_stats['assertions'] / test_stats['test_methods']
            if density >= 2:
                score += 40
            elif density >= 1:
                score += 30
            else:
                score += 10
        
        # 基于测试方法的命名规范
        test_cases = test_stats.get('test_cases', [])
        if test_cases:
            properly_named = sum(1 for tc in test_cases 
                               if tc['name'].startswith('test_'))
            naming_score = (properly_named / len(test_cases)) * 20
            score += naming_score
        
        return min(100.0, score)
    
    def run_coverage_tool(self, test_dir: str = "tests") -> Dict:
        """运行覆盖率工具（如果可用）"""
        coverage_data = {
            'coverage_percentage': 0.0,
            'covered_lines': 0,
            'total_lines': 0,
            'branch_coverage': 0.0,
            'missing_lines': []
        }
        
        try:
            # 尝试运行pytest-cov
            result = subprocess.run(
                ['pytest', '--cov=pandas', '--cov-report=xml', test_dir],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # 解析coverage.xml
            if Path('coverage.xml').exists():
                tree = ET.parse('coverage.xml')
                root = tree.getroot()
                
                for elem in root.findall('.//coverage'):
                    line_rate = float(elem.get('line-rate', 0)) * 100
                    branch_rate = float(elem.get('branch-rate', 0)) * 100
                    
                    coverage_data['coverage_percentage'] = line_rate
                    coverage_data['branch_coverage'] = branch_rate
                
                # 解析详细的覆盖率数据
                for package in root.findall('.//package'):
                    for cls in package.findall('.//class'):
                        class_name = cls.get('name', '')
                        for line in cls.findall('.//line'):
                            line_number = int(line.get('number', 0))
                            hits = int(line.get('hits', 0))
                            
                            if hits == 0:
                                coverage_data['missing_lines'].append({
                                    'class': class_name,
                                    'line': line_number
                                })
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"覆盖率工具运行失败: {e}")
            print("请确保已安装pytest-cov: pip install pytest-cov")
        
        except Exception as e:
            print(f"解析覆盖率数据时出错: {e}")
        
        return coverage_data
    
    def generate_test_report(self, test_stats: Dict, coverage_data: Dict = None) -> pd.DataFrame:
        """生成测试分析报告"""
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        
        # 创建DataFrame
        df_details = pd.DataFrame(test_stats['file_details'])
        
        # 计算摘要统计
        summary = {
            'total_test_files': test_stats['total_test_files'],
            'total_test_classes': test_stats['test_classes'],
            'total_test_methods': test_stats['test_methods'],
            'total_assertions': test_stats['assertions'],
            'total_fixtures': test_stats['test_fixtures'],
            'avg_assertions_per_test': test_stats['assertions'] / test_stats['test_methods'] if test_stats['test_methods'] > 0 else 0,
            'avg_tests_per_file': test_stats['test_methods'] / test_stats['total_test_files'] if test_stats['total_test_files'] > 0 else 0
        }
        
        if coverage_data:
            summary.update({
                'line_coverage': coverage_data.get('coverage_percentage', 0),
                'branch_coverage': coverage_data.get('branch_coverage', 0),
                'missing_lines_count': len(coverage_data.get('missing_lines', []))
            })
        
        # 保存详细数据
        df_details.to_csv(output_dir / "test_analysis_details.csv", index=False)
        
        # 保存摘要
        with open(output_dir / "test_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 保存覆盖率缺失行
        if coverage_data and 'missing_lines' in coverage_data:
            missing_df = pd.DataFrame(coverage_data['missing_lines'])
            missing_df.to_csv(output_dir / "coverage_missing_lines.csv", index=False)
        
        print(f"测试分析报告已保存到: {output_dir}/")
        print(f"测试文件总数: {summary['total_test_files']}")
        print(f"测试方法总数: {summary['total_test_methods']}")
        print(f"断言总数: {summary['total_assertions']}")
        
        if coverage_data:
            print(f"行覆盖率: {summary.get('line_coverage', 0):.2f}%")
            print(f"分支覆盖率: {summary.get('branch_coverage', 0):.2f}%")
        
        return df_details
# [file content end]