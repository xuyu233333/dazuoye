# [file name]: complexity_analyzer.py
# [file content begin]
import ast
import math
from typing import Dict, List, Tuple, Set
import pandas as pd
from pathlib import Path
import json

class AdvancedComplexityAnalyzer:
    """高级复杂度分析器"""
    
    def __init__(self):
        self.metrics = {}
        
    def analyze_complexity(self, code: str, ast_tree: ast.AST) -> Dict:
        """执行深度复杂度分析"""
        metrics = {}
        
        # 基本复杂度指标
        metrics.update(self._calculate_basic_complexity(code, ast_tree))
        
        # 认知复杂度
        metrics['cognitive_complexity'] = self._calculate_cognitive_complexity(ast_tree)
        
        # Halstead指标
        halstead_metrics = self._calculate_halstead_metrics(code)
        metrics.update(halstead_metrics)
        
        # 嵌套深度分析
        nesting_metrics = self._analyze_nesting_depth(ast_tree)
        metrics.update(nesting_metrics)
        
        # 函数内聚度
        cohesion_metrics = self._analyze_cohesion(ast_tree)
        metrics.update(cohesion_metrics)
        
        # 计算总体复杂度评分
        metrics['overall_complexity_score'] = self._calculate_overall_score(metrics)
        
        return metrics
    
    def _calculate_basic_complexity(self, code: str, ast_tree: ast.AST) -> Dict:
        """计算基本复杂度指标"""
        metrics = {}
        
        # 统计各种语句类型
        stmt_counts = self._count_statement_types(ast_tree)
        metrics.update(stmt_counts)
        
        # 条件复杂度
        conditions = self._count_conditions(ast_tree)
        metrics['condition_count'] = conditions
        
        # 循环复杂度
        loops = self._count_loops(ast_tree)
        metrics['loop_count'] = loops
        
        # 异常处理复杂度
        exceptions = self._count_exceptions(ast_tree)
        metrics['exception_count'] = exceptions
        
        return metrics
    
    def _count_statement_types(self, tree: ast.AST) -> Dict:
        """统计各种语句类型的数量"""
        counts = {
            'assignments': 0,
            'aug_assignments': 0,
            'returns': 0,
            'yields': 0,
            'raises': 0,
            'asserts': 0,
            'deletes': 0,
            'passes': 0,
            'breaks': 0,
            'continues': 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                counts['assignments'] += 1
            elif isinstance(node, ast.AugAssign):
                counts['aug_assignments'] += 1
            elif isinstance(node, ast.Return):
                counts['returns'] += 1
            elif isinstance(node, ast.Yield) or isinstance(node, ast.YieldFrom):
                counts['yields'] += 1
            elif isinstance(node, ast.Raise):
                counts['raises'] += 1
            elif isinstance(node, ast.Assert):
                counts['asserts'] += 1
            elif isinstance(node, ast.Delete):
                counts['deletes'] += 1
            elif isinstance(node, ast.Pass):
                counts['passes'] += 1
            elif isinstance(node, ast.Break):
                counts['breaks'] += 1
            elif isinstance(node, ast.Continue):
                counts['continues'] += 1
        
        return counts
    
    def _count_conditions(self, tree: ast.AST) -> int:
        """统计条件语句数量"""
        condition_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                condition_count += 1
                # 统计elif和else
                condition_count += len(node.orelse)
            elif isinstance(node, ast.IfExp):
                condition_count += 1
            elif isinstance(node, ast.Assert):
                condition_count += 1
        
        return condition_count
    
    def _count_loops(self, tree: ast.AST) -> int:
        """统计循环语句数量"""
        loop_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                loop_count += 1
        
        return loop_count
    
    def _count_exceptions(self, tree: ast.AST) -> int:
        """统计异常处理语句数量"""
        exception_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                exception_count += 1
                exception_count += len(node.handlers)
                if node.finalbody:
                    exception_count += 1
        
        return exception_count
    
    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """计算认知复杂度（简化版）"""
        complexity = 0
        
        class CognitiveVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 0
                self.nesting_level = 0
                self.increase_for_nesting = True
            
            def visit_If(self, node):
                # 每个if语句加1
                self.complexity += 1
                
                # 嵌套加1
                if self.increase_for_nesting:
                    self.complexity += self.nesting_level
                
                # 增加嵌套级别
                self.nesting_level += 1
                old_increase = self.increase_for_nesting
                self.increase_for_nesting = True
                
                self.generic_visit(node)
                
                # 恢复状态
                self.nesting_level -= 1
                self.increase_for_nesting = old_increase
            
            def visit_For(self, node):
                self.complexity += 1
                if self.increase_for_nesting:
                    self.complexity += self.nesting_level
                
                self.nesting_level += 1
                old_increase = self.increase_for_nesting
                self.increase_for_nesting = True
                
                self.generic_visit(node)
                
                self.nesting_level -= 1
                self.increase_for_nesting = old_increase
            
            def visit_While(self, node):
                self.complexity += 1
                if self.increase_for_nesting:
                    self.complexity += self.nesting_level
                
                self.nesting_level += 1
                old_increase = self.increase_for_nesting
                self.increase_for_nesting = True
                
                self.generic_visit(node)
                
                self.nesting_level -= 1
                self.increase_for_nesting = old_increase
            
            def visit_Try(self, node):
                self.complexity += 1
                if self.increase_for_nesting:
                    self.complexity += self.nesting_level
                
                self.nesting_level += 1
                old_increase = self.increase_for_nesting
                self.increase_for_nesting = False  # except块不增加嵌套复杂度
                
                self.generic_visit(node)
                
                self.nesting_level -= 1
                self.increase_for_nesting = old_increase
        
        visitor = CognitiveVisitor()
        visitor.visit(tree)
        return visitor.complexity
    
    def _calculate_halstead_metrics(self, code: str) -> Dict:
        """计算Halstead软件科学指标"""
        # 提取操作符和操作数
        operators = set()
        operands = set()
        operator_count = 0
        operand_count = 0
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # 操作符：语句类型、运算符等
                node_type = type(node).__name__
                operators.add(node_type)
                operator_count += 1
                
                # 操作数：变量名、常量等
                if isinstance(node, ast.Name):
                    operands.add(node.id)
                    operand_count += 1
                elif isinstance(node, ast.Constant):
                    if isinstance(node.value, (str, int, float, bool)):
                        operands.add(str(node.value))
                        operand_count += 1
                elif isinstance(node, ast.Attribute):
                    operands.add(node.attr)
                    operand_count += 1
        
        except SyntaxError:
            pass
        
        # 计算Halstead指标
        n1 = len(operators)  # 不同操作符数量
        n2 = len(operands)   # 不同操作数数量
        N1 = operator_count  # 操作符总数
        N2 = operand_count   # 操作数总数
        
        # 避免除零错误
        if n1 == 0 or n2 == 0:
            return {
                'halstead_vocabulary': 0,
                'halstead_length': 0,
                'halstead_volume': 0,
                'halstead_difficulty': 0,
                'halstead_effort': 0
            }
        
        vocabulary = n1 + n2
        length = N1 + N2
        volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        effort = difficulty * volume
        
        return {
            'halstead_vocabulary': vocabulary,
            'halstead_length': length,
            'halstead_volume': volume,
            'halstead_difficulty': difficulty,
            'halstead_effort': effort
        }
    
    def _analyze_nesting_depth(self, tree: ast.AST) -> Dict:
        """分析嵌套深度"""
        max_depth = 0
        avg_depth = 0
        depth_distribution = {}
        
        class NestingVisitor(ast.NodeVisitor):
            def __init__(self):
                self.depth = 0
                self.max_depth = 0
                self.total_depth = 0
                self.node_count = 0
                self.depth_counts = {}
            
            def visit_FunctionDef(self, node):
                self._visit_with_depth(node)
            
            def visit_ClassDef(self, node):
                self._visit_with_depth(node)
            
            def visit_If(self, node):
                self._visit_with_depth(node)
            
            def visit_For(self, node):
                self._visit_with_depth(node)
            
            def visit_While(self, node):
                self._visit_with_depth(node)
            
            def visit_Try(self, node):
                self._visit_with_depth(node)
            
            def _visit_with_depth(self, node):
                self.depth += 1
                self.max_depth = max(self.max_depth, self.depth)
                self.total_depth += self.depth
                self.node_count += 1
                self.depth_counts[self.depth] = self.depth_counts.get(self.depth, 0) + 1
                
                self.generic_visit(node)
                
                self.depth -= 1
        
        visitor = NestingVisitor()
        visitor.visit(tree)
        
        avg_depth = visitor.total_depth / visitor.node_count if visitor.node_count > 0 else 0
        
        return {
            'max_nesting_depth': visitor.max_depth,
            'avg_nesting_depth': avg_depth,
            'nesting_distribution': visitor.depth_counts
        }
    
    def _analyze_cohesion(self, tree: ast.AST) -> Dict:
        """分析函数内聚度（LCOM4）"""
        cohesion_metrics = {
            'lcom4_score': 0,
            'tight_class_cohesion': 0,
            'loose_class_cohesion': 0
        }
        
        try:
            # 查找类定义
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_metrics = self._calculate_class_cohesion(node)
                    cohesion_metrics.update(class_metrics)
                    break  # 只分析第一个类
        except:
            pass
        
        return cohesion_metrics
    
    def _calculate_class_cohesion(self, class_node: ast.ClassDef) -> Dict:
        """计算类的内聚度指标"""
        methods = []
        attributes = set()
        
        # 提取方法和属性
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(item)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.add(target.id)
            elif isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    attributes.add(item.target.id)
        
        if not methods or not attributes:
            return {
                'lcom4_score': 0,
                'tight_class_cohesion': 0,
                'loose_class_cohesion': 0
            }
        
        # 计算LCOM4（缺少公共属性的方法对数量）
        disconnected_pairs = 0
        total_pairs = len(methods) * (len(methods) - 1) // 2
        
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                # 检查两个方法是否共享属性访问
                attrs_i = self._get_accessed_attributes(methods[i])
                attrs_j = self._get_accessed_attributes(methods[j])
                
                if not (attrs_i & attrs_j):
                    disconnected_pairs += 1
        
        lcom4 = disconnected_pairs
        
        # 计算紧内聚度（TCC）
        tcc = 1.0 - (disconnected_pairs / total_pairs) if total_pairs > 0 else 1.0
        
        # 计算松内聚度（LCC）- 考虑间接访问
        lcc = tcc  # 简化版本
        
        return {
            'lcom4_score': lcom4,
            'tight_class_cohesion': tcc,
            'loose_class_cohesion': lcc
        }
    
    def _get_accessed_attributes(self, func_node: ast.FunctionDef) -> Set[str]:
        """获取函数访问的属性集合"""
        attributes = set()
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Attribute):
                attributes.add(node.attr)
            elif isinstance(node, ast.Name):
                # 可能是属性访问
                pass
        
        return attributes
    
    def _calculate_overall_score(self, metrics: Dict) -> float:
        """计算总体复杂度评分"""
        score = 100.0
        
        # 根据认知复杂度扣分
        if 'cognitive_complexity' in metrics:
            cc = metrics['cognitive_complexity']
            if cc > 30:
                score -= 40
            elif cc > 20:
                score -= 25
            elif cc > 10:
                score -= 10
        
        # 根据Halstead体积扣分
        if 'halstead_volume' in metrics:
            volume = metrics['halstead_volume']
            if volume > 1000:
                score -= 30
            elif volume > 500:
                score -= 15
        
        # 根据嵌套深度扣分
        if 'max_nesting_depth' in metrics:
            depth = metrics['max_nesting_depth']
            if depth > 5:
                score -= 20
            elif depth > 3:
                score -= 10
        
        # 根据LCOM4扣分（低内聚）
        if 'lcom4_score' in metrics and metrics['lcom4_score'] > 5:
            score -= 15
        
        return max(0.0, min(100.0, score))
    
    def generate_complexity_report(self, analysis_results: List[Dict]) -> pd.DataFrame:
        """生成复杂度分析报告"""
        output_dir = Path("reports/complexity")
        output_dir.mkdir(exist_ok=True)
        
        # 创建DataFrame
        df = pd.DataFrame(analysis_results)
        
        # 计算汇总统计
        summary = {}
        
        if not df.empty:
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            for col in numeric_cols:
                if col not in ['file_path', 'module']:
                    summary[f'{col}_mean'] = float(df[col].mean())
                    summary[f'{col}_median'] = float(df[col].median())
                    summary[f'{col}_std'] = float(df[col].std())
                    summary[f'{col}_max'] = float(df[col].max())
                    summary[f'{col}_min'] = float(df[col].min())
        
        # 找出复杂度最高的文件
        if 'overall_complexity_score' in df.columns:
            most_complex = df.nsmallest(10, 'overall_complexity_score')[['file_path', 'overall_complexity_score']]
            summary['most_complex_files'] = most_complex.to_dict('records')
        
        # 保存结果
        df.to_csv(output_dir / "complexity_analysis.csv", index=False)
        
        with open(output_dir / "complexity_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 生成可视化
        self._generate_complexity_visualization(df, output_dir)
        
        print(f"复杂度分析报告已保存到: {output_dir}/")
        print(f"分析文件数: {len(df)}")
        
        if 'overall_complexity_score' in df.columns:
            avg_score = df['overall_complexity_score'].mean()
            print(f"平均复杂度评分: {avg_score:.2f}")
            print(f"最佳文件: {df['overall_complexity_score'].max():.2f}")
            print(f"最差文件: {df['overall_complexity_score'].min():.2f}")
        
        return df
    
    def _generate_complexity_visualization(self, df: pd.DataFrame, output_dir: Path):
        """生成复杂度可视化图表"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 选择关键指标进行可视化
            key_metrics = [
                'overall_complexity_score',
                'cognitive_complexity',
                'halstead_volume',
                'max_nesting_depth',
                'condition_count',
                'loop_count'
            ]
            
            # 筛选存在的列
            available_metrics = [m for m in key_metrics if m in df.columns]
            
            if len(available_metrics) < 2:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 复杂度分数分布
            if 'overall_complexity_score' in df.columns:
                ax = axes[0, 0]
                ax.hist(df['overall_complexity_score'].dropna(), bins=20, 
                       edgecolor='black', alpha=0.7)
                ax.axvline(df['overall_complexity_score'].mean(), color='red', 
                          linestyle='--', label=f'平均: {df["overall_complexity_score"].mean():.2f}')
                ax.set_xlabel('复杂度评分')
                ax.set_ylabel('文件数量')
                ax.set_title('复杂度评分分布')
                ax.legend()
            
            # 2. 认知复杂度 vs Halstead体积
            if 'cognitive_complexity' in df.columns and 'halstead_volume' in df.columns:
                ax = axes[0, 1]
                scatter = ax.scatter(df['cognitive_complexity'], df['halstead_volume'],
                                   alpha=0.6, s=50, c=df.get('overall_complexity_score', 50))
                ax.set_xlabel('认知复杂度')
                ax.set_ylabel('Halstead体积')
                ax.set_title('认知复杂度 vs Halstead体积')
                if 'overall_complexity_score' in df.columns:
                    plt.colorbar(scatter, ax=ax, label='总体复杂度评分')
            
            # 3. 嵌套深度分布
            if 'max_nesting_depth' in df.columns:
                ax = axes[1, 0]
                depth_counts = df['max_nesting_depth'].value_counts().sort_index()
                ax.bar(depth_counts.index, depth_counts.values)
                ax.set_xlabel('最大嵌套深度')
                ax.set_ylabel('文件数量')
                ax.set_title('嵌套深度分布')
            
            # 4. 条件语句和循环语句数量
            if 'condition_count' in df.columns and 'loop_count' in df.columns:
                ax = axes[1, 1]
                x = range(len(df))
                width = 0.35
                ax.bar([i - width/2 for i in x], df['condition_count'].head(20), 
                      width, label='条件语句', alpha=0.7)
                ax.bar([i + width/2 for i in x], df['loop_count'].head(20), 
                      width, label='循环语句', alpha=0.7)
                ax.set_xlabel('文件索引（前20个）')
                ax.set_ylabel('数量')
                ax.set_title('条件语句 vs 循环语句')
                ax.legend()
                ax.set_xticks(x)
                ax.set_xticklabels([str(i) for i in range(1, 21)])
            
            plt.tight_layout()
            plt.savefig(output_dir / "complexity_analysis.png", dpi=150)
            plt.close()
            
        except ImportError:
            print("未安装matplotlib/seaborn，跳过可视化")
        except Exception as e:
            print(f"生成可视化时出错: {e}")
# [file content end]