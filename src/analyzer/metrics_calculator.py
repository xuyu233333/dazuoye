import ast
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze
import math
from typing import Dict, Any

class MetricsCalculator:
    """代码度量指标计算器"""
    
    def calculate_file_metrics(self, code: str, ast_tree: ast.AST) -> Dict[str, Any]:
        """计算文件的各项度量指标"""
        metrics = {}
        
        # 1. 基本统计
        lines = code.split('\n')
        metrics['total_lines'] = len(lines)
        metrics['non_empty_lines'] = sum(1 for line in lines if line.strip())
        metrics['comment_lines'] = sum(1 for line in lines 
                                      if line.strip().startswith('#'))
        
        # 2. 使用radon计算复杂度
        try:
            # 圈复杂度
            complexity_results = cc_visit(code)
            metrics['cyclomatic_complexity'] = sum(
                func.complexity for func in complexity_results
            ) if complexity_results else 0
            metrics['function_count'] = len(complexity_results)
            
            # 维护性指数
            mi_score = mi_visit(code, multi=True)
            metrics['maintainability_index'] = mi_score
            
            # 原始分析（代码行数统计）
            raw_metrics = analyze(code)
            metrics['loc'] = raw_metrics.loc
            metrics['lloc'] = raw_metrics.lloc
            metrics['sloc'] = raw_metrics.sloc
            metrics['comments'] = raw_metrics.comments
            
        except Exception as e:
            print(f"计算复杂度时出错: {e}")
            metrics.update({
                'cyclomatic_complexity': 0,
                'maintainability_index': 100,
                'loc': len(lines),
                'lloc': metrics['non_empty_lines'],
                'sloc': metrics['non_empty_lines'] - metrics['comment_lines'],
                'comments': metrics['comment_lines']
            })
        
        # 3. 自定义指标
        metrics.update(self._calculate_custom_metrics(ast_tree, lines))
        
        return metrics
    
    def _calculate_custom_metrics(self, ast_tree: ast.AST, lines: list) -> Dict[str, Any]:
        """计算自定义指标"""
        # 提取函数信息
        functions = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'lines': self._get_function_lines(node, lines),
                    'args': len(node.args.args),
                    'has_return': any(isinstance(n, ast.Return) 
                                    for n in ast.walk(node))
                }
                functions.append(func_info)
        
        # 计算基于函数的指标
        if functions:
            avg_lines = sum(f['lines'] for f in functions) / len(functions)
            avg_args = sum(f['args'] for f in functions) / len(functions)
            functions_with_return = sum(1 for f in functions if f['has_return'])
        else:
            avg_lines = avg_args = 0
            functions_with_return = 0
        
        return {
            'function_count': len(functions),
            'avg_lines_per_function': avg_lines,
            'avg_args_per_function': avg_args,
            'functions_with_return': functions_with_return,
            'max_function_lines': max((f['lines'] for f in functions), default=0)
        }
    
    def _get_function_lines(self, func_node: ast.FunctionDef, lines: list) -> int:
        """计算函数行数"""
        start_line = func_node.lineno - 1  # 转为0-based索引
        
        # 查找函数结束行
        end_line = start_line
        indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
        
        for i in range(start_line + 1, len(lines)):
            if lines[i].strip() and len(lines[i]) - len(lines[i].lstrip()) <= indent_level:
                break
            end_line = i
        
        return end_line - start_line + 1