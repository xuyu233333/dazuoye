import ast
import libcst as cst
from typing import Dict, Any, List
import inspect

class ASTParser:
    """AST解析器，支持标准ast和libcst"""
    
    def __init__(self):
        self.standard_ast = True
        self.libcst_available = True
    
    def parse_code(self, code: str) -> ast.AST:
        """使用标准ast解析代码"""
        try:
            return ast.parse(code)
        except SyntaxError as e:
            print(f"语法错误: {e}")
            raise
    
    def parse_with_libcst(self, code: str) -> cst.Module:
        """使用libcst解析代码（更精确的源码信息）"""
        try:
            return cst.parse_module(code)
        except Exception as e:
            print(f"libcst解析错误: {e}")
            raise
    
    def extract_functions(self, ast_tree: ast.AST) -> List[Dict[str, Any]]:
        """提取所有函数定义信息"""
        functions = []
        
        class FunctionVisitor(ast.NodeVisitor):
            def __init__(self):
                self.functions = []
                self.current_class = None
            
            def visit_ClassDef(self, node):
                old_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = old_class
            
            def visit_FunctionDef(self, node):
                # 计算函数行数
                start_line = node.lineno
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                
                # 统计参数数量
                arg_count = len(node.args.args)
                
                # 检查是否有文档字符串
                has_docstring = ast.get_docstring(node) is not None
                
                # 统计注释数量
                comment_count = sum(1 for n in ast.walk(node) 
                                  if isinstance(n, ast.Expr) 
                                  and isinstance(n.value, ast.Constant)
                                  and isinstance(n.value.value, str))
                
                func_info = {
                    'name': node.name,
                    'class': self.current_class,
                    'start_line': start_line,
                    'end_line': end_line,
                    'lines': end_line - start_line + 1,
                    'arg_count': arg_count,
                    'has_docstring': has_docstring,
                    'comment_count': comment_count,
                    'decorators': [d.id for d in node.decorator_list 
                                 if isinstance(d, ast.Name)]
                }
                self.functions.append(func_info)
                
                self.generic_visit(node)
        
        visitor = FunctionVisitor()
        visitor.visit(ast_tree)
        return visitor.functions
    
    def extract_imports(self, ast_tree: ast.AST) -> List[Dict[str, Any]]:
        """提取导入语句信息"""
        imports = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'module': alias.name,
                        'alias': alias.asname,
                        'type': 'import'
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append({
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'type': 'from_import'
                    })
        
        return imports