import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any

class StyleChecker:
    """代码规范检查器"""
    
    def __init__(self):
        self.checkers = {
            'flake8': self._check_with_flake8,
            'pycodestyle': self._check_with_pycodestyle
        }
    
    def check_file(self, file_path: str) -> List[Dict[str, Any]]:
        """检查单个文件的代码规范"""
        issues = []
        
        # 使用flake8检查
        flake8_issues = self._check_with_flake8(file_path)
        issues.extend(flake8_issues)
        
        # 可以添加其他检查器
        
        return issues
    
    def _check_with_flake8(self, file_path: str) -> List[Dict[str, Any]]:
        """使用flake8进行检查"""
        issues = []
        
        try:
            # 运行flake8
            result = subprocess.run(
                ['flake8', '--format=%(row)d,%(col)d,%(code)s,%(text)s', file_path],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            # 解析输出
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(',', 3)
                    if len(parts) >= 4:
                        issues.append({
                            'line': int(parts[0]),
                            'column': int(parts[1]),
                            'code': parts[2],
                            'message': parts[3],
                            'tool': 'flake8'
                        })
                        
        except Exception as e:
            print(f"flake8检查出错: {e}")
        
        return issues
    
    def _check_with_pycodestyle(self, file_path: str) -> List[Dict[str, Any]]:
        """使用pycodestyle进行检查"""
        issues = []
        
        try:
            result = subprocess.run(
                ['pycodestyle', '--format=%(row)d,%(col)d,%(code)s,%(text)s', file_path],
                capture_output=True,
                text=True
            )
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(',', 3)
                    if len(parts) >= 4:
                        issues.append({
                            'line': int(parts[0]),
                            'column': int(parts[1]),
                            'code': parts[2],
                            'message': parts[3],
                            'tool': 'pycodestyle'
                        })
                        
        except Exception as e:
            print(f"pycodestyle检查出错: {e}")
        
        return issues
    
    def check_pep8_compliance(self, code: str) -> Dict[str, Any]:
        """检查PEP 8合规性"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            issues = self.check_file(temp_file)
            
            # 分类统计
            issue_types = {}
            for issue in issues:
                code_prefix = issue['code'][0] if issue['code'] else 'E'
                issue_types[code_prefix] = issue_types.get(code_prefix, 0) + 1
            
            return {
                'total_issues': len(issues),
                'issue_types': issue_types,
                'issues': issues[:10]  # 只返回前10个问题
            }
        finally:
            os.unlink(temp_file)