# [file name]: security_analyzer.py
# [file content begin]
import ast
import re
from typing import Dict, List, Set, Tuple
import pandas as pd
from pathlib import Path
import json

class SecurityAnalyzer:
    """分析代码中的安全漏洞和风险"""
    
    def __init__(self):
        # 定义安全漏洞模式
        self.vulnerability_patterns = {
            'exec_usage': [
                (r'\beval\s*\(', 'eval函数使用', '高风险'),
                (r'\bexec\s*\(', 'exec函数使用', '高风险'),
                (r'\bcompile\s*\(.*\)\s*\.*', '动态代码编译', '中风险'),
            ],
            'injection_risks': [
                (r'subprocess\.(call|check_call|check_output|run)', '命令注入风险', '高风险'),
                (r'os\.system\s*\(', '系统命令执行', '高风险'),
                (r'\.execute\s*\(.*\)', 'SQL/命令执行', '高风险'),
                (r'\.executemany\s*\(', '批量执行风险', '中风险'),
            ],
            'hardcoded_secrets': [
                (r'password\s*=\s*["\'].*["\']', '硬编码密码', '高风险'),
                (r'api_key\s*=\s*["\'].*["\']', '硬编码API密钥', '高风险'),
                (r'secret\s*=\s*["\'].*["\']', '硬编码密钥', '高风险'),
                (r'token\s*=\s*["\'].*["\']', '硬编码令牌', '高风险'),
            ],
            'input_validation': [
                (r'input\s*\(', '直接使用input()', '中风险'),
                (r'\.read\s*\(.*\)', '未验证的文件读取', '中风险'),
                (r'\.loads\s*\(.*\)', '未验证的JSON解析', '中风险'),
            ],
            'crypto_issues': [
                (r'md5\s*\(', '使用MD5哈希', '中风险'),
                (r'sha1\s*\(', '使用SHA1哈希', '中风险'),
                (r'random\.randint', '密码学不安全随机数', '高风险'),
                (r'random\.choice', '密码学不安全随机数', '高风险'),
            ],
            'file_operations': [
                (r'open\s*\(.*,\s*["\']w["\']\)', '文件写入风险', '中风险'),
                (r'\.write\s*\(.*\)', '文件写入操作', '低风险'),
                (r'\.save\s*\(.*\)', '文件保存操作', '低风险'),
            ]
        }
        
        # 定义敏感函数
        self.sensitive_functions = {
            'os.system', 'os.popen', 'subprocess.call', 'subprocess.Popen',
            'eval', 'exec', 'compile', 'input',
            'pickle.load', 'pickle.loads', 'marshal.load', 'marshal.loads',
            'yaml.load', 'json.loads', 'xml.etree.ElementTree.parse'
        }
        
        # 定义安全最佳实践模式
        self.best_practices = {
            'input_validation': r'assert\s+|isinstance\s*\(|if\s+.*is\s+not\s+None',
            'error_handling': r'try:\s*|\bexcept\s+',
            'type_hints': r'def\s+\w+\s*\(.*\)\s*->\s*\w+:',
            'logging': r'logging\.(info|warning|error|debug)',
            'context_managers': r'with\s+open\s*\(|with\s+\w+\(\)\s+as\s+'
        }
    
    def analyze_file_security(self, file_path: str, code: str) -> Dict:
        """分析单个文件的安全状况"""
        issues = []
        best_practices_found = set()
        
        # 1. 正则表达式匹配漏洞模式
        for category, patterns in self.vulnerability_patterns.items():
            for pattern, description, risk_level in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    issues.append({
                        'category': category,
                        'pattern': pattern,
                        'description': description,
                        'risk_level': risk_level,
                        'line': self._get_line_number(code, match.start()),
                        'match': match.group(0)[:100]  # 截断匹配文本
                    })
        
        # 2. AST分析敏感函数调用
        try:
            tree = ast.parse(code)
            ast_issues = self._analyze_ast_security(tree)
            issues.extend(ast_issues)
        except SyntaxError:
            pass
        
        # 3. 检查安全最佳实践
        for practice, pattern in self.best_practices.items():
            if re.search(pattern, code):
                best_practices_found.add(practice)
        
        # 4. 计算安全分数
        security_score = self._calculate_security_score(len(issues), len(best_practices_found))
        
        return {
            'file_path': file_path,
            'total_issues': len(issues),
            'high_risk_issues': len([i for i in issues if i['risk_level'] == '高风险']),
            'medium_risk_issues': len([i for i in issues if i['risk_level'] == '中风险']),
            'low_risk_issues': len([i for i in issues if i['risk_level'] == '低风险']),
            'best_practices_count': len(best_practices_found),
            'best_practices': list(best_practices_found),
            'security_score': security_score,
            'issues': issues[:20]  # 只保留前20个问题避免过大
        }
    
    def _analyze_ast_security(self, tree: ast.AST) -> List[Dict]:
        """使用AST分析安全漏洞"""
        issues = []
        
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.issues = []
            
            def visit_Call(self, node):
                # 检查函数调用
                if isinstance(node.func, ast.Attribute):
                    func_name = f"{self._get_attr_name(node.func)}"
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id
                else:
                    func_name = ""
                
                # 检查敏感函数
                if func_name in self.sensitive_functions:
                    self.issues.append({
                        'category': 'sensitive_function',
                        'description': f'敏感函数调用: {func_name}',
                        'risk_level': '高风险' if func_name in ['eval', 'exec', 'os.system'] else '中风险',
                        'line': node.lineno if hasattr(node, 'lineno') else 0,
                        'match': func_name
                    })
                
                # 检查pickle.loads参数
                if func_name == 'pickle.loads':
                    if node.args and isinstance(node.args[0], ast.Name):
                        arg_name = node.args[0].id
                        self.issues.append({
                            'category': 'insecure_deserialization',
                            'description': f'不安全的反序列化: pickle.loads({arg_name})',
                            'risk_level': '高风险',
                            'line': node.lineno if hasattr(node, 'lineno') else 0,
                            'match': f'pickle.loads({arg_name})'
                        })
                
                self.generic_visit(node)
            
            def _get_attr_name(self, node: ast.Attribute) -> str:
                """获取属性访问的完整名称"""
                if isinstance(node.value, ast.Name):
                    return f"{node.value.id}.{node.attr}"
                elif isinstance(node.value, ast.Attribute):
                    return f"{self._get_attr_name(node.value)}.{node.attr}"
                return node.attr
        
        visitor = SecurityVisitor()
        visitor.visit(tree)
        return visitor.issues
    
    def _get_line_number(self, code: str, position: int) -> int:
        """根据字符位置计算行号"""
        lines = code[:position].split('\n')
        return len(lines)
    
    def _calculate_security_score(self, issue_count: int, best_practice_count: int) -> float:
        """计算安全分数"""
        base_score = 100.0
        
        # 根据问题数量扣分
        issue_penalty = min(issue_count * 5, 50)  # 每个问题最多扣5分，最多扣50分
        
        # 根据最佳实践加分
        practice_bonus = min(best_practice_count * 5, 20)  # 每个最佳实践加5分，最多加20分
        
        score = base_score - issue_penalty + practice_bonus
        return max(0.0, min(100.0, score))
    
    def analyze_dependencies(self, imports: List[Dict]) -> Dict:
        """分析依赖包的安全性"""
        vulnerable_packages = []
        outdated_packages = []
        
        # 已知的安全漏洞包模式（示例）
        known_vulnerable = {
            'pickle': '可能存在反序列化漏洞',
            'marshal': '可能存在反序列化漏洞',
            'xml.etree.ElementTree': 'XML外部实体注入风险',
            'yaml': 'YAML反序列化漏洞',
            'subprocess': '命令注入风险',
            'os': '系统命令执行风险'
        }
        
        for imp in imports:
            module = imp.get('module', '')
            if module in known_vulnerable:
                vulnerable_packages.append({
                    'package': module,
                    'risk': known_vulnerable[module],
                    'recommendation': '使用更安全的替代方案或确保输入验证'
                })
        
        return {
            'vulnerable_packages': vulnerable_packages,
            'outdated_packages': outdated_packages,
            'total_risky_packages': len(vulnerable_packages)
        }
    
    def generate_security_report(self, analysis_results: List[Dict]) -> pd.DataFrame:
        """生成安全分析报告"""
        output_dir = Path("reports/security")
        output_dir.mkdir(exist_ok=True)
        
        # 汇总统计
        summary = {
            'total_files': len(analysis_results),
            'total_issues': sum(r['total_issues'] for r in analysis_results),
            'high_risk_files': sum(1 for r in analysis_results if r['high_risk_issues'] > 0),
            'avg_security_score': sum(r['security_score'] for r in analysis_results) / len(analysis_results),
            'files_with_no_best_practices': sum(1 for r in analysis_results if r['best_practices_count'] == 0),
            'most_common_issue': self._find_most_common_issue(analysis_results)
        }
        
        # 创建DataFrame
        df_summary = pd.DataFrame([{
            'file_path': r['file_path'],
            'security_score': r['security_score'],
            'total_issues': r['total_issues'],
            'high_risk': r['high_risk_issues'],
            'medium_risk': r['medium_risk_issues'],
            'low_risk': r['low_risk_issues'],
            'best_practices': ', '.join(r['best_practices'])
        } for r in analysis_results])
        
        # 收集所有问题
        all_issues = []
        for result in analysis_results:
            for issue in result.get('issues', []):
                issue['file_path'] = result['file_path']
                all_issues.append(issue)
        
        df_issues = pd.DataFrame(all_issues)
        
        # 按风险等级分组
        risk_summary = df_issues.groupby('risk_level').agg({
            'category': 'count',
            'description': lambda x: ', '.join(x.value_counts().head(3).index.tolist())
        }).rename(columns={'category': 'count'})
        
        # 保存结果
        df_summary.to_csv(output_dir / "security_summary.csv", index=False)
        if not df_issues.empty:
            df_issues.to_csv(output_dir / "security_issues.csv", index=False)
        
        # 保存JSON报告
        report = {
            'summary': summary,
            'risk_distribution': risk_summary.to_dict(),
            'top_risky_files': df_summary.nlargest(10, 'high_risk').to_dict('records'),
            'security_recommendations': self._generate_recommendations(summary, risk_summary)
        }
        
        with open(output_dir / "security_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # 生成可视化数据
        self._generate_security_visualization(df_summary, df_issues, output_dir)
        
        print(f"安全分析报告已保存到: {output_dir}/")
        print(f"分析文件数: {summary['total_files']}")
        print(f"安全漏洞总数: {summary['total_issues']}")
        print(f"高风险文件数: {summary['high_risk_files']}")
        print(f"平均安全分数: {summary['avg_security_score']:.2f}")
        
        return df_summary
    
    def _find_most_common_issue(self, results: List[Dict]) -> str:
        """查找最常见的问题类型"""
        issue_counter = {}
        
        for result in results:
            for issue in result.get('issues', []):
                issue_type = issue.get('description', 'Unknown')
                issue_counter[issue_type] = issue_counter.get(issue_type, 0) + 1
        
        if issue_counter:
            return max(issue_counter.items(), key=lambda x: x[0])[0]
        return "无"
    
    def _generate_recommendations(self, summary: Dict, risk_summary: pd.DataFrame) -> List[str]:
        """生成安全建议"""
        recommendations = []
        
        if summary['high_risk_files'] > 0:
            recommendations.append("立即审查高风险文件，修复eval/exec等危险函数的使用")
        
        if summary['avg_security_score'] < 70:
            recommendations.append("整体安全分数较低，建议进行全面的安全代码审查")
        
        if '高风险' in risk_summary.index:
            high_risk_count = risk_summary.loc['高风险', 'count']
            if high_risk_count > 10:
                recommendations.append(f"发现{high_risk_count}个高风险问题，需要优先处理")
        
        if summary['files_with_no_best_practices'] > summary['total_files'] * 0.3:
            recommendations.append("超过30%的文件未使用安全最佳实践，建议加强安全培训")
        
        # 通用建议
        recommendations.extend([
            "对所有用户输入进行严格的验证和清理",
            "避免使用eval、exec和pickle.loads等危险函数",
            "使用参数化查询防止SQL注入",
            "对敏感数据进行加密存储",
            "实现适当的错误处理，避免信息泄露",
            "定期更新依赖包到最新版本",
            "实施代码审查和安全测试"
        ])
        
        return recommendations
    
    def _generate_security_visualization(self, df_summary: pd.DataFrame, 
                                       df_issues: pd.DataFrame, output_dir: Path):
        """生成安全可视化图表"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(15, 10))
            
            # 1. 安全分数分布
            plt.subplot(2, 2, 1)
            plt.hist(df_summary['security_score'], bins=20, edgecolor='black', alpha=0.7)
            plt.axvline(df_summary['security_score'].mean(), color='red', linestyle='--')
            plt.xlabel('安全分数')
            plt.ylabel('文件数量')
            plt.title('安全分数分布')
            
            # 2. 风险等级分布
            plt.subplot(2, 2, 2)
            if not df_issues.empty:
                risk_counts = df_issues['risk_level'].value_counts()
                plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
                plt.title('安全风险等级分布')
            
            # 3. 问题类型TOP 10
            plt.subplot(2, 2, 3)
            if not df_issues.empty:
                top_issues = df_issues['description'].value_counts().head(10)
                plt.barh(range(len(top_issues)), top_issues.values)
                plt.yticks(range(len(top_issues)), top_issues.index)
                plt.xlabel('出现次数')
                plt.title('最常见的安全问题TOP 10')
                plt.gca().invert_yaxis()
            
            # 4. 安全分数与问题数量的关系
            plt.subplot(2, 2, 4)
            plt.scatter(df_summary['total_issues'], df_summary['security_score'], 
                       alpha=0.6, s=50)
            plt.xlabel('问题数量')
            plt.ylabel('安全分数')
            plt.title('安全分数 vs 问题数量')
            
            plt.tight_layout()
            plt.savefig(output_dir / "security_analysis.png", dpi=150)
            plt.close()
            
        except ImportError:
            print("未安装matplotlib/seaborn，跳过可视化")
# [file content end]