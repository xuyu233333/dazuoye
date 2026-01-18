# [file name]: dependency_analyzer.py
# [file content begin]
import ast
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Set, Tuple
from pathlib import Path

class DependencyAnalyzer:
    """分析代码中的依赖关系"""
    
    def __init__(self):
        self.import_graph = nx.DiGraph()
        self.internal_deps = defaultdict(set)
        self.external_deps = defaultdict(set)
        
    def analyze_file(self, file_path: str, code: str) -> Dict:
        """分析单个文件的依赖关系"""
        imports = self._extract_imports(code)
        
        # 分类导入
        internal = []
        external = []
        
        for imp in imports:
            if self._is_internal_import(imp['module']):
                internal.append(imp)
            else:
                external.append(imp)
        
        return {
            'file': file_path,
            'total_imports': len(imports),
            'internal_imports': len(internal),
            'external_imports': len(external),
            'import_details': imports
        }
    
    def _extract_imports(self, code: str) -> List[Dict]:
        """从代码中提取导入语句"""
        imports = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append({
                            'module': alias.name,
                            'name': None,
                            'alias': alias.asname,
                            'type': 'import',
                            'line': node.lineno if hasattr(node, 'lineno') else 0
                        })
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    level = node.level  # 相对导入级别
                    
                    for alias in node.names:
                        imports.append({
                            'module': module,
                            'name': alias.name,
                            'alias': alias.asname,
                            'type': 'from_import',
                            'level': level,
                            'line': node.lineno if hasattr(node, 'lineno') else 0
                        })
        except SyntaxError:
            pass
        
        return imports
    
    def _is_internal_import(self, module_name: str) -> bool:
        """判断是否为内部导入（pandas相关）"""
        if not module_name:
            return False
        
        internal_prefixes = ['pandas', '.', '..']
        
        # 检查是否以pandas开头或者是相对导入
        return any(module_name.startswith(prefix) for prefix in internal_prefixes)
    
    def build_dependency_graph(self, analysis_results: List[Dict]) -> nx.DiGraph:
        """构建依赖关系图"""
        G = nx.DiGraph()
        
        # 添加节点
        for result in analysis_results:
            file_name = Path(result['file']).stem
            G.add_node(file_name, **result)
            
            # 添加边（依赖关系）
            for imp in result['import_details']:
                if imp['type'] == 'from_import' and imp['module']:
                    # 简化处理：假设模块名对应文件名
                    source_module = Path(imp['module']).stem
                    if source_module in G.nodes:
                        G.add_edge(file_name, source_module)
        
        return G
    
    def analyze_circular_dependencies(self, graph: nx.DiGraph) -> List[List[str]]:
        """检测循环依赖"""
        try:
            cycles = list(nx.simple_cycles(graph))
            return cycles
        except:
            return []
    
    def calculate_metrics(self, graph: nx.DiGraph) -> Dict:
        """计算依赖相关指标"""
        if len(graph.nodes) == 0:
            return {}
        
        metrics = {
            'total_nodes': len(graph.nodes),
            'total_edges': len(graph.edges),
            'density': nx.density(graph),
            'avg_in_degree': sum(d for _, d in graph.in_degree()) / len(graph.nodes),
            'avg_out_degree': sum(d for _, d in graph.out_degree()) / len(graph.nodes),
        }
        
        # 计算中心性指标
        if len(graph.nodes) > 1:
            try:
                metrics['betweenness_centrality'] = nx.betweenness_centrality(graph)
                metrics['degree_centrality'] = nx.degree_centrality(graph)
                
                # 找出关键节点
                top_betweenness = sorted(
                    metrics['betweenness_centrality'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                metrics['key_modules'] = top_betweenness
                
            except Exception as e:
                print(f"计算中心性时出错: {e}")
        
        return metrics
    
    def visualize_dependencies(self, graph: nx.DiGraph, output_path: str = "reports/dependencies.png"):
        """可视化依赖关系"""
        plt.figure(figsize=(15, 12))
        
        # 使用不同的布局算法
        try:
            pos = nx.spring_layout(graph, k=1, iterations=50)
        except:
            pos = nx.circular_layout(graph)
        
        # 计算节点大小（基于度）
        node_sizes = [300 + 500 * (graph.degree[node] / max(graph.degree().values()) if graph.degree() else 1)
                     for node in graph.nodes()]
        
        # 计算节点颜色（基于入度）
        in_degrees = dict(graph.in_degree())
        max_in_degree = max(in_degrees.values()) if in_degrees else 1
        node_colors = [plt.cm.Reds(0.3 + 0.7 * (in_degrees.get(node, 0) / max_in_degree))
                      for node in graph.nodes()]
        
        # 绘制图形
        nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, 
                              node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(graph, pos, edge_color='gray', 
                              arrows=True, arrowsize=10, alpha=0.5)
        nx.draw_networkx_labels(graph, pos, font_size=8, font_weight='bold')
        
        # 添加标题和说明
        plt.title(f"代码依赖关系图 (节点数: {len(graph.nodes)}, 边数: {len(graph.edges)})", fontsize=14)
        plt.axis('off')
        
        # 添加图例
        plt.text(0.05, 0.05, 
                f"节点大小表示连接度\n颜色深度表示入度",
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存为GEXF格式（可用于Gephi等工具）
        nx.write_gexf(graph, output_path.replace('.png', '.gexf'))
        
        print(f"依赖关系图已保存到: {output_path}")
    
    def generate_dependency_report(self, analysis_results: List[Dict]) -> pd.DataFrame:
        """生成依赖分析报告"""
        # 构建依赖图
        graph = self.build_dependency_graph(analysis_results)
        
        # 计算指标
        metrics = self.calculate_metrics(graph)
        
        # 检测循环依赖
        cycles = self.analyze_circular_dependencies(graph)
        
        # 可视化
        self.visualize_dependencies(graph)
        
        # 生成汇总数据
        summary = []
        for result in analysis_results:
            summary.append({
                'file': result['file'],
                'total_imports': result['total_imports'],
                'internal_imports': result['internal_imports'],
                'external_imports': result['external_imports'],
                'has_circular_deps': any(result['file'] in cycle for cycle in cycles)
            })
        
        df = pd.DataFrame(summary)
        
        # 保存详细结果
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        
        # 保存循环依赖信息
        if cycles:
            cycles_df = pd.DataFrame({
                'cycle_id': range(len(cycles)),
                'modules': [', '.join(cycle) for cycle in cycles],
                'length': [len(cycle) for cycle in cycles]
            })
            cycles_df.to_csv(output_dir / "circular_dependencies.csv", index=False)
            print(f"发现 {len(cycles)} 个循环依赖")
        
        # 保存关键模块信息
        if 'key_modules' in metrics:
            key_modules_df = pd.DataFrame(metrics['key_modules'], 
                                         columns=['module', 'betweenness_centrality'])
            key_modules_df.to_csv(output_dir / "key_modules.csv", index=False)
        
        return df
# [file content end]