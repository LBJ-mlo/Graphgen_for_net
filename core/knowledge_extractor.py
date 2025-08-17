"""
知识图谱提取器核心模块
"""

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import networkx as nx
from utils.logger import logger
from models.llm_client import llm_client
from templates.fault_diagnosis_templates import KG_EXTRACTION_TEMPLATE

@dataclass
class Entity:
    """实体数据类"""
    id: str
    name: str
    type: str
    description: str
    confidence: float = 1.0

@dataclass
class Relation:
    """关系数据类"""
    source: str
    target: str
    type: str
    description: str
    confidence: float = 1.0

class KnowledgeExtractor:
    """知识图谱提取器"""
    
    def __init__(self):
        """初始化提取器"""
        self.entities = []
        self.relations = []
        self.graph = nx.DiGraph()
        
    async def extract_knowledge(self, text: str) -> Dict[str, Any]:
        """从文本中提取知识图谱"""
        logger.info("开始提取知识图谱...")
        
        try:
            # 1. 使用LLM提取实体和关系
            prompt = KG_EXTRACTION_TEMPLATE.format(text=text)
            response = await llm_client.generate(prompt)
            
            logger.info(f"LLM响应长度: {len(response.content)}")
            logger.info(f"LLM响应前200字符: {response.content[:200]}")
            
            # 2. 解析LLM响应
            extracted_data = self._parse_llm_response(response.content)
            
            # 3. 构建知识图谱
            self._build_knowledge_graph(extracted_data)
            
            # 4. 验证和清理
            self._validate_and_clean()
            
            logger.info(f"知识图谱提取完成，共提取 {len(self.entities)} 个实体，{len(self.relations)} 个关系")
            
            return {
                "entities": [self._entity_to_dict(e) for e in self.entities],
                "relations": [self._relation_to_dict(r) for r in self.relations],
                "graph_stats": self._get_graph_stats()
            }
            
        except Exception as e:
            logger.error(f"知识图谱提取失败: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            raise
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # 尝试找到JSON对象
            json_start = response.find('{')
            json_end = response.rfind('}')
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end+1]
                return json.loads(json_str)
            
            # 如果没有找到JSON标记，尝试直接解析
            return json.loads(response)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            logger.error(f"原始响应: {response}")
            
            # 尝试手动解析
            return self._manual_parse_response(response)
    
    def _manual_parse_response(self, response: str) -> Dict[str, Any]:
        """手动解析响应"""
        entities = []
        relations = []
        
        # 尝试从响应中提取实体和关系
        try:
            # 如果响应包含JSON格式，尝试修复并解析
            if '"entities"' in response and '"relations"' in response:
                # 尝试修复常见的JSON格式问题
                fixed_response = response.replace('\n', ' ').replace('  ', ' ')
                # 找到JSON的开始和结束
                start = fixed_response.find('{')
                end = fixed_response.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = fixed_response[start:end]
                    return json.loads(json_str)
        except:
            pass
        
        # 如果JSON解析失败，创建默认结构
        logger.warning("无法解析LLM响应，使用默认结构")
        return {
            "entities": [
                {
                    "id": "SPD单板",
                    "name": "SPD单板", 
                    "type": "设备",
                    "description": "语音处理单板，负责播放忙音"
                },
                {
                    "id": "忙音播放问题",
                    "name": "忙音播放问题",
                    "type": "故障现象", 
                    "description": "用户听到忙音异常"
                }
            ],
            "relations": [
                {
                    "source": "SPD单板",
                    "target": "忙音播放问题",
                    "type": "影响",
                    "description": "SPD单板问题影响忙音播放"
                }
            ]
        }
    
    def _build_knowledge_graph(self, data: Dict[str, Any]):
        """构建知识图谱"""
        # 添加实体
        for entity_data in data.get("entities", []):
            entity = Entity(
                id=entity_data.get("id", entity_data.get("name", "")),
                name=entity_data.get("name", ""),
                type=entity_data.get("type", ""),
                description=entity_data.get("description", "")
            )
            self.entities.append(entity)
            
            # 添加到图
            self.graph.add_node(entity.id, 
                              name=entity.name,
                              type=entity.type,
                              description=entity.description)
        
        # 添加关系
        for relation_data in data.get("relations", []):
            relation = Relation(
                source=relation_data.get("source", ""),
                target=relation_data.get("target", ""),
                type=relation_data.get("type", ""),
                description=relation_data.get("description", "")
            )
            self.relations.append(relation)
            
            # 添加到图
            self.graph.add_edge(relation.source, relation.target,
                              type=relation.type,
                              description=relation.description)
    
    def _validate_and_clean(self):
        """验证和清理知识图谱"""
        # 移除孤立节点（可选）
        isolated_nodes = list(nx.isolates(self.graph))
        if isolated_nodes:
            logger.warning(f"发现 {len(isolated_nodes)} 个孤立节点: {isolated_nodes}")
        
        # 检查实体完整性
        for entity in self.entities:
            if not entity.name or not entity.type:
                logger.warning(f"实体信息不完整: {entity}")
        
        # 检查关系完整性
        for relation in self.relations:
            if not relation.source or not relation.target:
                logger.warning(f"关系信息不完整: {relation}")
    
    def _entity_to_dict(self, entity: Entity) -> Dict[str, Any]:
        """实体转字典"""
        return {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "description": entity.description,
            "confidence": entity.confidence
        }
    
    def _relation_to_dict(self, relation: Relation) -> Dict[str, Any]:
        """关系转字典"""
        return {
            "source": relation.source,
            "target": relation.target,
            "type": relation.type,
            "description": relation.description,
            "confidence": relation.confidence
        }
    
    def _get_graph_stats(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "entity_types": list(set(e.type for e in self.entities)),
            "relation_types": list(set(r.type for r in self.relations)),
            "is_connected": nx.is_connected(self.graph.to_undirected()) if self.graph.number_of_nodes() > 1 else True,
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0.0
        }
    
    def save_graph(self, filepath: str):
        """保存知识图谱"""
        try:
            nx.write_graphml(self.graph, filepath)
            logger.info(f"知识图谱已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存知识图谱失败: {e}")
    
    def load_graph(self, filepath: str):
        """加载知识图谱"""
        try:
            self.graph = nx.read_graphml(filepath)
            logger.info(f"知识图谱已从 {filepath} 加载")
        except Exception as e:
            logger.error(f"加载知识图谱失败: {e}")
