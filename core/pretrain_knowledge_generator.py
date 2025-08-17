"""
预训练知识生成器核心模块
"""

import json
import re
import networkx as nx
from typing import Dict, List, Any
from utils.logger import logger
from models.llm_client import llm_client

class PretrainKnowledgeGenerator:
    """预训练知识生成器"""
    
    def __init__(self):
        """初始化生成器"""
        pass
        
    async def generate_pretrain_knowledge(self, 
                                        knowledge_graph: Dict[str, Any]) -> Dict[str, Any]:
        """生成预训练知识"""
        logger.info("开始生成预训练知识...")
        
        try:
            # 准备知识图谱信息
            kg_info = self._prepare_knowledge_graph_info(knowledge_graph)
            
            # 生成自然语言格式的预训练知识
            pretrain_text = await self._generate_natural_language_knowledge(kg_info)
            
            # 生成结构化的训练样本
            training_samples = self._generate_training_samples(pretrain_text, knowledge_graph)
            
            # 生成多跳推理数据
            multi_hop_data = await self._generate_multi_hop_reasoning(knowledge_graph)
            
            results = {
                "pretrain_text": pretrain_text,
                "training_samples": training_samples,
                "multi_hop_reasoning": multi_hop_data,
                "metadata": {
                    "text_length": len(pretrain_text),
                    "sample_count": len(training_samples),
                    "entity_count": len(knowledge_graph.get("entities", [])),
                    "relation_count": len(knowledge_graph.get("relations", [])),
                    "reasoning_path_count": multi_hop_data["metadata"]["path_count"],
                    "reasoning_chain_count": multi_hop_data["metadata"]["chain_count"],
                    "reasoning_qa_count": multi_hop_data["metadata"]["qa_count"]
                }
            }
            
            logger.info(f"预训练知识生成完成")
            return results
            
        except Exception as e:
            logger.error(f"预训练知识生成失败: {e}")
            raise
    
    def _prepare_knowledge_graph_info(self, knowledge_graph: Dict[str, Any]) -> str:
        """准备知识图谱信息"""
        entities = knowledge_graph.get("entities", [])
        relations = knowledge_graph.get("relations", [])
        
        entity_info = []
        for entity in entities:
            entity_info.append(f"- {entity['name']} ({entity['type']}): {entity['description']}")
        
        relation_info = []
        for relation in relations:
            relation_info.append(f"- {relation['source']} --[{relation['type']}]--> {relation['target']}")
        
        return f"实体: {chr(10).join(entity_info)}\n关系: {chr(10).join(relation_info)}"
    
    async def _generate_natural_language_knowledge(self, kg_info: str) -> str:
        """生成自然语言格式的预训练知识"""
        from templates.fault_diagnosis_templates import PRETRAIN_KNOWLEDGE_TEMPLATE
        
        prompt = PRETRAIN_KNOWLEDGE_TEMPLATE.format(knowledge_graph=kg_info)
        response = await llm_client.generate(prompt)
        return response.content
    
    def _generate_training_samples(self, pretrain_text: str, knowledge_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成结构化的训练样本"""
        samples = []
        
        # 1. 实体描述样本
        for entity in knowledge_graph.get("entities", []):
            sample = {
                "type": "entity_description",
                "entity": entity["name"],
                "entity_type": entity["type"],
                "description": entity["description"],
                "training_text": f"在电信网络故障诊断中，{entity['name']}是一个{entity['type']}，{entity['description']}。"
            }
            samples.append(sample)
        
        # 2. 关系描述样本
        for relation in knowledge_graph.get("relations", []):
            # 获取具体的实体名称
            source_entity = None
            target_entity = None
            
            # 查找源实体和目标实体的具体名称
            for entity in knowledge_graph.get("entities", []):
                if entity["id"] == relation["source"]:
                    source_entity = entity["name"]
                if entity["id"] == relation["target"]:
                    target_entity = entity["name"]
            
            # 如果找到了实体名称，使用具体名称；否则使用ID
            source_name = source_entity if source_entity else relation["source"]
            target_name = target_entity if target_entity else relation["target"]
            
            sample = {
                "type": "relation_description",
                "source": source_name,
                "target": target_name,
                "relation_type": relation["type"],
                "description": relation["description"],
                "training_text": f"在故障诊断中，{source_name}会{relation['type']}{target_name}，具体表现为{relation['description']}。"
            }
            samples.append(sample)
        
        # 3. 故障诊断流程样本
        sample = {
            "type": "diagnosis_process",
            "training_text": "电信网络故障诊断是一个系统性的过程。首先需要识别故障现象，然后分析可能的故障原因，接着制定解决方案，最后验证修复效果。在整个过程中，需要综合考虑设备状态、配置参数、环境因素等多个方面。"
        }
        samples.append(sample)
        
        # 4. 知识图谱总结样本
        entity_names = [e["name"] for e in knowledge_graph.get("entities", [])]
        relation_types = [r["type"] for r in knowledge_graph.get("relations", [])]
        
        sample = {
            "type": "knowledge_summary",
            "entities": entity_names,
            "relations": relation_types,
            "training_text": f"该故障案例涉及{len(entity_names)}个关键实体，包括{', '.join(entity_names[:3])}等，实体间存在{', '.join(set(relation_types))}等关系类型。"
        }
        samples.append(sample)
        
        return samples
    
    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """解析JSON响应"""
        try:
            json_matches = re.findall(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            results = []
            
            for json_str in json_matches:
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list):
                        results.extend(data)
                    else:
                        results.append(data)
                except json.JSONDecodeError:
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"解析JSON响应失败: {e}")
            return []
    
    def save_knowledge(self, knowledge: Dict[str, Any], filepath: str):
        """保存预训练知识"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(knowledge, f, ensure_ascii=False, indent=2)
            logger.info(f"预训练知识已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存预训练知识失败: {e}")
    
    # 多跳推理相关方法
    async def _generate_multi_hop_reasoning(self, knowledge_graph: Dict[str, Any]) -> Dict[str, Any]:
        """生成多跳推理数据"""
        logger.info("开始生成多跳推理数据...")
        
        try:
            # 1. 构建知识图谱网络
            graph = self._build_knowledge_graph(knowledge_graph)
            
            # 2. 发现推理路径
            reasoning_paths = self._discover_reasoning_paths(graph, knowledge_graph)
            
            # 3. 生成推理链
            reasoning_chains = await self._generate_reasoning_chains(reasoning_paths, knowledge_graph)
            
            # 4. 生成推理问答对
            reasoning_qa_pairs = self._generate_reasoning_qa_pairs(reasoning_chains)
            
            results = {
                "reasoning_paths": reasoning_paths,
                "reasoning_chains": reasoning_chains,
                "reasoning_qa_pairs": reasoning_qa_pairs,
                "metadata": {
                    "path_count": len(reasoning_paths),
                    "chain_count": len(reasoning_chains),
                    "qa_count": len(reasoning_qa_pairs)
                }
            }
            
            logger.info(f"多跳推理数据生成完成")
            return results
            
        except Exception as e:
            logger.error(f"多跳推理数据生成失败: {e}")
            raise
    
    def _build_knowledge_graph(self, knowledge_graph: Dict[str, Any]) -> nx.DiGraph:
        """构建知识图谱网络"""
        graph = nx.DiGraph()
        
        # 创建实体ID到名称的映射
        entity_id_to_name = {}
        for entity in knowledge_graph.get("entities", []):
            entity_id_to_name[entity["id"]] = entity["name"]
            graph.add_node(entity["name"], 
                          type=entity["type"], 
                          description=entity["description"])
        
        # 添加关系边
        for relation in knowledge_graph.get("relations", []):
            # 将实体ID转换为实体名称
            source_name = entity_id_to_name.get(relation["source"], relation["source"])
            target_name = entity_id_to_name.get(relation["target"], relation["target"])
            
            graph.add_edge(source_name, 
                          target_name, 
                          type=relation["type"],
                          description=relation["description"])
        
        return graph
    
    def _discover_reasoning_paths(self, graph: nx.DiGraph, knowledge_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """发现推理路径"""
        paths = []
        
        # 获取所有实体
        entities = [e["name"] for e in knowledge_graph.get("entities", [])]
        
        # 寻找2跳和3跳的推理路径
        for source in entities:
            for target in entities:
                if source != target:
                    # 寻找2跳路径
                    paths_2hop = list(nx.all_simple_paths(graph, source, target, cutoff=2))
                    for path in paths_2hop:
                        if len(path) == 3:  # 2跳路径：A->B->C
                            path_info = self._extract_path_info(graph, path)
                            paths.append(path_info)
                    
                    # 寻找3跳路径
                    paths_3hop = list(nx.all_simple_paths(graph, source, target, cutoff=3))
                    for path in paths_3hop:
                        if len(path) == 4:  # 3跳路径：A->B->C->D
                            path_info = self._extract_path_info(graph, path)
                            paths.append(path_info)
        
        return paths
    
    def _extract_path_info(self, graph: nx.DiGraph, path: List[str]) -> Dict[str, Any]:
        """提取路径信息"""
        path_info = {
            "source": path[0],
            "target": path[-1],
            "path": path,
            "length": len(path) - 1,
            "edges": []
        }
        
        # 提取边信息
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i + 1])
            edge_info = {
                "source": path[i],  # 这里已经是实体名称，因为图是用实体名称构建的
                "target": path[i + 1],
                "type": edge_data.get("type", ""),
                "description": edge_data.get("description", "")
            }
            path_info["edges"].append(edge_info)
        
        return path_info
    
    async def _generate_reasoning_chains(self, reasoning_paths: List[Dict[str, Any]], 
                                       knowledge_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成推理链"""
        reasoning_chains = []
        
        for path_info in reasoning_paths:
            # 构建推理链提示
            prompt = self._build_reasoning_chain_prompt(path_info, knowledge_graph)
            
            # 调用LLM生成推理链
            response = await llm_client.generate(prompt)
            
            reasoning_chain = {
                "path_info": path_info,
                "reasoning_text": response.content,
                "type": f"{path_info['length']}跳推理"
            }
            
            reasoning_chains.append(reasoning_chain)
        
        return reasoning_chains
    
    def _build_reasoning_chain_prompt(self, path_info: Dict[str, Any], knowledge_graph: Dict[str, Any]) -> str:
        """构建推理链生成提示"""
        
        # 获取路径中的实体信息
        entities_info = {}
        for entity in knowledge_graph.get("entities", []):
            if entity["name"] in path_info["path"]:
                entities_info[entity["name"]] = entity
        
        # 构建实体间的联系描述
        entity_connections = []
        for i in range(len(path_info["path"]) - 1):
            current_entity = path_info["path"][i]
            next_entity = path_info["path"][i + 1]
            edge = path_info["edges"][i]
            
            current_desc = entities_info.get(current_entity, {}).get("description", "")
            next_desc = entities_info.get(next_entity, {}).get("description", "")
            
            connection = f"{current_entity}({current_desc})通过{edge['type']}关系连接到{next_entity}({next_desc})"
            entity_connections.append(connection)
        
        prompt = f"""基于故障诊断知识图谱，生成多跳推理链的自然语言描述。

推理路径：{' -> '.join(path_info['path'])}
路径长度：{path_info['length']}跳

路径中的实体信息：
"""
        
        for entity_name in path_info["path"]:
            if entity_name in entities_info:
                entity = entities_info[entity_name]
                prompt += f"- {entity_name} ({entity['type']}): {entity['description']}\n"
        
        prompt += f"""
实体间的联系：
"""
        
        for i, connection in enumerate(entity_connections):
            prompt += f"- 步骤{i+1}: {connection}\n"
        
        prompt += f"""
路径中的关系：
"""
        
        for edge in path_info["edges"]:
            prompt += f"- {edge['source']} --[{edge['type']}]--> {edge['target']}: {edge['description']}\n"
        
        prompt += f"""
请生成一个完整的推理链，说明如何从{path_info['source']}推理到{path_info['target']}：

要求：
1. 使用具体的实体名称，不要用数字或代号
2. 建立多个实体之间的逻辑联系
3. 用自然语言描述整个推理过程
4. 体现故障诊断的专业性和逻辑性

请生成自然语言描述："""
        
        return prompt
    
    def _generate_reasoning_qa_pairs(self, reasoning_chains: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """生成推理问答对"""
        qa_pairs = []
        
        for chain in reasoning_chains:
            path_info = chain["path_info"]
            reasoning_text = chain["reasoning_text"]
            
            # 生成推理问答对
            qa_pairs.append({
                "question": f"如何从{path_info['source']}推理到{path_info['target']}？",
                "answer": reasoning_text,
                "type": "multi_hop_reasoning",
                "path_length": path_info["length"]
            })
            
            qa_pairs.append({
                "question": f"{path_info['source']}和{path_info['target']}之间有什么间接关系？",
                "answer": reasoning_text,
                "type": "indirect_relation",
                "path_length": path_info["length"]
            })
            
            # 生成路径相关的问答对
            path_str = " -> ".join(path_info["path"])
            qa_pairs.append({
                "question": f"故障如何沿着路径 {path_str} 传播？",
                "answer": reasoning_text,
                "type": "fault_propagation",
                "path_length": path_info["length"]
            })
            
            # 生成实体间联系的问答对
            if len(path_info["path"]) > 2:
                middle_entities = path_info["path"][1:-1]
                qa_pairs.append({
                    "question": f"在{path_info['source']}到{path_info['target']}的推理过程中，{', '.join(middle_entities)}起到了什么作用？",
                    "answer": reasoning_text,
                    "type": "intermediate_entities",
                    "path_length": path_info["length"]
                })
            
            # 生成具体实体关系的问答对
            for i, edge in enumerate(path_info["edges"]):
                qa_pairs.append({
                    "question": f"{edge['source']}和{edge['target']}之间是什么关系？",
                    "answer": f"{edge['source']}通过{edge['type']}关系影响{edge['target']}，具体表现为{edge['description']}。",
                    "type": "direct_relation",
                    "path_length": 1
                })
        
        return qa_pairs
