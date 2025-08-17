"""
GraphGen Enhanced 完整执行流程
整合：数据生成 -> 去重 -> 质量检测
"""

import asyncio
import json
import os
import time
from typing import Dict, Any, List
from utils.logger import logger
from main import GraphGenEnhanced
from utils.minhash_deduplicator import process_pretrain_data_with_minhash
from huanjue.hallucination_detector import HallucinationDetector
from config import DATA_FILTER_CONFIG, FILE_SAVE_CONFIG

class CompletePipeline:
    """完整执行流程类"""
    
    def __init__(self):
        self.graphgen = GraphGenEnhanced()
        self.hallucination_detector = HallucinationDetector()
        
    async def run_complete_pipeline(self, fault_text: str, case_id: str = None) -> Dict[str, Any]:
        """运行完整流程：生成 -> 去重 -> 检测"""
        logger.info("开始执行完整流程：数据生成 -> 去重 -> 质量检测")
        
        pipeline_results = {
            "case_id": case_id,
            "original_text": fault_text,
            "pipeline_steps": {},
            "final_results": {}
        }
        
        try:
            # 步骤1: 数据生成
            logger.info("步骤1: 知识图谱提取和预训练数据生成")
            start_time = time.time()
            generation_result = await self.graphgen.process_fault_case(fault_text, case_id)
            generation_time = time.time() - start_time
            
            pipeline_results["pipeline_steps"]["generation"] = {
                "status": "success",
                "time": generation_time,
                "result": generation_result
            }
            logger.info(f"数据生成完成，耗时: {generation_time:.2f}秒")
            
            # 步骤2: 数据去重
            logger.info("步骤2: 预训练数据去重")
            start_time = time.time()
            deduplication_result = self._run_deduplication(generation_result, case_id)
            deduplication_time = time.time() - start_time
            
            pipeline_results["pipeline_steps"]["deduplication"] = {
                "status": "success",
                "time": deduplication_time,
                "result": deduplication_result
            }
            logger.info(f"数据去重完成，耗时: {deduplication_time:.2f}秒")
            
            # 步骤3: 按类别质量检测
            logger.info("步骤3: 按类别进行幻觉检测和质量评估")
            start_time = time.time()
            # 使用去重后的数据进行质量检测
            deduplicated_data = deduplication_result.get("processed_data", {}).get("deduplicated", {})
            quality_result = self._run_quality_detection_by_type(fault_text, generation_result, deduplicated_data)
            quality_time = time.time() - start_time
            
            pipeline_results["pipeline_steps"]["quality_detection"] = {
                "status": "success",
                "time": quality_time,
                "result": quality_result
            }
            logger.info(f"按类别质量检测完成，耗时: {quality_time:.2f}秒")
            
            # 步骤4: 按类别数据过滤
            logger.info("步骤4: 根据类别质量评估结果过滤数据")
            start_time = time.time()
            # 使用去重后的数据进行过滤
            filtered_result = self._filter_quality_data_by_type(generation_result, quality_result, deduplicated_data)
            filter_time = time.time() - start_time
            
            pipeline_results["pipeline_steps"]["data_filtering"] = {
                "status": "success",
                "time": filter_time,
                "result": {
                    "passed_quality_check": quality_result.get("passed_quality_check", False),
                    "filtered_data": filtered_result
                }
            }
            logger.info(f"数据过滤完成，耗时: {filter_time:.2f}秒")
            
            # 整合最终结果
            pipeline_results["final_results"] = {
                "total_time": generation_time + deduplication_time + quality_time + filter_time,
                "generation_stats": {
                    "entities": filtered_result["metadata"]["total_entities"],
                    "relations": filtered_result["metadata"]["total_relations"],
                    "entity_types": filtered_result["metadata"]["entity_types"],
                    "relation_types": filtered_result["metadata"]["relation_types"]
                },
                "deduplication_stats": deduplication_result.get("stats", {}),
                "quality_assessment": {
                    "overall_score": quality_result.get("quality_results", {}).get("overall", {}).get("quality_metrics", {}).get("overall_score", 0),
                    "overall_passed": quality_result.get("overall_passed", False),
                    "category_results": quality_result.get("quality_results", {})
                },
                "filter_stats": filtered_result.get("filter_stats", {}),
                "quality_filtered": filtered_result.get("quality_filtered", False)
            }
            
            # 保存关键结果（融合所有重要信息）
            self._save_key_results(pipeline_results, filtered_result, quality_result, case_id)
            
            logger.info("完整流程执行成功！")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"完整流程执行失败: {e}")
            raise
    
    def _run_deduplication(self, generation_result: Dict[str, Any], case_id: str = None) -> Dict[str, Any]:
        """执行数据去重"""
        try:
            # 为每个案例创建唯一的输出目录
            if case_id:
                output_dir = f"pretrain_data_deduplicated_{case_id}"
            else:
                output_dir = "pretrain_data_deduplicated"
            
            similarity_threshold = 0.5
            
            # 从生成结果中获取预训练数据
            pretrain_knowledge = generation_result.get("pretrain_knowledge", {})
            
            # 准备去重数据
            deduplication_data = {
                "training_samples": pretrain_knowledge.get("training_samples", []),
                "qa_pairs": pretrain_knowledge.get("qa_pairs", []),
                "reasoning_chains": pretrain_knowledge.get("reasoning_chains", []),
                "reasoning_paths": pretrain_knowledge.get("reasoning_paths", [])
            }
            
            # 创建临时文件用于去重处理
            temp_file = f"temp_{case_id}_deduplication.json" if case_id else "temp_deduplication.json"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(deduplication_data, f, ensure_ascii=False, indent=2)
            
            processed_data = process_pretrain_data_with_minhash(
                input_file=temp_file,
                output_dir=output_dir,
                similarity_threshold=similarity_threshold
            )
            
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            # 统计去重结果
            stats = {}
            for data_type, items in processed_data["deduplicated"].items():
                original_count = len(processed_data["extracted"].get(data_type, []))
                deduplicated_count = len(items)
                removed_count = original_count - deduplicated_count
                removal_rate = (removed_count / original_count * 100) if original_count > 0 else 0
                
                stats[data_type] = {
                    "original": original_count,
                    "deduplicated": deduplicated_count,
                    "removed": removed_count,
                    "removal_rate": removal_rate
                }
            
            return {
                "status": "success",
                "output_dir": output_dir,
                "stats": stats,
                "processed_data": processed_data
            }
            
        except Exception as e:
            logger.error(f"去重处理失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def _run_quality_detection_by_type(self, original_text: str, generation_result: Dict[str, Any], deduplicated_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """按类别执行质量检测"""
        try:
            quality_results = {}
            
            # 获取预训练数据（优先使用去重后的数据）
            if deduplicated_data:
                pretrain_knowledge = deduplicated_data
            else:
                pretrain_knowledge = generation_result.get("pretrain_knowledge", {})
            
            # 1. 对整体预训练文本进行质量检测
            pretrain_text = generation_result.get("pretrain_knowledge", {}).get("pretrain_text", "")
            overall_quality = self._assess_overall_quality(original_text, pretrain_text)
            quality_results["overall"] = overall_quality
            
            # 2. 按数据类型进行质量检测
            data_types = {
                "entity_descriptions": pretrain_knowledge.get("training_samples", []),
                "relation_descriptions": pretrain_knowledge.get("training_samples", []),
                "qa_pairs": pretrain_knowledge.get("qa_pairs", []),
                "reasoning_chains": pretrain_knowledge.get("reasoning_chains", []),
                "reasoning_paths": pretrain_knowledge.get("reasoning_paths", [])
            }
            
            for data_type, items in data_types.items():
                if items:
                    type_quality = self._assess_data_type_quality(original_text, items, data_type)
                    quality_results[data_type] = type_quality
                    confidence_level = type_quality['quality_metrics'].get('confidence_level', 'unknown')
                    logger.info(f"{data_type} 质量评估: 评分={type_quality['quality_metrics']['overall_score']:.2f}, 置信度={confidence_level}, 通过={type_quality['passed_quality_check']}")
            
            return {
                "status": "success",
                "quality_results": quality_results,
                "overall_passed": overall_quality.get("passed_quality_check", False)
            }
            
        except Exception as e:
            logger.error(f"按类别质量检测失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def _assess_overall_quality(self, original_text: str, pretrain_text: str) -> Dict[str, Any]:
        """评估整体质量"""
        try:
            detection_result = self.hallucination_detector.detect_hallucination(
                original_text, pretrain_text
            )
            
            cleaned_detection_result = self._clean_for_json(detection_result)
            
            # 获取幻觉检测的置信度级别
            hallucination_status = cleaned_detection_result.get("hallucination_status", "")
            confidence_level = cleaned_detection_result.get("category", "")
            
            quality_metrics = {
                "overall_score": cleaned_detection_result.get("overall_score", 0),
                "hallucination_detected": not hallucination_status.startswith("no_hallucination"),
                "consistency_score": cleaned_detection_result.get("detailed_scores", {}).get("factual_consistency", 0),
                "completeness_score": cleaned_detection_result.get("detailed_scores", {}).get("reasoning_quality", 0),
                "accuracy_score": cleaned_detection_result.get("detailed_scores", {}).get("fundamental_errors", 0),
                "confidence_level": confidence_level,
                "hallucination_status": hallucination_status
            }
            
            # 只要达到 high_confidence 就认为质量良好
            passed_quality_check = confidence_level == "high_confidence"
            quality_metrics["passed_quality_check"] = passed_quality_check
            
            logger.info(f"整体质量评估: 置信度级别={confidence_level}, 幻觉状态={hallucination_status}, 通过={'是' if passed_quality_check else '否'}")
            
            return {
                "detection_result": cleaned_detection_result,
                "quality_metrics": quality_metrics,
                "passed_quality_check": passed_quality_check
            }
            
        except Exception as e:
            logger.error(f"整体质量评估失败: {e}")
            return {
                "quality_metrics": {"overall_score": 0, "passed_quality_check": False, "confidence_level": "unknown"},
                "passed_quality_check": False
            }
    
    def _assess_data_type_quality(self, original_text: str, items: List[Dict], data_type: str) -> Dict[str, Any]:
        """评估特定数据类型的质量"""
        try:
            # 根据数据类型设置不同的质量评估策略
            if data_type == "entity_descriptions":
                return self._assess_entity_quality(original_text, items)
            elif data_type == "relation_descriptions":
                return self._assess_relation_quality(original_text, items)
            elif data_type == "qa_pairs":
                return self._assess_qa_quality(original_text, items)
            elif data_type in ["reasoning_chains", "reasoning_paths"]:
                return self._assess_reasoning_quality(original_text, items)
            else:
                return self._assess_generic_quality(original_text, items, data_type)
                
        except Exception as e:
            logger.error(f"{data_type} 质量评估失败: {e}")
            return {
                "quality_metrics": {"overall_score": 0, "passed_quality_check": False},
                "passed_quality_check": False
            }
    
    def _assess_entity_quality(self, original_text: str, items: List[Dict]) -> Dict[str, Any]:
        """评估实体描述质量"""
        # 实体描述质量评估：重点检查实体定义的准确性和完整性
        entity_text = "\n".join([item.get("content", "") for item in items if item.get("type") == "entity_description"])
        
        if not entity_text:
            return {"quality_metrics": {"overall_score": 0, "passed_quality_check": False, "confidence_level": "unknown"}, "passed_quality_check": False}
        
        # 使用幻觉检测器进行质量评估
        try:
            detection_result = self.hallucination_detector.detect_hallucination(
                original_text, entity_text
            )
            
            cleaned_detection_result = self._clean_for_json(detection_result)
            confidence_level = cleaned_detection_result.get("category", "")
            
            quality_metrics = {
                "overall_score": cleaned_detection_result.get("overall_score", 0),
                "entity_accuracy": cleaned_detection_result.get("detailed_scores", {}).get("factual_consistency", 0),
                "entity_completeness": cleaned_detection_result.get("detailed_scores", {}).get("reasoning_quality", 0),
                "confidence_level": confidence_level,
                "passed_quality_check": confidence_level == "high_confidence"
            }
            
            return {
                "quality_metrics": quality_metrics,
                "passed_quality_check": quality_metrics["passed_quality_check"]
            }
            
        except Exception as e:
            logger.error(f"实体描述质量评估失败: {e}")
            return {
                "quality_metrics": {"overall_score": 0, "passed_quality_check": False, "confidence_level": "unknown"},
                "passed_quality_check": False
            }
    
    def _assess_relation_quality(self, original_text: str, items: List[Dict]) -> Dict[str, Any]:
        """评估关系描述质量"""
        # 关系描述质量评估：重点检查逻辑关系的正确性
        relation_text = "\n".join([item.get("content", "") for item in items if item.get("type") == "relation_description"])
        
        if not relation_text:
            return {"quality_metrics": {"overall_score": 0, "passed_quality_check": False, "confidence_level": "unknown"}, "passed_quality_check": False}
        
        # 使用幻觉检测器进行质量评估
        try:
            detection_result = self.hallucination_detector.detect_hallucination(
                original_text, relation_text
            )
            
            cleaned_detection_result = self._clean_for_json(detection_result)
            confidence_level = cleaned_detection_result.get("category", "")
            
            quality_metrics = {
                "overall_score": cleaned_detection_result.get("overall_score", 0),
                "relation_accuracy": cleaned_detection_result.get("detailed_scores", {}).get("factual_consistency", 0),
                "logic_consistency": cleaned_detection_result.get("detailed_scores", {}).get("reasoning_quality", 0),
                "confidence_level": confidence_level,
                "passed_quality_check": confidence_level == "high_confidence"
            }
            
            return {
                "quality_metrics": quality_metrics,
                "passed_quality_check": quality_metrics["passed_quality_check"]
            }
            
        except Exception as e:
            logger.error(f"关系描述质量评估失败: {e}")
            return {
                "quality_metrics": {"overall_score": 0, "passed_quality_check": False, "confidence_level": "unknown"},
                "passed_quality_check": False
            }
    
    def _assess_qa_quality(self, original_text: str, items: List[Dict]) -> Dict[str, Any]:
        """评估问答对质量"""
        # 问答对质量评估：重点检查问答的匹配度和准确性
        qa_text = "\n".join([f"Q: {item.get('question', '')} A: {item.get('answer', '')}" for item in items])
        
        if not qa_text:
            return {"quality_metrics": {"overall_score": 0, "passed_quality_check": False, "confidence_level": "unknown"}, "passed_quality_check": False}
        
        # 使用幻觉检测器进行质量评估
        try:
            detection_result = self.hallucination_detector.detect_hallucination(
                original_text, qa_text
            )
            
            cleaned_detection_result = self._clean_for_json(detection_result)
            confidence_level = cleaned_detection_result.get("category", "")
            
            quality_metrics = {
                "overall_score": cleaned_detection_result.get("overall_score", 0),
                "qa_accuracy": cleaned_detection_result.get("detailed_scores", {}).get("factual_consistency", 0),
                "qa_relevance": cleaned_detection_result.get("detailed_scores", {}).get("reasoning_quality", 0),
                "confidence_level": confidence_level,
                "passed_quality_check": confidence_level == "high_confidence"
            }
            
            return {
                "quality_metrics": quality_metrics,
                "passed_quality_check": quality_metrics["passed_quality_check"]
            }
            
        except Exception as e:
            logger.error(f"问答对质量评估失败: {e}")
            return {
                "quality_metrics": {"overall_score": 0, "passed_quality_check": False, "confidence_level": "unknown"},
                "passed_quality_check": False
            }
    
    def _assess_reasoning_quality(self, original_text: str, items: List[Dict]) -> Dict[str, Any]:
        """评估推理链质量"""
        # 推理链质量评估：重点检查推理逻辑的合理性
        reasoning_text = "\n".join([item.get("content", "") for item in items])
        
        if not reasoning_text:
            return {"quality_metrics": {"overall_score": 0, "passed_quality_check": False, "confidence_level": "unknown"}, "passed_quality_check": False}
        
        # 使用幻觉检测器进行质量评估
        try:
            detection_result = self.hallucination_detector.detect_hallucination(
                original_text, reasoning_text
            )
            
            cleaned_detection_result = self._clean_for_json(detection_result)
            confidence_level = cleaned_detection_result.get("category", "")
            
            quality_metrics = {
                "overall_score": cleaned_detection_result.get("overall_score", 0),
                "reasoning_logic": cleaned_detection_result.get("detailed_scores", {}).get("factual_consistency", 0),
                "reasoning_completeness": cleaned_detection_result.get("detailed_scores", {}).get("reasoning_quality", 0),
                "confidence_level": confidence_level,
                "passed_quality_check": confidence_level == "high_confidence"
            }
            
            return {
                "quality_metrics": quality_metrics,
                "passed_quality_check": quality_metrics["passed_quality_check"]
            }
            
        except Exception as e:
            logger.error(f"推理链质量评估失败: {e}")
            return {
                "quality_metrics": {"overall_score": 0, "passed_quality_check": False, "confidence_level": "unknown"},
                "passed_quality_check": False
            }
    
    def _assess_generic_quality(self, original_text: str, items: List[Dict], data_type: str) -> Dict[str, Any]:
        """通用质量评估"""
        content_text = "\n".join([str(item) for item in items])
        
        if not content_text:
            return {"quality_metrics": {"overall_score": 0, "passed_quality_check": False, "confidence_level": "unknown"}, "passed_quality_check": False}
        
        # 使用幻觉检测器进行质量评估
        try:
            detection_result = self.hallucination_detector.detect_hallucination(
                original_text, content_text
            )
            
            cleaned_detection_result = self._clean_for_json(detection_result)
            confidence_level = cleaned_detection_result.get("category", "")
            
            quality_metrics = {
                "overall_score": cleaned_detection_result.get("overall_score", 0),
                "content_quality": cleaned_detection_result.get("detailed_scores", {}).get("factual_consistency", 0),
                "confidence_level": confidence_level,
                "passed_quality_check": confidence_level == "high_confidence"
            }
            
            return {
                "quality_metrics": quality_metrics,
                "passed_quality_check": quality_metrics["passed_quality_check"]
            }
            
        except Exception as e:
            logger.error(f"通用质量评估失败: {e}")
            return {
                "quality_metrics": {"overall_score": 0, "passed_quality_check": False, "confidence_level": "unknown"},
                "passed_quality_check": False
            }
    
    def _check_quality_threshold(self, quality_metrics: Dict[str, Any]) -> bool:
        """检查是否通过质量阈值"""
        # 从配置文件获取质量评估标准
        thresholds = DATA_FILTER_CONFIG["quality_thresholds"]
        min_overall_score = thresholds["min_overall_score"]
        min_consistency_score = thresholds["min_consistency_score"]
        allow_hallucination = thresholds["allow_hallucination"]
        
        overall_score = quality_metrics.get("overall_score", 0)
        consistency_score = quality_metrics.get("consistency_score", 0)
        hallucination_detected = quality_metrics.get("hallucination_detected", True)
        
        # 检查是否通过所有阈值
        passed = (
            overall_score >= min_overall_score and
            consistency_score >= min_consistency_score and
            (allow_hallucination or not hallucination_detected)
        )
        
        logger.info(f"质量评估结果: 总体评分={overall_score:.2f}, 一致性评分={consistency_score:.2f}, 幻觉检测={hallucination_detected}")
        logger.info(f"质量阈值: 总体评分>={min_overall_score}, 一致性评分>={min_consistency_score}, 允许幻觉={allow_hallucination}")
        logger.info(f"质量评估: {'通过' if passed else '未通过'}")
        
        return passed
    
    def _filter_quality_data_by_type(self, generation_result: Dict[str, Any], quality_result: Dict[str, Any], deduplicated_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """根据类别质量评估结果进行按条过滤"""
        quality_results = quality_result.get("quality_results", {})
        
        # 复制原始结果
        filtered_result = generation_result.copy()
        
        # 使用去重后的数据或原始数据
        if deduplicated_data:
            pretrain_knowledge = deduplicated_data
        else:
            pretrain_knowledge = filtered_result.get("pretrain_knowledge", {})
        
        # 按类别进行过滤
        filtered_stats = {
            "total_filtered": 0,
            "passed_categories": [],
            "filtered_categories": [],
            "item_level_stats": {
                "entity_descriptions": {"original": 0, "passed": 0, "filtered": 0},
                "relation_descriptions": {"original": 0, "passed": 0, "filtered": 0},
                "qa_pairs": {"original": 0, "passed": 0, "filtered": 0},
                "reasoning_chains": {"original": 0, "passed": 0, "filtered": 0},
                "reasoning_paths": {"original": 0, "passed": 0, "filtered": 0}
            }
        }
        
        # 1. 按条过滤实体描述
        if "entity_descriptions" in quality_results:
            entity_quality = quality_results["entity_descriptions"]
            if entity_quality.get("passed_quality_check", False):
                # 类别通过，进行按条过滤
                original_entities = [item for item in pretrain_knowledge.get("training_samples", []) if item.get("type") == "entity_description"]
                filtered_entities = self._filter_items_by_quality(original_entities, "entity_description", generation_result.get("original_text", ""))
                
                # 更新统计数据
                filtered_stats["item_level_stats"]["entity_descriptions"]["original"] = len(original_entities)
                filtered_stats["item_level_stats"]["entity_descriptions"]["passed"] = len(filtered_entities)
                filtered_stats["item_level_stats"]["entity_descriptions"]["filtered"] = len(original_entities) - len(filtered_entities)
                
                # 更新数据
                pretrain_knowledge["training_samples"] = [
                    item for item in pretrain_knowledge.get("training_samples", [])
                    if item.get("type") != "entity_description"
                ] + filtered_entities
                
                filtered_stats["passed_categories"].append("entity_descriptions")
                logger.info(f"实体描述按条过滤: {len(original_entities)} -> {len(filtered_entities)} (过滤{len(original_entities) - len(filtered_entities)}条)")
            else:
                # 类别未通过，全部过滤
                pretrain_knowledge["training_samples"] = [
                    item for item in pretrain_knowledge.get("training_samples", [])
                    if item.get("type") != "entity_description"
                ]
                filtered_stats["filtered_categories"].append("entity_descriptions")
                filtered_stats["total_filtered"] += 1
                logger.warning("实体描述类别未通过质量评估，全部过滤")
        
        # 2. 按条过滤关系描述
        if "relation_descriptions" in quality_results:
            relation_quality = quality_results["relation_descriptions"]
            if relation_quality.get("passed_quality_check", False):
                # 类别通过，进行按条过滤
                original_relations = [item for item in pretrain_knowledge.get("training_samples", []) if item.get("type") == "relation_description"]
                filtered_relations = self._filter_items_by_quality(original_relations, "relation_description", generation_result.get("original_text", ""))
                
                # 更新统计数据
                filtered_stats["item_level_stats"]["relation_descriptions"]["original"] = len(original_relations)
                filtered_stats["item_level_stats"]["relation_descriptions"]["passed"] = len(filtered_relations)
                filtered_stats["item_level_stats"]["relation_descriptions"]["filtered"] = len(original_relations) - len(filtered_relations)
                
                # 更新数据
                pretrain_knowledge["training_samples"] = [
                    item for item in pretrain_knowledge.get("training_samples", [])
                    if item.get("type") != "relation_description"
                ] + filtered_relations
                
                filtered_stats["passed_categories"].append("relation_descriptions")
                logger.info(f"关系描述按条过滤: {len(original_relations)} -> {len(filtered_relations)} (过滤{len(original_relations) - len(filtered_relations)}条)")
            else:
                # 类别未通过，全部过滤
                pretrain_knowledge["training_samples"] = [
                    item for item in pretrain_knowledge.get("training_samples", [])
                    if item.get("type") != "relation_description"
                ]
                filtered_stats["filtered_categories"].append("relation_descriptions")
                filtered_stats["total_filtered"] += 1
                logger.warning("关系描述类别未通过质量评估，全部过滤")
        
        # 3. 按条过滤问答对
        if "qa_pairs" in quality_results:
            qa_quality = quality_results["qa_pairs"]
            if qa_quality.get("passed_quality_check", False):
                # 类别通过，进行按条过滤
                original_qa_pairs = pretrain_knowledge.get("qa_pairs", [])
                filtered_qa_pairs = self._filter_items_by_quality(original_qa_pairs, "qa_pairs", generation_result.get("original_text", ""))
                
                # 更新统计数据
                filtered_stats["item_level_stats"]["qa_pairs"]["original"] = len(original_qa_pairs)
                filtered_stats["item_level_stats"]["qa_pairs"]["passed"] = len(filtered_qa_pairs)
                filtered_stats["item_level_stats"]["qa_pairs"]["filtered"] = len(original_qa_pairs) - len(filtered_qa_pairs)
                
                # 更新数据
                pretrain_knowledge["qa_pairs"] = filtered_qa_pairs
                
                filtered_stats["passed_categories"].append("qa_pairs")
                logger.info(f"问答对按条过滤: {len(original_qa_pairs)} -> {len(filtered_qa_pairs)} (过滤{len(original_qa_pairs) - len(filtered_qa_pairs)}条)")
            else:
                # 类别未通过，全部过滤
                pretrain_knowledge["qa_pairs"] = []
                filtered_stats["filtered_categories"].append("qa_pairs")
                filtered_stats["total_filtered"] += 1
                logger.warning("问答对类别未通过质量评估，全部过滤")
        
        # 4. 按条过滤推理链
        if "reasoning_chains" in quality_results:
            reasoning_quality = quality_results["reasoning_chains"]
            if reasoning_quality.get("passed_quality_check", False):
                # 类别通过，进行按条过滤
                original_chains = pretrain_knowledge.get("reasoning_chains", [])
                filtered_chains = self._filter_items_by_quality(original_chains, "reasoning_chains", generation_result.get("original_text", ""))
                
                # 更新统计数据
                filtered_stats["item_level_stats"]["reasoning_chains"]["original"] = len(original_chains)
                filtered_stats["item_level_stats"]["reasoning_chains"]["passed"] = len(filtered_chains)
                filtered_stats["item_level_stats"]["reasoning_chains"]["filtered"] = len(original_chains) - len(filtered_chains)
                
                # 更新数据
                pretrain_knowledge["reasoning_chains"] = filtered_chains
                
                filtered_stats["passed_categories"].append("reasoning_chains")
                logger.info(f"推理链按条过滤: {len(original_chains)} -> {len(filtered_chains)} (过滤{len(original_chains) - len(filtered_chains)}条)")
            else:
                # 类别未通过，全部过滤
                pretrain_knowledge["reasoning_chains"] = []
                filtered_stats["filtered_categories"].append("reasoning_chains")
                filtered_stats["total_filtered"] += 1
                logger.warning("推理链类别未通过质量评估，全部过滤")
        
        # 5. 按条过滤推理路径
        if "reasoning_paths" in quality_results:
            paths_quality = quality_results["reasoning_paths"]
            if paths_quality.get("passed_quality_check", False):
                # 类别通过，进行按条过滤
                original_paths = pretrain_knowledge.get("reasoning_paths", [])
                filtered_paths = self._filter_items_by_quality(original_paths, "reasoning_paths", generation_result.get("original_text", ""))
                
                # 更新统计数据
                filtered_stats["item_level_stats"]["reasoning_paths"]["original"] = len(original_paths)
                filtered_stats["item_level_stats"]["reasoning_paths"]["passed"] = len(filtered_paths)
                filtered_stats["item_level_stats"]["reasoning_paths"]["filtered"] = len(original_paths) - len(filtered_paths)
                
                # 更新数据
                pretrain_knowledge["reasoning_paths"] = filtered_paths
                
                filtered_stats["passed_categories"].append("reasoning_paths")
                logger.info(f"推理路径按条过滤: {len(original_paths)} -> {len(filtered_paths)} (过滤{len(original_paths) - len(filtered_paths)}条)")
            else:
                # 类别未通过，全部过滤
                pretrain_knowledge["reasoning_paths"] = []
                filtered_stats["filtered_categories"].append("reasoning_paths")
                filtered_stats["total_filtered"] += 1
                logger.warning("推理路径类别未通过质量评估，全部过滤")
        
        # 更新过滤后的预训练数据
        filtered_result["pretrain_knowledge"] = pretrain_knowledge
        
        # 更新元数据
        self._update_metadata(filtered_result)
        
        # 添加过滤统计信息
        filtered_result["quality_filtered"] = filtered_stats["total_filtered"] > 0
        filtered_result["filter_stats"] = filtered_stats
        
        if filtered_stats["total_filtered"] > 0:
            logger.info(f"按类别过滤完成: 通过{len(filtered_stats['passed_categories'])}个类别, 过滤{len(filtered_stats['filtered_categories'])}个类别")
        else:
            logger.info("所有类别都通过质量评估，进行按条过滤")
        
        return filtered_result
    
    def _filter_items_by_quality(self, items: List[Dict], data_type: str, original_text: str) -> List[Dict]:
        """对数据项进行按条质量过滤"""
        filtered_items = []
        
        for i, item in enumerate(items):
            try:
                # 根据数据类型构建评估文本
                if data_type == "entity_description":
                    item_text = item.get("content", "")
                elif data_type == "relation_description":
                    item_text = item.get("content", "")
                elif data_type == "qa_pairs":
                    item_text = f"Q: {item.get('question', '')} A: {item.get('answer', '')}"
                elif data_type in ["reasoning_chains", "reasoning_paths"]:
                    item_text = item.get("content", "")
                else:
                    item_text = str(item)
                
                if not item_text.strip():
                    logger.warning(f"{data_type} 第{i+1}条数据为空，跳过")
                    continue
                
                # 对单条数据进行质量评估
                detection_result = self.hallucination_detector.detect_hallucination(
                    original_text, item_text
                )
                
                cleaned_detection_result = self._clean_for_json(detection_result)
                confidence_level = cleaned_detection_result.get("category", "")
                
                # 只有达到 high_confidence 的数据才保留
                if confidence_level == "high_confidence":
                    filtered_items.append(item)
                    logger.debug(f"{data_type} 第{i+1}条数据通过质量评估 (置信度: {confidence_level})")
                else:
                    logger.debug(f"{data_type} 第{i+1}条数据未通过质量评估 (置信度: {confidence_level})")
                    
            except Exception as e:
                logger.error(f"{data_type} 第{i+1}条数据质量评估失败: {e}")
                # 评估失败的数据不保留
                continue
        
        logger.info(f"{data_type} 按条过滤完成: {len(items)} -> {len(filtered_items)} (保留率: {len(filtered_items)/len(items)*100:.1f}%)")
        return filtered_items
    
    def _update_metadata(self, filtered_result: Dict[str, Any]):
        """更新过滤后的元数据"""
        pretrain_knowledge = filtered_result.get("pretrain_knowledge", {})
        
        # 重新计算统计信息
        training_samples = pretrain_knowledge.get("training_samples", [])
        qa_pairs = pretrain_knowledge.get("qa_pairs", [])
        reasoning_chains = pretrain_knowledge.get("reasoning_chains", [])
        reasoning_paths = pretrain_knowledge.get("reasoning_paths", [])
        
        # 更新元数据
        metadata = pretrain_knowledge.get("metadata", {})
        metadata.update({
            "sample_count": len(training_samples),
            "qa_count": len(qa_pairs),
            "reasoning_chain_count": len(reasoning_chains),
            "reasoning_path_count": len(reasoning_paths),
            "text_length": len(pretrain_knowledge.get("pretrain_text", ""))
        })
        
        pretrain_knowledge["metadata"] = metadata
        filtered_result["pretrain_knowledge"] = pretrain_knowledge
    
    def _clean_for_json(self, obj):
        """清理对象，确保可以序列化为JSON"""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # 对于其他类型，转换为字符串
            return str(obj)
    
    def _save_pipeline_results(self, results: Dict[str, Any], case_id: str):
        """保存完整流程结果"""
        if case_id is None:
            case_id = "unknown"
        
        # 清理结果对象，确保可以序列化
        cleaned_results = self._clean_for_json(results)
        
        output_file = f"data/{case_id}_pipeline_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"完整流程结果已保存: {output_file}")
    
    def _save_key_results(self, pipeline_results: Dict[str, Any], filtered_result: Dict[str, Any], 
                         quality_result: Dict[str, Any], case_id: str):
        """保存关键结果（融合所有重要信息）"""
        if case_id is None:
            case_id = "unknown"
        
        # 构建关键结果文件
        key_results = {
            "case_id": case_id,
            "original_text": pipeline_results.get("original_text", ""),
            "processing_summary": {
                "total_time": pipeline_results["final_results"]["total_time"],
                "generation_time": pipeline_results["pipeline_steps"]["generation"]["time"],
                "deduplication_time": pipeline_results["pipeline_steps"]["deduplication"]["time"],
                "quality_detection_time": pipeline_results["pipeline_steps"]["quality_detection"]["time"],
                "filter_time": pipeline_results["pipeline_steps"]["data_filtering"]["time"]
            },
            "quality_assessment": {
                "overall_score": quality_result.get("quality_results", {}).get("overall", {}).get("quality_metrics", {}).get("overall_score", 0),
                "overall_passed": quality_result.get("overall_passed", False),
                "category_results": quality_result.get("quality_results", {}),
                "quality_filtered": filtered_result.get("quality_filtered", False)
            },
            "generation_stats": {
                "entities": filtered_result["metadata"]["total_entities"],
                "relations": filtered_result["metadata"]["total_relations"],
                "entity_types": filtered_result["metadata"]["entity_types"],
                "relation_types": filtered_result["metadata"]["relation_types"]
            },
            "deduplication_stats": pipeline_results["final_results"]["deduplication_stats"],
            "filter_stats": filtered_result.get("filter_stats", {}),
            "final_data": {
                "knowledge_graph": filtered_result["knowledge_graph"],
                "pretrain_knowledge": filtered_result["pretrain_knowledge"]
            }
        }
        
        # 保存关键结果文件
        output_file = f"data/{case_id}_key_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(key_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"关键结果已保存: {output_file}")
        
        # 根据配置决定是否保存高质量数据文件
        if FILE_SAVE_CONFIG["save_high_quality_only"] and quality_result.get("overall_passed", False):
            high_quality_file = f"data/{case_id}_high_quality_data.json"
            high_quality_data = {
                "case_id": case_id,
                "quality_score": quality_result.get("quality_results", {}).get("overall", {}).get("quality_metrics", {}).get("overall_score", 0),
                "confidence_level": quality_result.get("quality_results", {}).get("overall", {}).get("quality_metrics", {}).get("confidence_level", "unknown"),
                "knowledge_graph": filtered_result["knowledge_graph"],
                "pretrain_knowledge": filtered_result["pretrain_knowledge"]
            }
            with open(high_quality_file, 'w', encoding='utf-8') as f:
                json.dump(high_quality_data, f, ensure_ascii=False, indent=2)
            logger.info(f"高质量数据已保存: {high_quality_file}")
        elif not quality_result.get("overall_passed", False):
            logger.warning("数据未达到high_confidence级别，未保存高质量数据文件")

async def main():
    """主函数 - 演示完整流程"""
    
    # 简化的测试案例
    test_case = {
        "id": "TEST_CASE_001",
        "text": """某局升级后忙音播放有问题，SPD单板上加载的忙音有问题。
        忙音是异步音，该问题一般是异步音有问题造成的，可以先查看一下SPD单板上是否已经加载该语音，
        如果没有加载，需要按照正确流程加载该语音，如果已经加载，需要重新加载该语音。
        此局点的问题是升级前正常，升级后问题出现。"""
    }
    
    # 创建完整流程实例
    pipeline = CompletePipeline()
    
    # 执行完整流程
    print("开始执行完整流程...")
    print(f"处理案例: {test_case['id']}")
    print(f"原始文本长度: {len(test_case['text'])} 字符")
    
    try:
        result = await pipeline.run_complete_pipeline(test_case["text"], test_case["id"])
        
        # 打印结果摘要
        print("\n完整流程执行结果摘要:")
        print("="*50)
        
        final_results = result["final_results"]
        print(f"总耗时: {final_results['total_time']:.2f}秒")
        
        # 生成统计
        gen_stats = final_results["generation_stats"]
        print(f"\n知识生成统计:")
        print(f"  实体数量: {gen_stats['entities']}")
        print(f"  关系数量: {gen_stats['relations']}")
        print(f"  实体类型: {', '.join(gen_stats['entity_types'])}")
        print(f"  关系类型: {', '.join(gen_stats['relation_types'])}")
        
        # 去重统计
        dedup_stats = final_results["deduplication_stats"]
        print(f"\n去重统计:")
        for data_type, stats in dedup_stats.items():
            print(f"  {data_type}: {stats['original']} -> {stats['deduplicated']} (移除 {stats['removed']} 条, {stats['removal_rate']:.1f}%)")
        
        # 质量检测
        quality_assessment = final_results.get("quality_assessment", {})
        filter_stats = final_results.get("filter_stats", {})
        
        print(f"\n质量检测:")
        print(f"  总体评分: {quality_assessment.get('overall_score', 0):.2f}")
        overall_confidence = quality_assessment.get("category_results", {}).get("overall", {}).get("quality_metrics", {}).get("confidence_level", "unknown")
        print(f"  整体置信度: {overall_confidence}")
        print(f"  整体通过质量评估: {'是' if quality_assessment.get('overall_passed', False) else '否'}")
        print(f"  数据被过滤: {'是' if final_results['quality_filtered'] else '否'}")
        
        # 显示各类别质量评估结果
        category_results = quality_assessment.get("category_results", {})
        if category_results:
            print(f"\n各类别质量评估:")
            for category, result in category_results.items():
                if category != "overall":
                    score = result.get("quality_metrics", {}).get("overall_score", 0)
                    confidence = result.get("quality_metrics", {}).get("confidence_level", "unknown")
                    passed = result.get("passed_quality_check", False)
                    print(f"  {category}: 评分={score:.2f}, 置信度={confidence}, 通过={'是' if passed else '否'}")
        
        # 显示过滤统计
        if filter_stats:
            print(f"\n过滤统计:")
            print(f"  通过类别数: {len(filter_stats.get('passed_categories', []))}")
            print(f"  过滤类别数: {len(filter_stats.get('filtered_categories', []))}")
            if filter_stats.get('filtered_categories'):
                print(f"  被过滤类别: {', '.join(filter_stats['filtered_categories'])}")
            
            # 显示按条过滤统计
            item_level_stats = filter_stats.get('item_level_stats', {})
            if item_level_stats:
                print(f"\n按条过滤统计:")
                for category, stats in item_level_stats.items():
                    if stats['original'] > 0:
                        retention_rate = stats['passed'] / stats['original'] * 100
                        print(f"  {category}: {stats['original']} -> {stats['passed']} (过滤{stats['filtered']}条, 保留率{retention_rate:.1f}%)")
        
        print(f"\n结果文件:")
        print(f"  关键结果: data/{test_case['id']}_key_results.json")
        if quality_assessment.get('overall_passed', False):
            print(f"  高质量数据: data/{test_case['id']}_high_quality_data.json")
        print(f"  去重数据: pretrain_data_deduplicated/")
        
        print("\n完整流程执行成功！")
        
    except Exception as e:
        print(f"流程执行失败: {e}")
        logger.error(f"流程执行失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
