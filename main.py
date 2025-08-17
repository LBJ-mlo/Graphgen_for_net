"""
GraphGen Enhanced 主程序入口
"""

import asyncio
import json
import os
from typing import Dict, Any, List
from utils.logger import logger
from core.knowledge_extractor import KnowledgeExtractor
from core.pretrain_knowledge_generator import PretrainKnowledgeGenerator

from config import DATA_DIR, CACHE_DIR

class GraphGenEnhanced:
    """GraphGen Enhanced 主类"""
    
    def __init__(self):
        """初始化"""
        self.knowledge_extractor = KnowledgeExtractor()
        self.pretrain_generator = PretrainKnowledgeGenerator()
        
    async def process_fault_case(self, fault_text: str, case_id: str = None) -> Dict[str, Any]:
        """处理故障案例"""
        logger.info(f"开始处理故障案例: {case_id or 'unknown'}")
        
        try:
            # 1. 知识图谱提取
            logger.info("步骤1: 知识图谱提取")
            knowledge_graph = await self.knowledge_extractor.extract_knowledge(fault_text)
            
            # 2. 预训练知识生成
            logger.info("步骤2: 预训练知识生成")
            pretrain_knowledge = await self.pretrain_generator.generate_pretrain_knowledge(knowledge_graph)
            
            # 3. 整合结果
            results = {
                "case_id": case_id,
                "original_text": fault_text,
                "knowledge_graph": knowledge_graph,
                "pretrain_knowledge": pretrain_knowledge,
                "metadata": {
                    "total_entities": knowledge_graph["graph_stats"]["node_count"],
                    "total_relations": knowledge_graph["graph_stats"]["edge_count"],
                    "entity_types": knowledge_graph["graph_stats"]["entity_types"],
                    "relation_types": knowledge_graph["graph_stats"]["relation_types"]
                }
            }
            
            # 4. 保存结果
            self._save_results(results, case_id)
            
            # 5. 导出训练数据
            self._export_training_data(results["pretrain_knowledge"], case_id)
            
            logger.info(f"故障案例处理完成: {case_id}")
            return results
            
        except Exception as e:
            logger.error(f"处理故障案例失败: {e}")
            raise
    
    def _save_results(self, results: Dict[str, Any], case_id: str):
        """保存结果"""
        if case_id is None:
            case_id = "unknown"
        
        # 保存知识图谱
        kg_file = os.path.join(DATA_DIR, f"{case_id}_knowledge_graph.json")
        with open(kg_file, 'w', encoding='utf-8') as f:
            json.dump(results["knowledge_graph"], f, ensure_ascii=False, indent=2)
        
        # 保存预训练知识
        pretrain_file = os.path.join(DATA_DIR, f"{case_id}_pretrain_knowledge.json")
        with open(pretrain_file, 'w', encoding='utf-8') as f:
            json.dump(results["pretrain_knowledge"], f, ensure_ascii=False, indent=2)
        
        # 保存完整结果
        complete_file = os.path.join(DATA_DIR, f"{case_id}_complete_results.json")
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {DATA_DIR}")
    
    def _export_training_data(self, pretrain_knowledge: Dict[str, Any], case_id: str):
        """导出训练数据"""
        import os
        
        # 创建训练数据目录
        training_dir = "training_data"
        os.makedirs(training_dir, exist_ok=True)
        
        try:
            # 1. 导出自然语言文本
            text = pretrain_knowledge.get("pretrain_text", "")
            txt_file = os.path.join(training_dir, f"{case_id}_pretrain_text.txt")
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # 2. 导出训练样本
            samples = pretrain_knowledge.get("training_samples", [])
            samples_file = os.path.join(training_dir, f"{case_id}_training_samples.jsonl")
            with open(samples_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    sample["case_id"] = case_id
                    json.dump(sample, f, ensure_ascii=False)
                    f.write('\n')
            
            # 3. 导出问答对
            qa_pairs = self._generate_qa_pairs(pretrain_knowledge)
            qa_file = os.path.join(training_dir, f"{case_id}_qa_pairs.jsonl")
            with open(qa_file, 'w', encoding='utf-8') as f:
                for qa in qa_pairs:
                    qa["case_id"] = case_id
                    json.dump(qa, f, ensure_ascii=False)
                    f.write('\n')
            
            # 4. 导出多跳推理数据
            multi_hop_data = pretrain_knowledge.get("multi_hop_reasoning", {})
            if multi_hop_data:
                # 导出推理路径
                reasoning_paths = multi_hop_data.get("reasoning_paths", [])
                paths_file = os.path.join(training_dir, f"{case_id}_reasoning_paths.jsonl")
                with open(paths_file, 'w', encoding='utf-8') as f:
                    for path in reasoning_paths:
                        path["case_id"] = case_id
                        json.dump(path, f, ensure_ascii=False)
                        f.write('\n')
                
                # 导出推理链
                reasoning_chains = multi_hop_data.get("reasoning_chains", [])
                chains_file = os.path.join(training_dir, f"{case_id}_reasoning_chains.jsonl")
                with open(chains_file, 'w', encoding='utf-8') as f:
                    for chain in reasoning_chains:
                        chain["case_id"] = case_id
                        json.dump(chain, f, ensure_ascii=False)
                        f.write('\n')
                
                # 导出推理问答对
                reasoning_qa_pairs = multi_hop_data.get("reasoning_qa_pairs", [])
                reasoning_qa_file = os.path.join(training_dir, f"{case_id}_reasoning_qa_pairs.jsonl")
                with open(reasoning_qa_file, 'w', encoding='utf-8') as f:
                    for qa in reasoning_qa_pairs:
                        qa["case_id"] = case_id
                        json.dump(qa, f, ensure_ascii=False)
                        f.write('\n')
            
            logger.info(f"训练数据导出完成: {case_id}")
            
        except Exception as e:
            logger.error(f"训练数据导出失败: {e}")
            raise
    
    def _generate_qa_pairs(self, pretrain_knowledge: Dict[str, Any]) -> List[Dict[str, str]]:
        """生成问答对"""
        qa_pairs = []
        samples = pretrain_knowledge.get("training_samples", [])
        
        for sample in samples:
            if sample["type"] == "entity_description":
                qa_pairs.append({
                    "question": f"什么是{sample['entity']}？",
                    "answer": sample["training_text"]
                })
                qa_pairs.append({
                    "question": f"{sample['entity']}在故障诊断中起什么作用？",
                    "answer": sample["training_text"]
                })
                qa_pairs.append({
                    "question": f"{sample['entity']}有什么特点？",
                    "answer": sample["training_text"]
                })
            
            elif sample["type"] == "relation_description":
                qa_pairs.append({
                    "question": f"{sample['source']}和{sample['target']}之间是什么关系？",
                    "answer": sample["training_text"]
                })
                qa_pairs.append({
                    "question": f"{sample['source']}如何影响{sample['target']}？",
                    "answer": sample["training_text"]
                })
                qa_pairs.append({
                    "question": f"当{sample['source']}出现问题时，{sample['target']}会怎样？",
                    "answer": sample["training_text"]
                })
            
            elif sample["type"] == "diagnosis_process":
                qa_pairs.append({
                    "question": "电信网络故障诊断的基本流程是什么？",
                    "answer": sample["training_text"]
                })
                qa_pairs.append({
                    "question": "如何进行故障诊断？",
                    "answer": sample["training_text"]
                })
                qa_pairs.append({
                    "question": "故障诊断的关键步骤有哪些？",
                    "answer": sample["training_text"]
                })
        
        return qa_pairs
    
    async def batch_process(self, fault_cases: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """批量处理故障案例"""
        logger.info(f"开始批量处理 {len(fault_cases)} 个故障案例")
        
        results = []
        for i, case in enumerate(fault_cases):
            try:
                logger.info(f"处理第 {i+1}/{len(fault_cases)} 个案例")
                result = await self.process_fault_case(case["text"], case.get("id"))
                results.append(result)
            except Exception as e:
                logger.error(f"处理案例 {case.get('id', i)} 失败: {e}")
                results.append({"error": str(e), "case_id": case.get("id", i)})
        
        return results

async def main():
    """主函数"""
    # 测试用的故障案例
    test_case = {
        "id": "TC0001251339",
        "text": """某局的忙音播放有问题

[meta-content-table]



|案例编码|TC0001251339|TC0001251339|关键字|忙音,播放,异步音|忙音,播放,异步音|忙音,播放,异步音|

|---|---|---|---|---|---|---|

|产品信息|LV1|LV2|LV3|LV4|产品名称|版本号|

|产品信息|云核心网|CS&IMS|Single Voice Core|VoBB|C&C08-128M|-----------------|

|故障类型|Level-1|其他|其他|其他|其他|其他|

|故障类型|Level-2|其他|其他|其他|其他|其他|

|故障类型|Level-3|-|-|-|-|-|

[/meta-content-table][meta-content-table]



|问题描述|某局升级（5K升6008），升级后发现忙音有问题，出现问题的用户是SPM模块带的。<br>无|某局升级（5K升6008），升级后发现忙音有问题，出现问题的用户是SPM模块带的。<br>无|某局升级（5K升6008），升级后发现忙音有问题，出现问题的用户是SPM模块带的。<br>无|某局升级（5K升6008），升级后发现忙音有问题，出现问题的用户是SPM模块带的。<br>无|某局升级（5K升6008），升级后发现忙音有问题，出现问题的用户是SPM模块带的。<br>无|

|---|---|---|---|---|---|

|处理过程|SPD板上加载的忙音有问题。|SPD板上加载的忙音有问题。|SPD板上加载的忙音有问题。|SPD板上加载的忙音有问题。|SPD板上加载的忙音有问题。|

|根因|N/A|N/A|N/A|N/A|N/A|

|解决方案|1、忙音是异步音，该问题一般是异步音有问题造成的，可以先查看一下SPD单板上是否已经加载该语音，如果没有加载，需要按照正确流程加载该语音，如果已经加载，需要重新加载该语音。<br>2、此局点的问题是升级前正常，升级后问题出现。<br>3、该局是SSP局，通过观察，发现该局有4块SPD单板配置了送异步音，由于只会有一块SPD单板放异步音，利用隔离SPD单板的方法，可以确认现在放异步音的单板。发现此单板是一块新扩的SPD单板，主要用来加载智能语音，加载语音后，异步音没有经过测试，但是因为此单板配置为送异步音，导致重新加载交换机后，选择了这块SPD单板播放异步音，出现上述故障。<br>4、SPD放异步音是按照主备用方式，正常运行时异步音由一块单板播放，强烈建议一个局点尤其是SSP局点，只需两块SPD配置为送异步音即可，并严格测试异步音加载是否正确，这样是可以保证异步音的正确播放的。如果多块SPD，尤其新增的SPD，配置了送异步音，如果加载完没有经过测试，很容易因为某种原因占用到此单板，导致异步音无法播放的严重问题。|1、忙音是异步音，该问题一般是异步音有问题造成的，可以先查看一下SPD单板上是否已经加载该语音，如果没有加载，需要按照正确流程加载该语音，如果已经加载，需要重新加载该语音。<br>2、此局点的问题是升级前正常，升级后问题出现。<br>3、该局是SSP局，通过观察，发现该局有4块SPD单板配置了送异步音，由于只会有一块SPD单板放异步音，利用隔离SPD单板的方法，可以确认现在放异步音的单板。发现此单板是一块新扩的SPD单板，主要用来加载智能语音，加载语音后，异步音没有经过测试，但是因为此单板配置为送异步音，导致重新加载交换机后，选择了这块SPD单板播放异步音，出现上述故障。<br>4、SPD放异步音是按照主备用方式，正常运行时异步音由一块单板播放，强烈建议一个局点尤其是SSP局点，只需两块SPD配置为送异步音即可，并严格测试异步音加载是否正确，这样是可以保证异步音的正确播放的。如果多块SPD，尤其新增的SPD，配置了送异步音，如果加载完没有经过测试，很容易因为某种原因占用到此单板，导致异步音无法播放的严重问题。|1、忙音是异步音，该问题一般是异步音有问题造成的，可以先查看一下SPD单板上是否已经加载该语音，如果没有加载，需要按照正确流程加载该语音，如果已经加载，需要重新加载该语音。<br>2、此局点的问题是升级前正常，升级后问题出现。<br>3、该局是SSP局，通过观察，发现该局有4块SPD单板配置了送异步音，由于只会有一块SPD单板放异步音，利用隔离SPD单板的方法，可以确认现在放异步音的单板。发现此单板是一块新扩的SPD单板，主要用来加载智能语音，加载语音后，异步音没有经过测试，但是因为此单板配置为送异步音，导致重新加载交换机后，选择了这块SPD单板播放异步音，出现上述故障。<br>4、SPD放异步音是按照主备用方式，正常运行时异步音由一块单板播放，强烈建议一个局点尤其是SSP局点，只需两块SPD配置为送异步音即可，并严格测试异步音加载是否正确，这样是可以保证异步音的正确播放的。如果多块SPD，尤其新增的SPD，配置了送异步音，如果加载完没有经过测试，很容易因为某种原因占用到此单板，导致异步音无法播放的严重问题。|1、忙音是异步音，该问题一般是异步音有问题造成的，可以先查看一下SPD单板上是否已经加载该语音，如果没有加载，需要按照正确流程加载该语音，如果已经加载，需要重新加载该语音。<br>2、此局点的问题是升级前正常，升级后问题出现。<br>3、该局是SSP局，通过观察，发现该局有4块SPD单板配置了送异步音，由于只会有一块SPD单板放异步音，利用隔离SPD单板的方法，可以确认现在放异步音的单板。发现此单板是一块新扩的SPD单板，主要用来加载智能语音，加载语音后，异步音没有经过测试，但是因为此单板配置为送异步音，导致重新加载交换机后，选择了这块SPD单板播放异步音，出现上述故障。<br>4、SPD放异步音是按照主备用方式，正常运行时异步音由一块单板播放，强烈建议一个局点尤其是SSP局点，只需两块SPD配置为送异步音即可，并严格测试异步音加载是否正确，这样是可以保证异步音的正确播放的。如果多块SPD，尤其新增的SPD，配置了送异步音，如果加载完没有经过测试，很容易因为某种原因占用到此单板，导致异步音无法播放的严重问题。|1、忙音是异步音，该问题一般是异步音有问题造成的，可以先查看一下SPD单板上是否已经加载该语音，如果没有加载，需要按照正确流程加载该语音，如果已经加载，需要重新加载该语音。<br>2、此局点的问题是升级前正常，升级后问题出现。<br>3、该局是SSP局，通过观察，发现该局有4块SPD单板配置了送异步音，由于只会有一块SPD单板放异步音，利用隔离SPD单板的方法，可以确认现在放异步音的单板。发现此单板是一块新扩的SPD单板，主要用来加载智能语音，加载语音后，异步音没有经过测试，但是因为此单板配置为送异步音，导致重新加载交换机后，选择了这块SPD单板播放异步音，出现上述故障。<br>4、SPD放异步音是按照主备用方式，正常运行时异步音由一块单板播放，强烈建议一个局点尤其是SSP局点，只需两块SPD配置为送异步音即可，并严格测试异步音加载是否正确，这样是可以保证异步音的正确播放的。如果多块SPD，尤其新增的SPD，配置了送异步音，如果加载完没有经过测试，很容易因为某种原因占用到此单板，导致异步音无法播放的严重问题。|

|建议与总结|-|-|-|-|-|

[/meta-content-table][meta-content-table]



|附件|-|-|-|-|-|

|---|---|---|---|---|---|

[/meta-content-table]

[meta-content-table]



|案例版本|V01|质量等级|C|

|---|---|---|---|

|发布时间|2004-07-29 08:00:00|更新时间|2022-09-22 21:43:16|

|作者|Zhao |优化者|-|

|案例语言|Chinese|问题单号|-|

|案例密级|Huawei Partner's Engineer|Huawei Partner's Engineer|Huawei Partner's Engineer|

[/meta-content-table]。text1是我生成的内容，text2是原始案例
"""
    }
    
    # 创建GraphGen Enhanced实例
    graphgen = GraphGenEnhanced()
    
    # 处理单个案例
    logger.info("开始处理测试案例...")
    result = await graphgen.process_fault_case(test_case["text"], test_case["id"])
    
    # 打印结果摘要
    print("\n" + "="*50)
    print("处理结果摘要:")
    print("="*50)
    print(f"案例ID: {result['case_id']}")
    print(f"实体数量: {result['metadata']['total_entities']}")
    print(f"关系数量: {result['metadata']['total_relations']}")
    print(f"实体类型: {result['metadata']['entity_types']}")
    print(f"关系类型: {result['metadata']['relation_types']}")
    
    print("\n提取的实体:")
    for entity in result['knowledge_graph']['entities'][:5]:  # 显示前5个
        print(f"  - {entity['name']} ({entity['type']}): {entity['description']}")
    
    print("\n提取的关系:")
    for relation in result['knowledge_graph']['relations'][:5]:  # 显示前5个
        print(f"  - {relation['source']} --[{relation['type']}]--> {relation['target']}")
    
    print("\n生成的预训练知识:")
    print(f"  自然语言文本长度: {result['pretrain_knowledge']['metadata']['text_length']} 字符")
    print(f"  训练样本数量: {result['pretrain_knowledge']['metadata']['sample_count']} 个")
    print(f"  实体数量: {result['pretrain_knowledge']['metadata']['entity_count']} 个")
    print(f"  关系数量: {result['pretrain_knowledge']['metadata']['relation_count']} 个")
    print(f"  推理路径数量: {result['pretrain_knowledge']['metadata']['reasoning_path_count']} 个")
    print(f"  推理链数量: {result['pretrain_knowledge']['metadata']['reasoning_chain_count']} 个")
    print(f"  推理问答对数量: {result['pretrain_knowledge']['metadata']['reasoning_qa_count']} 个")
    
    print("\n" + "="*50)
    print("处理完成！结果已保存到data目录")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
