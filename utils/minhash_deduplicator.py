"""
基于MinHash的文本去重器
使用MinHash和LSH算法进行高效的文本去重
"""

import json
import re
import hashlib
import numpy as np
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TextItem:
    """文本项数据类"""
    id: str
    content: str
    type: str
    metadata: Dict[str, Any] = None

class MinHashDeduplicator:
    """基于MinHash的文本去重器"""
    
    def __init__(self, 
                 num_permutations: int = 128,
                 similarity_threshold: float = 0.8,
                 shingle_size: int = 3):
        """
        初始化MinHash去重器
        
        Args:
            num_permutations: MinHash的排列数，越多越精确但越慢
            similarity_threshold: 相似度阈值，超过此值认为是重复
            shingle_size: k-shingle的大小
        """
        self.num_permutations = num_permutations
        self.similarity_threshold = similarity_threshold
        self.shingle_size = shingle_size
        
        # 生成随机排列
        self.permutations = self._generate_permutations()
        
        # 存储文档的MinHash签名
        self.signatures = {}
        
        # 存储LSH桶
        self.lsh_buckets = defaultdict(set)
        
        # 存储去重结果
        self.deduplicated_items = []
        self.duplicate_groups = []
        
        logger.info(f"MinHash去重器初始化完成: 排列数={num_permutations}, 相似度阈值={similarity_threshold}")
    
    def _generate_permutations(self) -> List[Tuple[int, int]]:
        """生成随机排列用于MinHash"""
        np.random.seed(42)  # 固定随机种子确保可重现
        
        # 生成随机排列参数 (a, b)
        # 每个排列的形式: h(x) = (a * x + b) % prime
        prime = 2**31 - 1  # 梅森素数
        permutations = []
        
        for _ in range(self.num_permutations):
            a = np.random.randint(1, prime)
            b = np.random.randint(0, prime)
            permutations.append((a, b))
        
        return permutations
    
    def _get_shingles(self, text: str) -> Set[str]:
        """获取文本的k-shingles"""
        # 文本预处理
        text = self._preprocess_text(text)
        
        # 生成k-shingles
        shingles = set()
        words = text.split()
        
        if len(words) < self.shingle_size:
            # 如果词数不足，使用整个文本
            shingles.add(text)
        else:
            for i in range(len(words) - self.shingle_size + 1):
                shingle = ' '.join(words[i:i + self.shingle_size])
                shingles.add(shingle)
        
        return shingles
    
    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 转换为小写
        text = text.lower()
        
        # 移除标点符号（保留中文标点）
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _compute_minhash_signature(self, shingles: Set[str]) -> List[int]:
        """计算MinHash签名"""
        signature = []
        prime = 2**31 - 1
        
        for a, b in self.permutations:
            min_hash = float('inf')
            
            for shingle in shingles:
                # 计算shingle的哈希值
                shingle_hash = hash(shingle) % prime
                
                # 应用排列函数
                hash_value = (a * shingle_hash + b) % prime
                
                if hash_value < min_hash:
                    min_hash = hash_value
            
            signature.append(min_hash)
        
        return signature
    
    def _compute_jaccard_similarity(self, shingles1: Set[str], shingles2: Set[str]) -> float:
        """计算Jaccard相似度"""
        intersection = len(shingles1 & shingles2)
        union = len(shingles1 | shingles2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _estimate_jaccard_from_signatures(self, sig1: List[int], sig2: List[int]) -> float:
        """从MinHash签名估计Jaccard相似度"""
        if len(sig1) != len(sig2):
            return 0.0
        
        # 计算签名中相同位置的值相等的比例
        matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
        return matches / len(sig1)
    
    def _build_lsh_buckets(self):
        """构建LSH桶"""
        self.lsh_buckets.clear()
        
        # 将签名分成多个band
        band_size = 4  # 每个band的大小
        num_bands = self.num_permutations // band_size
        
        for doc_id, signature in self.signatures.items():
            for band_idx in range(num_bands):
                start_idx = band_idx * band_size
                end_idx = start_idx + band_size
                band = tuple(signature['signature'][start_idx:end_idx])
                
                # 将文档添加到对应的桶中
                bucket_key = (band_idx, band)
                self.lsh_buckets[bucket_key].add(doc_id)
        
        logger.info(f"LSH桶构建完成，共 {len(self.lsh_buckets)} 个桶")
    
    def add_documents(self, documents: List[TextItem]):
        """添加文档到去重器"""
        logger.info(f"开始处理 {len(documents)} 个文档")
        
        for doc in documents:
            # 获取shingles
            shingles = self._get_shingles(doc.content)
            
            # 计算MinHash签名
            signature = self._compute_minhash_signature(shingles)
            
            # 存储签名
            self.signatures[doc.id] = {
                'signature': signature,
                'shingles': shingles,
                'item': doc
            }
        
        # 构建LSH桶
        self._build_lsh_buckets()
        
        logger.info(f"文档处理完成，共 {len(self.signatures)} 个文档")
    
    def find_duplicates(self) -> List[List[str]]:
        """查找重复文档组"""
        logger.info("开始查找重复文档...")
        
        duplicate_groups = []
        processed = set()
        
        # 遍历LSH桶
        for bucket_key, doc_ids in self.lsh_buckets.items():
            if len(doc_ids) < 2:  # 桶中只有一个文档，跳过
                continue
            
            # 检查桶中的文档对
            doc_id_list = list(doc_ids)
            for i in range(len(doc_id_list)):
                for j in range(i + 1, len(doc_id_list)):
                    doc_id1, doc_id2 = doc_id_list[i], doc_id_list[j]
                    
                    # 跳过已处理的文档
                    if doc_id1 in processed or doc_id2 in processed:
                        continue
                    
                    # 计算相似度
                    sig1 = self.signatures[doc_id1]['signature']
                    sig2 = self.signatures[doc_id2]['signature']
                    estimated_similarity = self._estimate_jaccard_from_signatures(sig1, sig2)
                    
                    # 如果估计相似度超过阈值，进行精确计算
                    if estimated_similarity >= self.similarity_threshold * 0.8:  # 稍微降低阈值
                        shingles1 = self.signatures[doc_id1]['shingles']
                        shingles2 = self.signatures[doc_id2]['shingles']
                        exact_similarity = self._compute_jaccard_similarity(shingles1, shingles2)
                        
                        if exact_similarity >= self.similarity_threshold:
                            # 找到重复组
                            group = [doc_id1, doc_id2]
                            
                            # 查找更多相似的文档
                            for doc_id3 in doc_ids:
                                if doc_id3 not in group and doc_id3 not in processed:
                                    sig3 = self.signatures[doc_id3]['signature']
                                    est_sim = self._estimate_jaccard_from_signatures(sig1, sig3)
                                    
                                    if est_sim >= self.similarity_threshold * 0.8:
                                        shingles3 = self.signatures[doc_id3]['shingles']
                                        exact_sim = self._compute_jaccard_similarity(shingles1, shingles3)
                                        
                                        if exact_sim >= self.similarity_threshold:
                                            group.append(doc_id3)
                            
                            duplicate_groups.append(group)
                            processed.update(group)
        
        self.duplicate_groups = duplicate_groups
        logger.info(f"重复查找完成，找到 {len(duplicate_groups)} 个重复组")
        
        return duplicate_groups
    
    def deduplicate(self) -> List[TextItem]:
        """执行去重，返回去重后的文档列表"""
        logger.info("开始执行去重...")
        
        # 查找重复
        duplicate_groups = self.find_duplicates()
        
        # 选择保留的文档
        keep_docs = set()
        for group in duplicate_groups:
            # 选择质量最高的文档（这里简单地选择第一个）
            # 在实际应用中，可以根据文档长度、完整性等指标选择
            keep_docs.add(group[0])
        
        # 添加没有重复的文档
        all_doc_ids = set(self.signatures.keys())
        unique_docs = all_doc_ids - set().union(*duplicate_groups)
        keep_docs.update(unique_docs)
        
        # 构建去重后的文档列表
        self.deduplicated_items = []
        for doc_id in keep_docs:
            self.deduplicated_items.append(self.signatures[doc_id]['item'])
        
        logger.info(f"去重完成: {len(self.signatures)} -> {len(self.deduplicated_items)} 个文档")
        
        return self.deduplicated_items
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取去重统计信息"""
        total_docs = len(self.signatures)
        deduplicated_docs = len(self.deduplicated_items)
        removed_docs = total_docs - deduplicated_docs
        duplicate_groups = len(self.duplicate_groups)
        
        stats = {
            "total_documents": total_docs,
            "deduplicated_documents": deduplicated_docs,
            "removed_documents": removed_docs,
            "duplicate_groups": duplicate_groups,
            "deduplication_rate": removed_docs / total_docs if total_docs > 0 else 0,
            "duplicate_groups_detail": self.duplicate_groups
        }
        
        return stats

class PretrainDataProcessor:
    """预训练数据处理器"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        """初始化数据处理器"""
        self.deduplicator = MinHashDeduplicator(similarity_threshold=similarity_threshold)
        self.processed_data = {}
        
    def process_complete_results(self, file_path: str) -> Dict[str, Any]:
        """处理complete_results.json文件"""
        logger.info(f"开始处理文件: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取数据
            extracted_data = self._extract_data(data)
            
            # 去重处理
            deduplicated_data = self._deduplicate_data(extracted_data)
            
            # 格式化输出
            formatted_data = self._format_data(deduplicated_data)
            
            self.processed_data = {
                "extracted": extracted_data,
                "deduplicated": deduplicated_data,
                "formatted": formatted_data
            }
            
            return self.processed_data
            
        except Exception as e:
            logger.error(f"处理文件失败: {e}")
            raise
    
    def _extract_data(self, data: Dict[str, Any]) -> Dict[str, List[TextItem]]:
        """从数据中提取文本项"""
        extracted = {
            "knowledge_texts": [],
            "reasoning_texts": [],
            "qa_pairs": [],
            "entity_descriptions": [],
            "relation_descriptions": [],
            "diagnosis_processes": []
        }
        
        pretrain_knowledge = data.get("pretrain_knowledge", {})
        
        # 1. 提取知识描述文本
        pretrain_text = pretrain_knowledge.get("pretrain_text", "")
        if pretrain_text:
            paragraphs = self._split_into_paragraphs(pretrain_text)
            for i, para in enumerate(paragraphs):
                item = TextItem(
                    id=f"knowledge_{i}",
                    content=para,
                    type="knowledge_description",
                    metadata={"source": "pretrain_text", "paragraph_index": i}
                )
                extracted["knowledge_texts"].append(item)
        
        # 2. 提取推理链文本
        multi_hop_reasoning = pretrain_knowledge.get("multi_hop_reasoning", {})
        reasoning_chains = multi_hop_reasoning.get("reasoning_chains", [])
        for i, chain in enumerate(reasoning_chains):
            reasoning_text = chain.get("reasoning_text", "")
            if reasoning_text:
                item = TextItem(
                    id=f"reasoning_{i}",
                    content=reasoning_text,
                    type="reasoning_chain",
                    metadata={"source": "multi_hop_reasoning", "chain_index": i}
                )
                extracted["reasoning_texts"].append(item)
        
        # 3. 提取问答对
        reasoning_qa_pairs = multi_hop_reasoning.get("reasoning_qa_pairs", [])
        for i, qa in enumerate(reasoning_qa_pairs):
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            qa_type = qa.get("type", "")
            
            if question and answer:
                # 将问答对合并为一个文本项
                qa_text = f"问题：{question}\n答案：{answer}"
                item = TextItem(
                    id=f"qa_{i}",
                    content=qa_text,
                    type="qa_pair",
                    metadata={"question": question, "answer": answer, "qa_type": qa_type}
                )
                extracted["qa_pairs"].append(item)
        
        # 4. 提取训练样本
        training_samples = pretrain_knowledge.get("training_samples", [])
        for i, sample in enumerate(training_samples):
            sample_type = sample.get("type", "")
            training_text = sample.get("training_text", "")
            
            if training_text:
                item = TextItem(
                    id=f"sample_{i}",
                    content=training_text,
                    type=sample_type,
                    metadata={"source": "training_samples", "sample_index": i}
                )
                
                if sample_type == "entity_description":
                    extracted["entity_descriptions"].append(item)
                elif sample_type == "relation_description":
                    extracted["relation_descriptions"].append(item)
                elif sample_type == "diagnosis_process":
                    extracted["diagnosis_processes"].append(item)
        
        return extracted
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """将文本分割为段落"""
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        # 过滤掉太短的段落
        filtered = [p for p in paragraphs if len(p) > 30]
        return filtered
    
    def _deduplicate_data(self, extracted_data: Dict[str, List[TextItem]]) -> Dict[str, List[TextItem]]:
        """对提取的数据进行去重"""
        deduplicated = {}
        
        for data_type, items in extracted_data.items():
            if not items:
                deduplicated[data_type] = []
                continue
            
            logger.info(f"对 {data_type} 进行去重: {len(items)} 条")
            
            # 重置去重器
            self.deduplicator = MinHashDeduplicator(similarity_threshold=0.8)
            
            # 添加文档
            self.deduplicator.add_documents(items)
            
            # 执行去重
            deduplicated_items = self.deduplicator.deduplicate()
            
            # 获取统计信息
            stats = self.deduplicator.get_statistics()
            logger.info(f"  {data_type} 去重结果: {stats['total_documents']} -> {stats['deduplicated_documents']} (移除 {stats['removed_documents']} 条)")
            
            deduplicated[data_type] = deduplicated_items
        
        return deduplicated
    
    def _format_data(self, deduplicated_data: Dict[str, List[TextItem]]) -> Dict[str, List[str]]:
        """格式化数据用于预训练"""
        formatted = {
            "pure_texts": [],
            "qa_formatted": [],
            "structured_texts": []
        }
        
        # 1. 纯文本数据
        for data_type in ["knowledge_texts", "reasoning_texts", "entity_descriptions", 
                         "relation_descriptions", "diagnosis_processes"]:
            for item in deduplicated_data.get(data_type, []):
                formatted["pure_texts"].append(item.content)
        
        # 2. 问答格式数据
        for item in deduplicated_data.get("qa_pairs", []):
            metadata = item.metadata or {}
            qa_pair = {
                "question": metadata.get("question", ""),
                "answer": metadata.get("answer", ""),
                "type": metadata.get("qa_type", "")
            }
            formatted["qa_formatted"].append(qa_pair)
        
        # 3. 结构化文本数据
        for data_type in ["knowledge_texts", "reasoning_texts", "entity_descriptions", 
                         "relation_descriptions", "diagnosis_processes"]:
            for item in deduplicated_data.get(data_type, []):
                structured_item = {
                    "type": item.type,
                    "content": item.content,
                    "metadata": item.metadata
                }
                formatted["structured_texts"].append(structured_item)
        
        return formatted
    
    def save_processed_data(self, output_dir: str):
        """保存处理后的数据"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        formatted_data = self.processed_data["formatted"]
        
        # 保存纯文本数据
        pure_text_file = os.path.join(output_dir, "pure_texts.txt")
        with open(pure_text_file, 'w', encoding='utf-8') as f:
            for text in formatted_data["pure_texts"]:
                f.write(text + "\n\n")
        
        # 保存问答格式数据
        qa_file = os.path.join(output_dir, "qa_pairs.jsonl")
        with open(qa_file, 'w', encoding='utf-8') as f:
            for qa in formatted_data["qa_formatted"]:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
        
        # 保存结构化文本数据
        structured_file = os.path.join(output_dir, "structured_texts.jsonl")
        with open(structured_file, 'w', encoding='utf-8') as f:
            for item in formatted_data["structured_texts"]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # 保存统计信息
        stats_file = os.path.join(output_dir, "processing_stats.json")
        stats = {}
        for data_type, items in self.processed_data["deduplicated"].items():
            original_count = len(self.processed_data["extracted"].get(data_type, []))
            deduplicated_count = len(items)
            stats[data_type] = {
                "original": original_count,
                "deduplicated": deduplicated_count,
                "removed": original_count - deduplicated_count
            }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"处理后的数据已保存到: {output_dir}")

def process_pretrain_data_with_minhash(input_file: str, output_dir: str, similarity_threshold: float = 0.8):
    """使用MinHash算法处理预训练数据"""
    logger.info("开始使用MinHash算法处理预训练数据")
    
    # 创建数据处理器
    processor = PretrainDataProcessor(similarity_threshold=similarity_threshold)
    
    # 处理数据
    processed_data = processor.process_complete_results(input_file)
    
    # 保存结果
    processor.save_processed_data(output_dir)
    
    # 输出统计信息
    logger.info("MinHash预训练数据处理完成")
    
    return processed_data