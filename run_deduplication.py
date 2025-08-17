"""
预训练数据去重运行脚本
使用MinHash算法对complete_results.json进行去重处理
"""

import os
import sys
from utils.minhash_deduplicator import process_pretrain_data_with_minhash
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    logger.info("开始预训练数据去重处理")
    
    # 配置参数
    input_file = "data/TC0001251339_complete_results.json"
    output_dir = "pretrain_data_deduplicated"
    similarity_threshold = 0.5  # 相似度阈值，可调整
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        logger.info("请确保文件路径正确，当前工作目录下的data文件夹中应该有TC0001251339_complete_results.json文件")
        return
    
    try:
        # 使用MinHash算法处理数据
        logger.info(f"输入文件: {input_file}")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"相似度阈值: {similarity_threshold}")
        
        processed_data = process_pretrain_data_with_minhash(
            input_file=input_file,
            output_dir=output_dir,
            similarity_threshold=similarity_threshold
        )
        
        # 输出处理结果
        logger.info("=" * 50)
        logger.info("预训练数据去重处理完成！")
        logger.info("=" * 50)
        
        # 显示统计信息
        formatted_data = processed_data["formatted"]
        logger.info(f"📊 处理结果统计:")
        logger.info(f"  📝 纯文本数据: {len(formatted_data['pure_texts'])} 条")
        logger.info(f"  ❓ 问答对数据: {len(formatted_data['qa_formatted'])} 对")
        logger.info(f"  ��️  结构化数据: {len(formatted_data['structured_texts'])} 条")
        
        # 显示去重统计
        logger.info(f"\n📈 去重统计:")
        for data_type, items in processed_data["deduplicated"].items():
            original_count = len(processed_data["extracted"].get(data_type, []))
            deduplicated_count = len(items)
            removed_count = original_count - deduplicated_count
            if original_count > 0:
                removal_rate = (removed_count / original_count) * 100
                logger.info(f"  {data_type}: {original_count} -> {deduplicated_count} (移除 {removed_count} 条, {removal_rate:.1f}%)")
        
        logger.info(f"\n💾 输出文件:")
        logger.info(f"  �� 纯文本: {output_dir}/pure_texts.txt")
        logger.info(f"  ❓ 问答对: {output_dir}/qa_pairs.jsonl")
        logger.info(f"  ��️  结构化: {output_dir}/structured_texts.jsonl")
        logger.info(f"  📊 统计信息: {output_dir}/processing_stats.json")
        
        logger.info(f"\n✅ 去重处理成功完成！")
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return

if __name__ == "__main__":
    main()