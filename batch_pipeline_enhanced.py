"""
增强版批量处理脚本 - 处理多个故障案例，避免文件冲突
"""

import asyncio
import json
import os
import time
import datetime
from typing import List, Dict, Any
from complete_pipeline import CompletePipeline
from utils.logger import logger

class EnhancedBatchProcessor:
    """增强版批量处理器"""
    
    def __init__(self):
        self.pipeline = CompletePipeline()
        self.batch_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    async def process_batch(self, cases: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """批量处理故障案例"""
        logger.info(f"开始批量处理 {len(cases)} 个案例，批次ID: {self.batch_id}")
        
        results = []
        for i, case in enumerate(cases):
            try:
                case_id = case.get("id", f"CASE_{i+1}")
                # 为每个案例创建唯一的ID，避免文件冲突
                unique_case_id = f"{self.batch_id}_{case_id}"
                
                logger.info(f"处理第 {i+1}/{len(cases)} 个案例: {case_id} (唯一ID: {unique_case_id})")
                
                result = await self.pipeline.run_complete_pipeline(
                    case["text"], 
                    unique_case_id
                )
                results.append(result)
                
                logger.info(f"案例 {case_id} 处理完成")
                
            except Exception as e:
                logger.error(f"处理案例 {case.get('id', f'CASE_{i+1}')} 失败: {e}")
                results.append({
                    "case_id": case.get("id", f"CASE_{i+1}"),
                    "unique_case_id": f"{self.batch_id}_{case.get('id', f'CASE_{i+1}')}",
                    "error": str(e),
                    "status": "failed"
                })
        
        # 保存批量处理结果
        self._save_batch_results(results)
        
        return results
    
    def _save_batch_results(self, results: List[Dict[str, Any]]):
        """保存批量处理结果"""
        batch_summary = {
            "batch_id": self.batch_id,
            "total_cases": len(results),
            "successful_cases": len([r for r in results if "error" not in r]),
            "failed_cases": len([r for r in results if "error" in r]),
            "processing_time": datetime.datetime.now().isoformat(),
            "results": results
        }
        
        # 使用批次ID创建唯一的文件名
        output_file = f"data/batch_processing_results_{self.batch_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"批量处理结果已保存: {output_file}")
    
    def get_case_files(self, case_id: str) -> Dict[str, str]:
        """获取案例相关的所有文件路径"""
        unique_case_id = f"{self.batch_id}_{case_id}"
        return {
            "key_results": f"data/{unique_case_id}_key_results.json",
            "high_quality_data": f"data/{unique_case_id}_high_quality_data.json",
            "deduplication_dir": f"pretrain_data_deduplicated_{unique_case_id}"
        }

async def main():
    """主函数 - 演示增强版批量处理"""
    
    # 示例故障案例列表
    test_cases = [
        {
            "id": "CASE_001",
            "text": """某局升级后忙音播放有问题，SPD单板上加载的忙音有问题。
            忙音是异步音，该问题一般是异步音有问题造成的，可以先查看一下SPD单板上是否已经加载该语音，
            如果没有加载，需要按照正确流程加载该语音，如果已经加载，需要重新加载该语音。"""
        },
        {
            "id": "CASE_002", 
            "text": """某局交换机出现端口故障，端口指示灯不亮，无法正常通信。
            检查发现端口硬件损坏，需要更换端口模块。更换后端口恢复正常工作。"""
        },
        {
            "id": "CASE_003",
            "text": """某局系统升级后出现用户无法登录问题，检查发现数据库连接异常。
            重启数据库服务后问题解决，建议定期检查数据库状态。"""
        }
    ]
    
    # 创建增强版批量处理器
    batch_processor = EnhancedBatchProcessor()
    
    # 执行批量处理
    print("开始增强版批量处理...")
    print(f"批次ID: {batch_processor.batch_id}")
    print(f"处理案例数量: {len(test_cases)}")
    
    try:
        results = await batch_processor.process_batch(test_cases)
        
        # 打印批量处理结果摘要
        print("\n批量处理结果摘要:")
        print("="*50)
        
        successful_count = len([r for r in results if "error" not in r])
        failed_count = len([r for r in results if "error" in r])
        
        print(f"批次ID: {batch_processor.batch_id}")
        print(f"总案例数: {len(results)}")
        print(f"成功案例: {successful_count}")
        print(f"失败案例: {failed_count}")
        
        if successful_count > 0:
            # 统计成功案例的平均指标
            total_entities = 0
            total_relations = 0
            total_quality_score = 0
            
            for result in results:
                if "error" not in result:
                    final_results = result.get("final_results", {})
                    gen_stats = final_results.get("generation_stats", {})
                    total_entities += gen_stats.get("entities", 0)
                    total_relations += gen_stats.get("relations", 0)
                    total_quality_score += final_results.get("quality_score", 0)
            
            avg_entities = total_entities / successful_count
            avg_relations = total_relations / successful_count
            avg_quality = total_quality_score / successful_count
            
            print(f"\n平均指标:")
            print(f"  平均实体数: {avg_entities:.1f}")
            print(f"  平均关系数: {avg_relations:.1f}")
            print(f"  平均质量评分: {avg_quality:.2f}")
        
        # 显示失败案例
        if failed_count > 0:
            print(f"\n失败案例:")
            for result in results:
                if "error" in result:
                    print(f"  {result['case_id']}: {result['error']}")
        
        # 显示文件路径信息
        print(f"\n文件路径信息:")
        print(f"  批量处理结果: data/batch_processing_results_{batch_processor.batch_id}.json")
        
        for i, case in enumerate(test_cases):
            case_id = case["id"]
            files = batch_processor.get_case_files(case_id)
            print(f"\n  案例 {case_id} 文件:")
            print(f"    完整流程结果: {files['pipeline_results']}")
            print(f"    知识图谱: {files['knowledge_graph']}")
            print(f"    预训练数据: {files['pretrain_knowledge']}")
            print(f"    去重数据目录: {files['deduplication_dir']}")
        
        print("\n增强版批量处理完成！")
        
    except Exception as e:
        print(f"批量处理失败: {e}")
        logger.error(f"批量处理失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
