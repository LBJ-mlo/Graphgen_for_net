"""
批量处理脚本 - 处理多个故障案例
"""

import asyncio
import json
import os
from typing import List, Dict, Any
from complete_pipeline import CompletePipeline
from utils.logger import logger

class BatchProcessor:
    """批量处理器"""
    
    def __init__(self):
        self.pipeline = CompletePipeline()
    
    async def process_batch(self, cases: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """批量处理故障案例"""
        logger.info(f"开始批量处理 {len(cases)} 个案例")
        
        results = []
        for i, case in enumerate(cases):
            try:
                logger.info(f"处理第 {i+1}/{len(cases)} 个案例: {case.get('id', f'CASE_{i+1}')}")
                
                result = await self.pipeline.run_complete_pipeline(
                    case["text"], 
                    case.get("id", f"CASE_{i+1}")
                )
                results.append(result)
                
                logger.info(f"案例 {case.get('id', f'CASE_{i+1}')} 处理完成")
                
            except Exception as e:
                logger.error(f"处理案例 {case.get('id', f'CASE_{i+1}')} 失败: {e}")
                results.append({
                    "case_id": case.get("id", f"CASE_{i+1}"),
                    "error": str(e),
                    "status": "failed"
                })
        
        # 保存批量处理结果
        self._save_batch_results(results)
        
        return results
    
    def _save_batch_results(self, results: List[Dict[str, Any]]):
        """保存批量处理结果"""
        batch_summary = {
            "total_cases": len(results),
            "successful_cases": len([r for r in results if "error" not in r]),
            "failed_cases": len([r for r in results if "error" in r]),
            "results": results
        }
        
        # 使用时间戳创建唯一的文件名
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/batch_processing_results_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"批量处理结果已保存: {output_file}")

async def main():
    """主函数 - 演示批量处理"""
    
    # 示例故障案例列表（原始文本可以简略）
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
    
    # 创建批量处理器
    batch_processor = BatchProcessor()
    
    # 执行批量处理
    print("开始批量处理...")
    print(f"处理案例数量: {len(test_cases)}")
    
    try:
        results = await batch_processor.process_batch(test_cases)
        
        # 打印批量处理结果摘要
        print("\n批量处理结果摘要:")
        print("="*50)
        
        successful_count = len([r for r in results if "error" not in r])
        failed_count = len([r for r in results if "error" in r])
        
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
        
        print(f"\n结果文件:")
        print(f"  批量处理结果: data/batch_processing_results.json")
        print(f"  各案例结果: data/{{case_id}}_pipeline_results.json")
        
        print("\n批量处理完成！")
        
    except Exception as e:
        print(f"批量处理失败: {e}")
        logger.error(f"批量处理失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
