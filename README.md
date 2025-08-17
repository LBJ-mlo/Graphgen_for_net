# GraphGen Enhanced

基于知识图谱的故障诊断预训练知识生成系统

## 项目概述

GraphGen Enhanced 是一个专门针对故障诊断领域的知识图谱构建和预训练知识生成系统。它能够从故障案例文本中提取结构化知识，并生成高质量的预训练数据。

## 核心功能

1. **知识图谱提取**: 从故障案例文本中提取实体和关系
2. **预训练知识生成**: 基于知识图谱生成结构化的预训练知识
3. **质量评估**: 评估生成知识的完整性和质量
4. **批量处理**: 支持批量处理多个故障案例

## 项目结构

```
GraphGen-Enhanced/
├── config.py                          # 配置文件
├── main.py                            # 主程序入口（包含所有核心功能）
├── requirements.txt                   # 依赖包
├── core/                              # 核心模块
│   ├── knowledge_extractor.py         # 知识图谱提取器
│   └── pretrain_knowledge_generator.py # 预训练知识生成器（包含多跳推理）
├── models/                            # 模型模块
│   └── llm_client.py                  # DeepSeek LLM客户端
├── templates/                         # 提示词模板
│   └── fault_diagnosis_templates.py   # 故障诊断专用模板
├── utils/                             # 工具模块
│   └── logger.py                      # 日志工具
├── data/                              # 数据目录
├── cache/                             # 缓存目录
└── logs/                              # 日志目录
```

## 安装和配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

在 `config.py` 中配置您的DeepSeek API密钥：

```python
DEEPSEEK_API_KEY = "your_api_key_here"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
```

## 使用方法

### 1. 启动系统

```bash
# 直接运行主程序
python main.py
```

### 2. 处理单个故障案例

```python
import asyncio
from main import GraphGenEnhanced

async def process_case():
    graphgen = GraphGenEnhanced()
    
    fault_text = """某局升级后忙音播放有问题，SPD单板上加载的忙音有问题。
    忙音是异步音，该问题一般是异步音有问题造成的。"""
    
    result = await graphgen.process_fault_case(fault_text, "CASE001")
    print(f"处理完成，提取了 {result['metadata']['total_entities']} 个实体")

asyncio.run(process_case())
```

### 3. 批量处理

```python
import asyncio
from main import GraphGenEnhanced

async def batch_process():
    graphgen = GraphGenEnhanced()
    
    fault_cases = [
        {"id": "CASE001", "text": "故障案例1..."},
        {"id": "CASE002", "text": "故障案例2..."},
        # 更多案例...
    ]
    
    results = await graphgen.batch_process(fault_cases)
    
    for result in results:
        if "error" not in result:
            print(f"案例 {result['case_id']} 处理成功")
        else:
            print(f"案例 {result['case_id']} 处理失败: {result['error']}")

asyncio.run(batch_process())
```

## 输出格式

### 知识图谱输出

```json
{
  "entities": [
    {
      "id": "SPD单板",
      "name": "SPD单板",
      "type": "设备",
      "description": "负责播放忙音的硬件设备",
      "confidence": 1.0
    }
  ],
  "relations": [
    {
      "source": "异步音加载问题",
      "target": "忙音播放问题",
      "type": "导致",
      "description": "异步音加载问题导致忙音播放问题",
      "confidence": 1.0
    }
  ],
  "graph_stats": {
    "node_count": 5,
    "edge_count": 4,
    "entity_types": ["设备", "故障现象", "故障原因"],
    "relation_types": ["导致", "影响", "解决"]
  }
}
```

### 预训练知识输出

系统生成自然语言格式的预训练数据，包括：

#### 1. 自然语言文本
```text
在电信网络故障诊断中，SPD单板是一个重要的硬件设备，负责播放忙音等异步音。当系统升级后出现忙音播放问题时，通常是由于异步音加载问题造成的。SPD单板需要正确加载语音文件才能正常工作，如果加载失败或配置错误，就会导致忙音无法正常播放。

故障诊断的基本流程包括：首先识别故障现象，然后分析可能的故障原因，接着制定解决方案，最后验证修复效果。在整个过程中，需要综合考虑设备状态、配置参数、环境因素等多个方面。
```

#### 2. 训练样本
```json
{
  "type": "entity_description",
  "entity": "SPD单板",
  "entity_type": "设备",
  "training_text": "在电信网络故障诊断中，SPD单板是一个设备，负责播放忙音的硬件设备。"
}
```

#### 3. 问答对
```json
{
  "question": "什么是SPD单板？",
  "answer": "在电信网络故障诊断中，SPD单板是一个设备，负责播放忙音的硬件设备。"
}
```

#### 4. 输出文件格式
- `{case_id}_pretrain_text.txt`: 自然语言预训练文本
- `{case_id}_pretrain_text.jsonl`: JSONL格式的预训练文本
- `{case_id}_training_samples.jsonl`: 训练样本数据
- `{case_id}_qa_pairs.jsonl`: 问答对数据
- `{case_id}_complete_training_data.json`: 完整训练数据

## 故障诊断专用配置

系统针对故障诊断领域进行了专门配置：

### 实体类型
- 设备：硬件设备、单板、模块等
- 软件：系统软件、版本等
- 故障现象：具体的故障表现
- 故障原因：根本原因分析
- 解决方案：具体的处理方法
- 操作步骤：详细的操作流程
- 配置参数：相关配置信息
- 时间节点：故障发生时间
- 影响范围：受影响的用户或系统
- 建议措施：预防和优化建议

### 关系类型
- 导致：故障原因导致故障现象
- 影响：故障影响特定设备或用户
- 解决：解决方案解决特定问题
- 配置：设备配置特定参数
- 升级：系统升级影响功能
- 测试：测试验证功能正常
- 建议：建议措施预防问题
- 依赖：组件间的依赖关系
- 包含：整体包含部分
- 触发：事件触发其他事件

## 质量评估

系统包含多个质量评估指标：

1. **完整性**: 知识是否完整覆盖了故障诊断的关键要素
2. **准确性**: 知识内容是否准确无误
3. **实用性**: 知识是否具有实际应用价值
4. **逻辑性**: 知识结构是否逻辑清晰
5. **专业性**: 知识是否具有专业深度

## 注意事项

1. 确保API密钥配置正确且有足够的配额
2. 处理大量案例时注意API调用频率限制
3. 生成的预训练知识需要人工验证和优化
4. 建议定期备份生成的知识数据

## 扩展开发

系统采用模块化设计，可以轻松扩展：

1. 添加新的实体类型和关系类型
2. 自定义质量评估指标
3. 集成其他LLM服务
4. 添加可视化功能

## 许可证

本项目采用 Apache License 2.0 许可证。
