# GraphGen Enhanced

一个完整的故障诊断知识图谱生成和预训练数据处理系统，支持端到端的知识提取、去重、质量检测和过滤流程。

## 🚀 项目特性

### 核心功能
- **知识图谱提取**: 从故障文本中自动提取实体和关系
- **预训练数据生成**: 生成多种类型的预训练数据
- **智能去重**: 基于 MinHash 算法的数据去重
- **质量检测**: 基于幻觉检测的数据质量评估
- **数据过滤**: 按类别和按条的质量过滤

### 支持的数据类型
- **实体描述**: 设备、信号、问题等实体的定义
- **关系描述**: 实体间的逻辑关系
- **问答对**: 故障诊断相关的问答
- **推理链**: 故障推理的逻辑链条
- **推理路径**: 完整的故障诊断路径

### 质量评估标准
- **置信度级别**: 只有达到 `high_confidence` 的数据才被认为是好的合成数据
- **按类别评估**: 对每种数据类型进行专门的质量评估
- **按条过滤**: 对每个类别中的每条数据单独进行质量评估

## 📁 项目结构

```
GraphGen-Enhanced/
├── core/                          # 核心模块
│   ├── knowledge_extractor.py     # 知识图谱提取
│   └── pretrain_knowledge_generator.py  # 预训练数据生成
├── huanjue/                       # 幻觉检测模块
│   ├── hallucination_detector.py  # 质量检测器
│   ├── entity_extractor.py        # 实体提取
│   └── main.py                    # 检测模块入口
├── utils/                         # 工具模块
│   ├── minhash_deduplicator.py    # MinHash去重
│   └── logger.py                  # 日志工具
├── models/                        # 模型模块
│   └── llm_client.py              # LLM客户端
├── templates/                     # 模板文件
│   └── fault_diagnosis_templates.py  # 故障诊断模板
├── data/                          # 数据目录
├── training_data/                 # 训练数据
├── logs/                          # 日志文件
├── main.py                        # 主程序入口
├── complete_pipeline.py           # 完整流程
├── batch_pipeline_enhanced.py     # 批量处理
├── run_deduplication.py           # 去重脚本
├── config.py                      # 配置文件
└── requirements.txt               # 依赖包
```

## 🛠️ 安装

### 环境要求
- Python 3.8+
- 依赖包见 `requirements.txt`

### 安装步骤

1. 克隆项目
```bash
git clone https://github.com/LBJ-mlo/Graphgen_for_net.git
cd Graphgen_for_net
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境
```bash
# 编辑 config.py 文件，配置您的 API 密钥和其他设置
```

## 📖 使用方法

### 1. 单案例处理

运行完整流程（生成 -> 去重 -> 质量检测 -> 过滤）：
```bash
python complete_pipeline.py
```

### 2. 批量处理

处理多个故障案例：
```bash
python batch_pipeline_enhanced.py
```

### 3. 单独模块

#### 知识图谱提取
```bash
python main.py
```

#### 数据去重
```bash
python run_deduplication.py
```

#### 质量检测
```bash
cd huanjue
python main.py
```

## 📊 输出文件

### 保存的文件类型

1. **关键结果文件**: `data/{case_id}_key_results.json`
   - 融合所有重要信息的完整结果
   - 包含处理时间、质量评估、统计信息等

2. **高质量数据文件**: `data/{case_id}_high_quality_data.json`
   - 只有达到 `high_confidence` 级别的数据
   - 包含知识图谱和预训练知识

3. **去重数据目录**: `pretrain_data_deduplicated_{case_id}/`
   - 按类别去重后的预训练数据文件
   - 包含各种数据类型的去重结果

### 文件命名规则
- 使用 `case_id` 确保文件名唯一性
- 避免多个案例处理时的文件冲突

## 🔧 配置说明

### 主要配置项

在 `config.py` 中配置：

```python
# LLM 配置
LLM_CONFIG = {
    "api_key": "your_api_key",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com"
}

# 质量评估配置
DATA_FILTER_CONFIG = {
    "quality_thresholds": {
        "min_overall_score": 0.7,
        "min_consistency_score": 0.6,
        "allow_hallucination": False
    }
}

# 文件保存配置
FILE_SAVE_CONFIG = {
    "save_high_quality_only": True,
    "save_deduplicated_data": True,
    "save_key_results": True
}
```

## 🔄 完整流程

```
原始故障文本
    ↓
知识图谱提取 + 预训练数据生成
    ↓
按类别MinHash去重 (entity_descriptions, relation_descriptions, qa_pairs, reasoning_chains, reasoning_paths)
    ↓
按类别质量检测 (整体 + 各类别)
    ↓
按条质量过滤 (只有high_confidence的数据保留)
    ↓
保存关键结果文件 (_key_results.json, _high_quality_data.json)
```

## 📈 性能特点

- **高效去重**: MinHash 算法确保快速去重
- **智能过滤**: 基于置信度的多层次质量过滤
- **文件整合**: 减少输出文件数量，只保存关键信息
- **避免冲突**: 每个案例有独立的文件命名

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。

## 📞 联系方式

如有问题，请通过 GitHub Issues 联系。

---

**注意**: 使用前请确保配置了正确的 API 密钥和相关设置。
