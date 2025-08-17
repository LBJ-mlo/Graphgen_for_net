"""
GraphGen Enhanced 配置文件
"""

import os
from typing import Dict, Any

# DeepSeek API配置
DEEPSEEK_API_KEY = "sk-d005b01ce325422eb59daa1cd5355144"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# 项目基础配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# 创建必要的目录
for dir_path in [CACHE_DIR, DATA_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 故障诊断专用配置
FAULT_DIAGNOSIS_CONFIG = {
    "entity_types": [
        "设备",           # 硬件设备，如SPD单板、交换机
        "软件",           # 软件系统，如C&C08-128M
        "故障现象",       # 故障表现，如忙音播放问题
        "故障原因",       # 根本原因，如异步音未加载
        "解决方案",       # 处理方法，如重新加载语音
        "操作步骤",       # 具体操作，如隔离SPD单板
        "配置参数",       # 系统配置，如送异步音配置
        "时间节点",       # 时间信息，如升级前后
        "影响范围",       # 影响对象，如SPM模块用户
        "建议措施"        # 预防建议，如严格测试
    ],
    "relation_types": [
        "导致",           # 故障原因导致故障现象
        "影响",           # 故障影响特定设备或用户
        "解决",           # 解决方案解决特定问题
        "配置",           # 设备配置特定参数
        "升级",           # 系统升级影响功能
        "测试",           # 测试验证功能正常
        "建议",           # 建议措施预防问题
        "依赖",           # 组件间的依赖关系
        "包含",           # 整体包含部分
        "触发"            # 事件触发其他事件
    ]
}

# 质量评估配置
QUALITY_CONFIG = {
    "completeness_weights": {
        "entity_coverage": 0.3,
        "relation_density": 0.2,
        "connectivity": 0.2,
        "knowledge_depth": 0.15,
        "concept_completeness": 0.15
    },
    "quality_thresholds": {
        "min_completeness": 0.8,
        "min_accuracy": 0.9,
        "min_diversity": 0.7,
        "min_coherence": 0.8,
        "min_overall_score": 0.8
    }
}

# 数据过滤配置
DATA_FILTER_CONFIG = {
    "quality_thresholds": {
        "min_overall_score": 0.7,      # 最低总体评分
        "min_consistency_score": 0.6,  # 最低一致性评分
        "allow_hallucination": False   # 是否允许幻觉
    },
    "filter_actions": {
        "remove_low_quality": True,    # 移除低质量数据
        "keep_original": True,         # 保留原始数据
        "save_filtered": True          # 保存过滤后的数据
    }
}

# 文件保存配置
FILE_SAVE_CONFIG = {
    "save_strategy": "minimal",        # 保存策略：minimal（最少文件）, detailed（详细文件）
    "save_key_results": True,          # 保存关键结果文件
    "save_high_quality_only": True,    # 仅保存高质量数据文件
    "save_deduplication_data": True,   # 保存去重数据
    "save_processing_logs": True       # 保存处理日志
}

# LLM配置
LLM_CONFIG = {
    "model_name": DEEPSEEK_MODEL,
    "api_key": DEEPSEEK_API_KEY,
    "base_url": DEEPSEEK_API_BASE,
    "max_tokens": 4096,
    "temperature": 0.1,
    "top_p": 0.9,
    "request_timeout": 60,
    "max_retries": 3
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.path.join(LOGS_DIR, "graphgen_enhanced.log")
}

# 提示词模板配置
PROMPT_TEMPLATES = {
    "entity_extraction": """
请从以下文本中提取实体，并按JSON格式返回：

文本：{text}

请提取以下类型的实体：
- systems: 系统名称
- services: 服务名称  
- tools: 工具名称
- time_periods: 时间信息
- problems: 问题描述
- actions: 操作动作

返回格式：
{{
    "systems": ["系统1", "系统2"],
    "services": ["服务1", "服务2"],
    "tools": ["工具1", "工具2"],
    "time_periods": ["时间1", "时间2"],
    "problems": ["问题1", "问题2"],
    "actions": ["动作1", "动作2"]
}}
""",
    "factual_consistency": """
请评估以下生成的知识与原始文本的事实一致性：

原始文本：{original_text}

生成知识：{generated_knowledge}

请从0到1评分，1表示完全一致，0表示完全不一致。

### 评分：
""",
    "reasoning_quality": """
请评估以下生成知识的推理质量：

原始文本：{original_text}

生成知识：{generated_knowledge}

请从0到1评分，1表示推理质量很高，0表示推理质量很低。

### 评分：
""",
    "fundamental_errors": """
请检查以下生成知识是否存在根本性错误：

原始文本：{original_text}

生成知识：{generated_knowledge}

请从0到1评分，1表示无错误，0表示存在严重错误。

### 评分：
"""
}
