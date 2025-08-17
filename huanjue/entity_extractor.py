# -*- coding: utf-8 -*-
"""
实体提取模块
"""

import json
import logging
from typing import Dict, List, Any
import requests
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL, PROMPT_TEMPLATES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityExtractor:
    """实体提取器"""
    
    def __init__(self):
        self.api_key = DEEPSEEK_API_KEY
        self.api_base = DEEPSEEK_API_BASE
        self.model = DEEPSEEK_MODEL
    
    def call_deepseek_api(self, prompt: str) -> str:
        """调用DeepSeek API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"API调用失败: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"API调用异常: {str(e)}")
            return ""
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """从文本中提取实体"""
        try:
            # 构建提示词
            prompt = PROMPT_TEMPLATES['entity_extraction'].format(text=text)
            
            # 调用API
            response = self.call_deepseek_api(prompt)
            
            if not response:
                logger.error("API返回空响应")
                return self._get_default_entities()
            
            # 解析JSON响应
            try:
                entities = json.loads(response)
                logger.info(f"成功提取实体: {entities}")
                return entities
            except json.JSONDecodeError:
                logger.error(f"JSON解析失败: {response}")
                return self._get_default_entities()
                
        except Exception as e:
            logger.error(f"实体提取异常: {str(e)}")
            return self._get_default_entities()
    
    def _get_default_entities(self) -> Dict[str, List[str]]:
        """获取默认实体结构"""
        return {
            "systems": [],
            "services": [],
            "tools": [],
            "time_periods": [],
            "problems": [],
            "actions": []
        }
    
    def generate_entity_combinations(self, entities: Dict[str, List[str]], max_combinations: int = 10) -> List[List[str]]:
        """生成实体组合"""
        combinations = []
        
        # 获取所有实体
        all_entities = []
        for category, entity_list in entities.items():
            all_entities.extend(entity_list)
        
        # 生成2-3个实体的组合
        for i in range(len(all_entities)):
            for j in range(i + 1, len(all_entities)):
                combinations.append([all_entities[i], all_entities[j]])
                
                # 添加3个实体的组合
                for k in range(j + 1, len(all_entities)):
                    combinations.append([all_entities[i], all_entities[j], all_entities[k]])
                    
                    if len(combinations) >= max_combinations:
                        break
                if len(combinations) >= max_combinations:
                    break
            if len(combinations) >= max_combinations:
                break
        
        logger.info(f"生成实体组合: {combinations}")
        return combinations[:max_combinations]
    
    def extract_and_combine(self, text: str, max_combinations: int = 10) -> Dict[str, Any]:
        """提取实体并生成组合"""
        # 提取实体
        entities = self.extract_entities(text)
        
        # 生成组合
        combinations = self.generate_entity_combinations(entities, max_combinations)
        
        return {
            "entities": entities,
            "combinations": combinations,
            "total_entities": sum(len(entity_list) for entity_list in entities.values()),
            "total_combinations": len(combinations)
        }
