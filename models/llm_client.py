"""
DeepSeek LLM客户端
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
import aiohttp
from dataclasses import dataclass
from utils.logger import logger
from config import LLM_CONFIG

@dataclass
class LLMResponse:
    """LLM响应数据类"""
    content: str
    usage: Dict[str, int]
    model: str
    finish_reason: str

class DeepSeekClient:
    """DeepSeek API客户端"""
    
    def __init__(self, 
                 api_key: str = None,
                 base_url: str = None,
                 model: str = None):
        """初始化客户端"""
        self.api_key = api_key or LLM_CONFIG["api_key"]
        self.base_url = base_url or LLM_CONFIG["base_url"]
        self.model = model or LLM_CONFIG["model_name"]
        self.max_tokens = LLM_CONFIG["max_tokens"]
        self.temperature = LLM_CONFIG["temperature"]
        self.top_p = LLM_CONFIG["top_p"]
        self.timeout = LLM_CONFIG["request_timeout"]
        self.max_retries = LLM_CONFIG["max_retries"]
        
        # 统计信息
        self.total_requests = 0
        self.total_tokens = 0
        self.error_count = 0
        
    async def generate(self, 
                      prompt: str, 
                      system_prompt: str = None,
                      temperature: float = None,
                      max_tokens: int = None) -> LLMResponse:
        """生成文本"""
        
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": self.top_p,
            "stream": False
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        
                        if response.status == 200:
                            data = await response.json()
                            
                            # 更新统计信息
                            self.total_requests += 1
                            if "usage" in data:
                                self.total_tokens += data["usage"].get("total_tokens", 0)
                            
                            return LLMResponse(
                                content=data["choices"][0]["message"]["content"],
                                usage=data.get("usage", {}),
                                model=data["model"],
                                finish_reason=data["choices"][0]["finish_reason"]
                            )
                        else:
                            error_text = await response.text()
                            logger.error(f"API请求失败: {response.status} - {error_text}")
                            
            except asyncio.TimeoutError:
                logger.warning(f"请求超时，重试 {attempt + 1}/{self.max_retries}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # 指数退避
                
            except Exception as e:
                logger.error(f"请求异常: {e}")
                self.error_count += 1
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(1)
        
        raise Exception("所有重试都失败了")
    
    async def generate_batch(self, 
                           prompts: List[str],
                           system_prompt: str = None,
                           max_concurrent: int = 5) -> List[LLMResponse]:
        """批量生成文本"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(prompt: str) -> LLMResponse:
            async with semaphore:
                return await self.generate(prompt, system_prompt)
        
        tasks = [generate_single(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"批量生成第{i}个结果失败: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "error_count": self.error_count,
            "success_rate": (self.total_requests - self.error_count) / max(self.total_requests, 1)
        }

# 创建全局客户端实例
llm_client = DeepSeekClient()
