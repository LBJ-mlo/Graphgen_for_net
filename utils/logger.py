"""
日志工具模块
"""

import logging
import os
from config import LOGGING_CONFIG

def setup_logger(name: str = "GraphGen-Enhanced") -> logging.Logger:
    """设置日志记录器"""
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOGGING_CONFIG["level"]))
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 创建格式化器
    formatter = logging.Formatter(LOGGING_CONFIG["format"])
    
    # 文件处理器
    file_handler = logging.FileHandler(
        LOGGING_CONFIG["file"], 
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# 创建全局日志记录器
logger = setup_logger()
