# Copyright 2024 Translation Evaluation Framework
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Translation module for the evaluation framework."""

import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

from .config import Config
from .logger import get_logger
from .utils import to_message

logger = get_logger(__name__)


class BaseTranslator(ABC):
    """Base class for all translators."""
    
    def __init__(self, source_lang: str, target_lang: str):
        """
        Initialize translator.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
        """
        self.source_lang = source_lang.lower()
        self.target_lang = target_lang.lower()
        
        if not Config.validate_language(source_lang):
            raise ValueError(f"Unsupported source language: {source_lang}")
        if not Config.validate_language(target_lang):
            raise ValueError(f"Unsupported target language: {target_lang}")
        
        logger.info(f"Initialized translator: {source_lang} -> {target_lang}")
    
    @abstractmethod
    def translate(self, text: str) -> str:
        """
        Translate text from source to target language.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        pass
    
    def translate_with_metrics(self, text: str) -> Dict[str, Any]:
        """
        Translate text and return metrics.
        
        Args:
            text: Text to translate
            
        Returns:
            Dictionary containing translation and metrics
        """
        start_time = time.time()
        
        try:
            translated_text = self.translate(text)
            success = True
            error_message = None
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            translated_text = ""
            success = False
            error_message = str(e)
        
        end_time = time.time()
        
        return {
            'source_text': text,
            'translated_text': translated_text,
            'source_lang': self.source_lang,
            'target_lang': self.target_lang,
            'translation_time': end_time - start_time,
            'success': success,
            'error_message': error_message,
            'timestamp': time.time()
        }


class LlamaTranslator(BaseTranslator):
    """Translator using Llama model via llama-cpp-python."""
    
    def __init__(self, source_lang: str, target_lang: str, model_path: Optional[str] = None):
        """
        Initialize Llama translator.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            model_path: Path to model file (optional)
        """
        super().__init__(source_lang, target_lang)
        
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python is required for LlamaTranslator")
        
        model_config = Config.get_model_config()
        if model_path:
            model_config['model_path'] = model_path
        
        if not model_config['model_path']:
            raise ValueError("Model path is required. Set MODEL_PATH environment variable or provide model_path parameter.")
        
        logger.info(f"Loading Llama model from: {model_config['model_path']}")
        
        try:
            self.llm = Llama(**model_config)
            logger.info("Llama model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
            raise
        
        # Create language mapping
        self.lang_names = {
            'zh': '中文',
            'en': '英文', 
            'ja': '日文',
            'fr': '法语',
            'it': '意大利语',
            'es': '西班牙语',
            'pt': '葡萄牙语'
        }
    
    def translate(self, text: str) -> str:
        """
        Translate text using Llama model.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        if not text.strip():
            return ""
        
        source_lang_name = self.lang_names.get(self.source_lang, self.source_lang)
        target_lang_name = self.lang_names.get(self.target_lang, self.target_lang)
        
        system_prompt = (
            f"You are a professional translator. Translate the following text from "
            f"{source_lang_name} to {target_lang_name}. "
            f"Provide only the translation without any additional explanations or comments."
        )
        
        user_prompt = f"Translate this text: {text}"
        
        messages = to_message(user_prompt, system_prompt)
        
        try:
            logger.debug(f"Translating: {text[:50]}...")
            
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=Config.DEFAULT_TEMPERATURE,
                max_tokens=Config.DEFAULT_MAX_TOKENS
            )
            
            translated_text = response['choices'][0]['message']['content'].strip()
            
            logger.debug(f"Translation result: {translated_text[:50]}...")
            
            return translated_text
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            raise


class MockTranslator(BaseTranslator):
    """Mock translator for testing purposes."""
    
    def translate(self, text: str) -> str:
        """
        Mock translation that returns text with prefix.
        
        Args:
            text: Text to translate
            
        Returns:
            Mock translated text
        """
        return f"[{self.source_lang}->{self.target_lang}] {text}"


class LlamaQwenTranslator(BaseTranslator):
    """Translator using Llama model via llama-cpp-python of qwen."""
    
    def __init__(self, source_lang: str, target_lang: str, model_path: Optional[str] = None):
        """
        Initialize Llama translator.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            model_path: Path to model file (optional)
        """
        super().__init__(source_lang, target_lang)
        
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python is required for LlamaTranslator")
        
        model_config = Config.get_model_config()
        if model_path:
            model_config['model_path'] = model_path
        
        if not model_config['model_path']:
            raise ValueError("Model path is required. Set MODEL_PATH environment variable or provide model_path parameter.")
        
        logger.info(f"Loading Llama model from: {model_config['model_path']}")
        
        try:
            self.llm = Llama(**model_config)
            logger.info("Llama model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
            raise
        
        # Create language mapping
        self.lang_names = {
            'zh': '中文',
            'en': '英文', 
            'ja': '日文',
            'fr': '法语',
            'it': '意大利语',
            'es': '西班牙语',
            'pt': '葡萄牙语'
        }
    
    def translate(self, text: str) -> str:
        """
        Translate text using Llama model.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        if not text.strip():
            return ""
        
        source_language = self.lang_names.get(self.source_lang.lower(), self.source_lang.lower())
        target_language = self.lang_names.get(self.target_lang.lower(), self.target_lang.lower())
        
        system_prompt = (
                    """
        **角色**：你是一名专业的{target_language}翻译专家。

        **任务**：请将以下{source_language}文本翻译成{target_language}。

        **翻译要求与规范**：
        1.  **保留规则**：原文中出现的以下内容必须**完全保留其原始形式**，不得进行任何翻译、改写或转换：
            *   技术术语、行业特定词汇与专业名词
            *   产品名称、公司名称、品牌名称、商标
            *   代码片段、函数名、变量名、文件名、路径
            *   专有名词（如人名、地名）、缩写、首字母缩略词（例如：API, NLP, RAG, JSON, Python, ChatGPT）
        2.  **语言质量**：译文必须符合{target_language}的母语表达习惯，确保行文流畅、自然地道，彻底避免生硬晦涩的直译。
        3.  **专业准确**：在翻译过程中必须保持原文的专业性和信息的准确性，清晰无误地传达原文的真实含义。
        4.  **优化处理**：如遇原文存在表达模糊、歧义或逻辑不清的情况，允许在忠实于原意的基础上，依照{target_language}的语言习惯对语序、措辞或句式进行必要的调整，以显著提升译文的可读性和逻辑性。

        **重要注意事项**：
        *   **输出纯净**：最终的翻译结果必须是**纯净的{target_language}文本**（除根据规则必须保留的{source_language}内容外），不得包含任何额外的解释、注释、说明或标记（如省略号、项目符号）。
        *   **仅输出译文**：请直接提供完成后的译文，无需重复任务说明或待翻译文本。

        **待翻译的{target_language}文本**：""".format(source_language=source_language, target_language=target_language)
                )
        
        user_prompt = f"Translate this text: {text}"
        
        messages = to_message(user_prompt, system_prompt)
        
        try:
            logger.debug(f"Translating: {text[:50]}...")
            
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=Config.DEFAULT_TEMPERATURE,
                max_tokens=Config.DEFAULT_MAX_TOKENS
            )
            
            translated_text = response['choices'][0]['message']['content'].strip()
            
            logger.debug(f"Translation result: {translated_text[:50]}...")
            
            return translated_text
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            raise

translator_classes = {
    'llama': LlamaTranslator,
    'qwen': LlamaQwenTranslator,
}


def create_translator(
    translator_type: str,
    source_lang: str,
    target_lang: str,
    **kwargs
) -> BaseTranslator:
    """
    Factory function to create translator instances.
    
    Args:
        translator_type: Type of translator ('llama', 'mock')
        source_lang: Source language code
        target_lang: Target language code
        **kwargs: Additional arguments for translator
        
    Returns:
        Translator instance
    """
    
    if translator_type not in translator_classes:
        raise ValueError(f"Unknown translator type: {translator_type}")
    
    translator_class = translator_classes[translator_type]
    return translator_class(source_lang, target_lang, **kwargs)
