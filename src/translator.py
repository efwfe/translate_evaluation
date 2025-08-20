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
            'zh': 'Chinese',
            'en': 'English', 
            'ja': 'Japanese',
            'fr': 'French',
            'it': 'Italian',
            'es': 'Spanish',
            'pt': 'Portuguese'
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
    translator_classes = {
        'llama': LlamaTranslator,
        'mock': MockTranslator,
    }
    
    if translator_type not in translator_classes:
        raise ValueError(f"Unknown translator type: {translator_type}")
    
    translator_class = translator_classes[translator_type]
    return translator_class(source_lang, target_lang, **kwargs)
