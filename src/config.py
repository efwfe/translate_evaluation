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

"""Configuration management for translation evaluation framework."""

import os
import pathlib
from typing import Optional


# Base directories
BASE_DIR = pathlib.Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# Sub-directories
SUB_PLOTS_DIR = PLOTS_DIR / "individual"

# Create directories if they don't exist
for directory in [PLOTS_DIR, SUB_PLOTS_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class Config:
    """Configuration class for translation evaluation."""
    
    # Supported languages
    SUPPORTED_LANGUAGES = ['zh', 'en', 'ja', 'fr', 'it', 'es', 'pt']
    
    # Supported domains
    SUPPORTED_DOMAINS = ['tech', 'business', 'travel', 'education', 'healthcare', 'legal']
    
    # Evaluation settings
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_TOKENS = 2048
    DEFAULT_CONTEXT_WINDOW = 2048
    
    # Logging settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', '')
    USE_GPU = os.getenv('USE_GPU', 'false').lower() == 'true'
    
    # Output settings
    SAVE_PLOTS = os.getenv('SAVE_PLOTS', 'true').lower() == 'true'
    SAVE_CSV = os.getenv('SAVE_CSV', 'true').lower() == 'true'
    SHOW_PLOTS = os.getenv('SHOW_PLOTS', 'false').lower() == 'true'
    
    # Evaluation settings
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '1'))
    PARALLEL_WORKERS = int(os.getenv('PARALLEL_WORKERS', '1'))
    
    @classmethod
    def get_model_config(cls) -> dict:
        """Get model configuration."""
        return {
            'model_path': cls.MODEL_PATH,
            'n_ctx': cls.DEFAULT_CONTEXT_WINDOW,
            'chat_format': 'chatml',
            'verbose': False,
            'n_gpu_layers': -1 if cls.USE_GPU else 0,
        }
    
    @classmethod
    def validate_language(cls, language: str) -> bool:
        """Validate if language is supported."""
        return language.lower() in cls.SUPPORTED_LANGUAGES
    
    @classmethod
    def validate_domain(cls, domain: str) -> bool:
        """Validate if domain is supported."""
        return domain.lower() in cls.SUPPORTED_DOMAINS


# Global configuration instance
config = Config()
