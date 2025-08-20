# Translation Evaluation Framework

A comprehensive evaluation framework for machine translation systems using BLEU scores across multiple domains and language pairs.

## Overview

This project provides a standardized evaluation pipeline for translation functions, supporting 7 languages (Chinese, English, Japanese, French, Italian, Spanish, Portuguese) across various domains including technology, business, travel, education, healthcare, and legal.

## Features

- **Multi-language Support**: Evaluate translations between any pair of 7 supported languages
- **Domain-specific Evaluation**: Test translation quality across different specialized domains
- **BLEU Score Metrics**: Industry-standard automatic evaluation using SacreBLEU
- **Visual Analytics**: Generate comprehensive plots and charts for performance analysis
- **Flexible Translation Interface**: Easy integration with any translation function

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd evaluate

# Install dependencies
pip install -r requirements.txt
# or using uv
uv sync
```

### Basic Usage

The core evaluation function requires implementing a translation function that takes a source text string and returns a translated string:

```python
from evaluate import evaluate_translation, plot_bleu_score

# Implement your translation function
def my_translate_function(source_text: str) -> str:
    """
    Your translation implementation here.
    
    Args:
        source_text (str): Text to translate
        
    Returns:
        str: Translated text
    """
    # Example: Using a simple API call or model inference
    translated = your_translation_model.translate(source_text)
    return translated

# Evaluate your translation function
bleu_scores = evaluate_translation('zh', 'en', my_translate_function)

# Generate visualization
plot_bleu_score(bleu_scores, 'zh', 'en')
```

## Translation Function Implementation Guide

### Function Signature

Your translation function must follow this signature:

```python
def translate(source_text: str) -> str:
    """
    Translate source text to target text.
    
    Args:
        source_text (str): Input text in source language
        
    Returns:
        str: Translated text in target language
    """
    pass
```

### Implementation Examples

#### 1. Using Google Translate API

```python
from googletrans import Translator

def google_translate(source_text: str) -> str:
    translator = Translator()
    result = translator.translate(source_text, src='zh', dest='en')
    return result.text
```

#### 2. Using Hugging Face Transformers

```python
from transformers import MarianMTModel, MarianTokenizer

class HuggingFaceTranslator:
    def __init__(self, model_name: str):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
    
    def translate(self, source_text: str) -> str:
        inputs = self.tokenizer(source_text, return_tensors="pt", padding=True)
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage
translator = HuggingFaceTranslator("Helsinki-NLP/opus-mt-zh-en")
bleu_scores = evaluate_translation('zh', 'en', translator.translate)
```

#### 3. Using OpenAI API

```python
import openai

def openai_translate(source_text: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional translator. Translate the following Chinese text to English. Return only the translation."},
            {"role": "user", "content": source_text}
        ]
    )
    return response.choices[0].message.content.strip()
```

## Evaluation Metrics

### BLEU Score

The framework uses [SacreBLEU](https://github.com/mjpost/sacrebleu) for standardized BLEU score calculation:

- **Sentence-level BLEU**: Calculated for each translation pair
- **Domain-level aggregation**: Scores grouped by domain (tech, business, travel, etc.)
- **Overall average**: Mean BLEU score across all domains

### Supported Language Pairs

The framework supports evaluation between any pair of these languages:
- Chinese (zh)
- English (en) 
- Japanese (ja)
- French (fr)
- Italian (it)
- Spanish (es)
- Portuguese (pt)

Total: 21 unique language pairs (7 choose 2)

## Data Structure

### Translation Dataset

The evaluation dataset (`data/translation.json`) contains structured translation examples:

```json
{
    "id": "tech_01",
    "domain": "tech",
    "zh": "人工智能正在推动医疗诊断的进步。",
    "en": "Artificial intelligence is driving progress in medical diagnostics.",
    "ja": "人工知能は医療診断の進歩を推進しています。",
    "fr": "L'intelligence artificielle stimule les progrès du diagnostic médical.",
    "it": "L'intelligenza artificiale sta spingendo i progressi nella diagnostica medica.",
    "es": "La inteligencia artificial está impulsando el progreso en el diagnóstico médico.",
    "pt": "A inteligência artificial está impulsionando o progresso no diagnóstico médico."
}
```

### Domains

Current dataset includes examples from:
- **Technology**: AI, software, hardware
- **Business**: Finance, marketing, management
- **Travel**: Tourism, transportation, hospitality
- **Education**: Academic, learning, research
- **Healthcare**: Medical, pharmaceutical, wellness
- **Legal**: Law, regulations, compliance

## Advanced Usage

### Batch Evaluation Across All Language Pairs

```python
from evaluate import get_all_domains, evaluate_translation, plot_models_bleu_scores
import itertools

def evaluate_all_pairs(translate_function):
    domains = get_all_domains()  # Gets all language pair combinations
    results = {}
    
    for source_lang, target_lang in domains:
        bleu_scores = evaluate_translation(source_lang, target_lang, translate_function)
        # Flatten scores across domains
        all_scores = list(itertools.chain(*bleu_scores.values()))
        results[f'{source_lang}->{target_lang}'] = all_scores
    
    return results

# Evaluate and visualize
results = evaluate_all_pairs(my_translate_function)
plot_models_bleu_scores(results)
```

### Model Comparison

```python
def compare_translation_models():
    models = {
        'Google Translate': google_translate,
        'GPT-3.5': openai_translate,
        'MarianMT': huggingface_translate
    }
    
    comparison_results = {}
    for model_name, translate_func in models.items():
        bleu_scores = evaluate_translation('zh', 'en', translate_func)
        avg_score = get_avg_score(bleu_scores)
        comparison_results[model_name] = avg_score
    
    return comparison_results
```

### Custom Evaluation Pipeline

```python
def custom_evaluation_pipeline(translate_func, source_lang='zh', target_lang='en'):
    # Run evaluation
    bleu_scores = evaluate_translation(source_lang, target_lang, translate_func)
    
    # Calculate metrics
    avg_score = get_avg_score(bleu_scores)
    domain_averages = {domain: sum(scores)/len(scores) 
                      for domain, scores in bleu_scores.items()}
    
    # Generate plots
    plot_bleu_score(bleu_scores, source_lang, target_lang)
    
    # Return comprehensive results
    return {
        'overall_average': avg_score,
        'domain_averages': domain_averages,
        'raw_scores': bleu_scores
    }
```

## File Structure

```
evaluate/
├── config.py              # Configuration and paths
├── evaluate.py             # Core evaluation functions
├── utils.py               # Utility functions for data handling
├── translate.py           # Translation function template
├── data/
│   └── translation.json   # Evaluation dataset
├── plots/                 # Generated visualization outputs
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Configuration

The `config.py` file defines project paths:

```python
import pathlib

BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
```

## Dependencies

Key dependencies include:
- `sacrebleu`: BLEU score calculation
- `matplotlib`: Visualization and plotting
- `transformers`: Hugging Face model support
- `torch`: PyTorch for deep learning models
- `datasets`: Dataset handling utilities

See `requirements.txt` for complete dependency list.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your translation function following the interface guidelines
4. Add tests and documentation
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@software{translation_evaluation_framework,
  title={Translation Evaluation Framework},
  author={[Your Name]},
  year={2024},
  url={[Your Repository URL]}
}
```