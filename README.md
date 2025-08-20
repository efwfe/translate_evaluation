# Translation Evaluation Framework

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive, production-ready evaluation framework for machine translation systems using BLEU scores across multiple domains and language pairs.

## 🌟 Features

- **Multi-language Support**: Evaluate translations between any pair of 7 supported languages (Chinese, English, Japanese, French, Italian, Spanish, Portuguese)
- **Domain-specific Evaluation**: Test translation quality across different specialized domains (technology, business, travel, education, healthcare, legal)
- **Multiple Evaluation Metrics**: Industry-standard automatic evaluation using SacreBLEU with language-specific tokenization
- **Comprehensive Logging**: Structured logging with multiple levels and file/console output
- **CSV Export**: Export detailed results and statistics to CSV files for further analysis
- **Rich Visualizations**: Generate comprehensive plots and charts for performance analysis
- **Flexible Translation Interface**: Easy integration with any translation system
- **Apache 2.0 Licensed**: Open source with permissive licensing

## 🚀 Quick Start

### Basic Usage


#### Python API

```python
from src.evaluator import TranslationEvaluator
from src.translator import create_translator
from src.visualizer import EvaluationVisualizer

# Create evaluator
evaluator = TranslationEvaluator()
evaluator.load_evaluation_data()

# Create translator
translator = create_translator(
    translator_type='mock',  # or 'llama'
    source_lang='zh',
    target_lang='en'
)

# Evaluate
results = evaluator.evaluate_translator(translator, max_samples=100)

# Visualize results
visualizer = EvaluationVisualizer()
visualizer.create_evaluation_report(results)

# Save results
evaluator.save_results(results, "my_evaluation")
```

#### Command Line Interface

The framework provides a comprehensive command-line interface for easy evaluation:

```bash
# Evaluate single language pair with mock translator (for testing)
python -m src.main --source zh --target en --translator qwen --max-samples 10

# Evaluate with Llama model
python -m src.main --source zh --target en --translator llama --model-path /path/to/model.gguf
python -m src.main --source zh --target en --translator qwen --model-path /path/to/model.gguf
# Advanced usage with custom settings
python -m src.main \
  --source zh --target en \
  --translator llama \
  --model-path /path/to/model.gguf \
  --max-samples 100 \
  --output-prefix "my_evaluation" \
  --show-plots \
  --log-level DEBUG

```

**Common CLI Options:**
- `--source`, `-s`: Source language code (zh, en, ja, fr, it, es, pt)
- `--target`, `-t`: Target language code (required with --source)
- `--all-pairs`, `-a`: Evaluate all possible language pairs
- `--pairs`, `-p`: Specify custom language pairs (e.g., "zh,en en,fr")
- `--translator`: Choose translator type (`mock`, `llama`)
- `--model-path`: Path to model file (required for Llama translator)
- `--max-samples`: Limit number of evaluation samples
- `--output-prefix`: Custom prefix for output files
- `--show-plots`: Display plots interactively
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--verbose`, `-v`: Enable verbose output

## 📊 Output and Results

The framework generates multiple types of output:

### 1. CSV Files
- **Detailed Results**: Individual translation scores with metadata
- **Summary Statistics**: Aggregated metrics by domain and language pair
- **Evaluation Metadata**: Timestamps, configuration, and system information

### 2. Visualizations
- **Domain Score Distributions**: Box plots showing score ranges by domain
- **Language Pair Comparisons**: Comparative analysis across different language pairs
- **Score Histograms**: Distribution analysis of BLEU scores
- **Comprehensive Reports**: Multi-plot evaluation summaries

### 3. Structured Logs
- **Evaluation Progress**: Real-time progress tracking with timestamps
- **Error Handling**: Detailed error messages and stack traces
- **Performance Metrics**: Translation times and system resource usage
- **Statistical Summaries**: Automatic calculation of means, medians, and standard deviations

## 🔧 Configuration

### Environment Variables

```bash
# Model configuration
export MODEL_PATH="/path/to/your/model.gguf"
export USE_GPU="true"

# Logging configuration
export LOG_LEVEL="INFO"

# Output configuration
export SAVE_PLOTS="true"
export SAVE_CSV="true"
export SHOW_PLOTS="false"

# Evaluation configuration
export BATCH_SIZE="1"
export PARALLEL_WORKERS="1"
```

### Configuration File

Create a `.env` file in the project root:

```env
MODEL_PATH=/path/to/model.gguf
LOG_LEVEL=INFO
SAVE_PLOTS=true
SAVE_CSV=true
USE_GPU=false
```

## 🏗️ Project Structure

```
translation-evaluation-framework/
├── src/                          # Source code
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration management
│   ├── evaluator.py             # Core evaluation logic
│   ├── logger.py                # Logging utilities
│   ├── main.py                  # CLI entry point
│   ├── translator.py            # Translation interfaces
│   ├── utils.py                 # Utility functions
│   └── visualizer.py            # Visualization tools
├── data/                        # Evaluation datasets
│   └── translation.json         # Multi-language test data
├── results/                     # Evaluation results (auto-created)
├── logs/                        # Log files (auto-created)
├── plots/                       # Generated visualizations (auto-created)
├── LICENSE                      # Apache 2.0 License
├── README.md                    # This file
├── pyproject.toml              # Project configuration
└── requirements.txt            # Python dependencies
```

## 🔌 Implementing Custom Translators

### Basic Translator

```python
from src.translator import BaseTranslator

class MyTranslator(BaseTranslator):
    def __init__(self, source_lang: str, target_lang: str, **kwargs):
        super().__init__(source_lang, target_lang)
        # Initialize your translation model here
        
    def translate(self, text: str) -> str:
        # Implement your translation logic
        return your_translation_result
```

### Using External APIs

```python
import openai
from src.translator import BaseTranslator

class OpenAITranslator(BaseTranslator):
    def __init__(self, source_lang: str, target_lang: str, api_key: str):
        super().__init__(source_lang, target_lang)
        openai.api_key = api_key
        
    def translate(self, text: str) -> str:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Translate from {self.source_lang} to {self.target_lang}"},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content
```

## 📈 Evaluation Metrics

### BLEU Score Calculation

The framework uses [SacreBLEU](https://github.com/mjpost/sacrebleu) with language-specific tokenization:

- **Chinese/Japanese**: Character-level tokenization (`char`)
- **Romance Languages** (French, Italian, Spanish, Portuguese): International tokenization (`intl`)
- **Other Languages**: Default tokenization (`13a`)

### Statistical Measures

For each evaluation, the framework calculates:

- **Mean BLEU Score**: Average performance across all samples
- **Median**: Middle value for robust central tendency
- **Standard Deviation**: Measure of score variability
- **Quartiles**: Q1, Q3 for distribution analysis
- **Min/Max**: Range of performance
- **Sample Count**: Number of evaluated translations

## 🛠️ Development

### Setting up Development Environment

```bash
# Clone and install in development mode
git clone https://github.com/example/translation-evaluation-framework.git
cd translation-evaluation-framework
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/
isort src/

# Type checking
mypy src/

# Linting
flake8 src/
```

### Adding New Languages

1. Update `SUPPORTED_LANGUAGES` in `src/config.py`
2. Add tokenization rules in `auto_sentence_bleu()` method
3. Update language name mapping in translator classes
4. Add test data for the new language in `data/translation.json`

### Adding New Evaluation Metrics

1. Implement metric calculation in `src/evaluator.py`
2. Update result structures to include new metrics
3. Add visualization support in `src/visualizer.py`
4. Update CSV export to include new fields

## 📝 Data Format

The evaluation data should be in JSON format:

```json
[
    {
        "id": "tech_01",
        "domain": "tech",
        "zh": "人工智能正在推动医疗诊断的进步。",
        "en": "Artificial intelligence is driving progress in medical diagnostics.",
        "ja": "人工知能は医療診断の進歩を推進しています。",
        "fr": "L'intelligence artificielle stimule les progrès du diagnostic médical.",
        "it": "L'intelligenza artificiale sta spingendo i progressi nella diagnostica medica.",
        "es": "La inteligencia artificial está impulsando el progreso en el diagnóstico médico.",
        "pt": "A inteligência artificial está impulsionando os avanços no diagnóstico médico."
    }
]
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [SacreBLEU](https://github.com/mjpost/sacrebleu) for standardized BLEU score calculation
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for efficient model inference
- The machine translation research community for evaluation standards and best practices

## 📞 Support

- 📧 Email: team@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/example/translation-evaluation-framework/issues)
- 📖 Documentation: [GitHub Wiki](https://github.com/example/translation-evaluation-framework/wiki)

## 🗺️ Roadmap

- [ ] Support for additional evaluation metrics (METEOR, BERTScore)
- [ ] Web interface for easy evaluation management
- [ ] Integration with popular translation APIs
- [ ] Parallel evaluation for improved performance
- [ ] Custom domain support with user-provided datasets
- [ ] Real-time evaluation monitoring and alerts