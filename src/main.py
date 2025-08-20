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

"""Main entry point for translation evaluation framework."""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

from .config import Config
from .evaluator import TranslationEvaluator, get_all_language_pairs
from .logger import get_logger
from .translator import create_translator,translator_classes
from .visualizer import EvaluationVisualizer

logger = get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Translation Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single language pair with Llama translator
  python -m src.main --source zh --target en --translator llama --model-path /path/to/model.gguf
  
  # Evaluate all language pairs with mock translator
  python -m src.main --all-pairs --translator mock --max-samples 100
  
  # Evaluate specific pairs
  python -m src.main --pairs zh,en en,fr fr,es --translator llama
        """
    )
    
    # Language configuration
    lang_group = parser.add_mutually_exclusive_group(required=True)
    lang_group.add_argument(
        '--source', '-s',
        help='Source language code'
    )
    lang_group.add_argument(
        '--all-pairs', '-a',
        action='store_true',
        help='Evaluate all possible language pairs'
    )
    lang_group.add_argument(
        '--pairs', '-p',
        help='Comma-separated list of language pairs (e.g., "zh,en en,fr")'
    )
    
    parser.add_argument(
        '--target', '-t',
        help='Target language code (required with --source)'
    )
    
    # Translator configuration
    parser.add_argument(
        '--translator',
        choices=list(translator_classes.keys()),
        default='llama',
        help='Translator type to use'
    )
    
    parser.add_argument(
        '--model-path',
        help='Path to model file (for Llama translator)'
    )
    
    # Evaluation configuration
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to evaluate per language pair'
    )
    
    parser.add_argument(
        '--data-path',
        help='Path to evaluation data file'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-prefix',
        default='evaluation_results',
        help='Prefix for output files'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to files'
    )
    
    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='Display plots interactively'
    )
    
    # Logging configuration
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output (equivalent to --log-level DEBUG)'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Check source/target combination
    if args.source and not args.target:
        raise ValueError("--target is required when using --source")
    
    # Validate language codes
    if args.source and not Config.validate_language(args.source):
        raise ValueError(f"Unsupported source language: {args.source}")
    
    if args.target and not Config.validate_language(args.target):
        raise ValueError(f"Unsupported target language: {args.target}")
    
    # Validate model path for Llama translator
    if args.translator == 'llama':
        model_path = args.model_path or Config.MODEL_PATH
        if not model_path:
            raise ValueError("Model path is required for Llama translator. Use --model-path or set MODEL_PATH environment variable.")
        if not Path(model_path).exists():
            raise ValueError(f"Model file not found: {model_path}")
    
    # Validate data path
    if args.data_path and not Path(args.data_path).exists():
        raise ValueError(f"Data file not found: {args.data_path}")


def parse_language_pairs(pairs_str: str) -> List[Tuple[str, str]]:
    """
    Parse language pairs string.
    
    Args:
        pairs_str: Comma-separated language pairs (e.g., "zh,en en,fr")
        
    Returns:
        List of (source, target) tuples
    """
    pairs = []
    for pair_str in pairs_str.split():
        parts = pair_str.split(',')
        if len(parts) != 2:
            raise ValueError(f"Invalid language pair format: {pair_str}. Use 'source,target' format.")
        
        source, target = parts
        if not Config.validate_language(source):
            raise ValueError(f"Unsupported source language: {source}")
        if not Config.validate_language(target):
            raise ValueError(f"Unsupported target language: {target}")
        
        pairs.append((source, target))
    
    return pairs


def main() -> int:
    """Main entry point."""
    try:
        args = parse_arguments()
        
        # Configure logging
        if args.verbose:
            args.log_level = 'DEBUG'
        
        # Set log level in config
        Config.LOG_LEVEL = args.log_level
        
        # Validate arguments
        validate_arguments(args)
        
        logger.info("Starting translation evaluation")
        logger.info(f"Arguments: {vars(args)}")
        
        # Initialize evaluator
        evaluator = TranslationEvaluator(data_path=args.data_path)
        evaluator.load_evaluation_data()
        
        # Initialize visualizer
        visualizer = EvaluationVisualizer(
            save_plots=not args.no_save,
            show_plots=args.show_plots
        )
        
        # Determine language pairs to evaluate
        if args.all_pairs:
            language_pairs = get_all_language_pairs()
            logger.info(f"Evaluating all {len(language_pairs)} language pairs")
        elif args.pairs:
            language_pairs = parse_language_pairs(args.pairs)
            logger.info(f"Evaluating {len(language_pairs)} specified language pairs")
        else:
            language_pairs = [(args.source, args.target)]
            logger.info(f"Evaluating single language pair: {args.source} -> {args.target}")
        
        # Evaluate based on number of language pairs
        if len(language_pairs) == 1:
            # Single language pair evaluation
            source_lang, target_lang = language_pairs[0]
            
            # Create translator
            translator = create_translator(
                translator_type=args.translator,
                source_lang=source_lang,
                target_lang=target_lang,
                model_path=args.model_path
            )
            
            # Evaluate
            results = evaluator.evaluate_translator(
                translator=translator,
                max_samples=args.max_samples
            )
            
            # Visualize
            if results and 'domain_statistics' in results:
                domain_scores = {}
                for result in results.get('individual_results', []):
                    domain = result.get('domain', 'unknown')
                    if domain not in domain_scores:
                        domain_scores[domain] = []
                    domain_scores[domain].append(result['bleu_score'])
                
                if domain_scores:
                    visualizer.plot_domain_scores(domain_scores, source_lang, target_lang)
            
        else:
            # Multiple language pairs evaluation
            def multi_translate_func(text: str, source_lang: str, target_lang: str) -> str:
                translator = create_translator(
                    translator_type=args.translator,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    model_path=args.model_path
                )
                return translator.translate(text)
            
            # Evaluate all pairs
            results = evaluator.evaluate_multiple_language_pairs(
                language_pairs=language_pairs,
                translate_func=multi_translate_func,
                max_samples=args.max_samples
            )
            
            # Visualize
            if results and 'pair_results' in results:
                visualizer.plot_language_pair_comparison(results['pair_results'])
        
        # Save results
        if not args.no_save and results:
            evaluator.save_results(results, args.output_prefix)
            
            # Create comprehensive report
            visualizer.create_evaluation_report(results)
        
        # Print summary
        if results:
            if 'overall_statistics' in results:
                stats = results['overall_statistics']
                logger.info(f"Evaluation completed successfully!")
                logger.info(f"Total samples: {stats.get('count', 0)}")
                logger.info(f"Average BLEU score: {stats.get('mean', 0):.2f}")
                logger.info(f"Standard deviation: {stats.get('std_dev', 0):.2f}")
            else:
                logger.info("Evaluation completed successfully!")
        else:
            logger.warning("No results generated")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose if 'args' in locals() else False:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
