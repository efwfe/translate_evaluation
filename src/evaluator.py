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

"""Evaluation module for translation quality assessment."""

import itertools
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Any

import sacrebleu
from tqdm import tqdm

from .config import Config, DATA_DIR, RESULTS_DIR
from .logger import get_logger
from .translator import BaseTranslator
from .utils import load_data, save_to_csv, create_evaluation_summary, calculate_statistics

logger = get_logger(__name__)


class TranslationEvaluator:
    """Main class for evaluating translation quality."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            data_path: Path to translation data file
        """
        self.data_path = data_path or (DATA_DIR / "translation.json")
        self.data = None
        self.results = {}
        
        logger.info(f"Initialized TranslationEvaluator with data path: {self.data_path}")
    
    def load_evaluation_data(self) -> None:
        """Load evaluation data from file."""
        try:
            self.data = load_data(self.data_path)
            logger.info(f"Loaded {len(self.data)} evaluation samples")
            
            # Log data statistics
            domains = set(item.get('domain', 'unknown') for item in self.data)
            languages = set()
            for item in self.data:
                for key in item.keys():
                    if len(key) == 2 and key.isalpha():
                        languages.add(key)
            
            logger.info(f"Found domains: {sorted(domains)}")
            logger.info(f"Found languages: {sorted(languages)}")
            
        except Exception as e:
            logger.error(f"Failed to load evaluation data: {e}")
            raise
    
    def auto_sentence_bleu(self, hypothesis: str, reference: str, target_lang: str) -> float:
        """
        Calculate BLEU score with automatic tokenization based on language.
        
        Args:
            hypothesis: Generated translation
            reference: Reference translation
            target_lang: Target language code
            
        Returns:
            BLEU score
        """
        target_lang = target_lang.lower()
        
        # Choose tokenization method based on language
        if target_lang in ["zh", "ja"]:  # Chinese, Japanese
            tokenize = "char"  # Character-level tokenization
        elif target_lang in ["fr", "it", "es", "pt"]:  # Romance languages
            tokenize = "intl"  # International tokenization (handles accents)
        else:
            tokenize = "13a"  # Default tokenization
        
        try:
            bleu = sacrebleu.sentence_bleu(hypothesis, [reference], tokenize=tokenize)
            return bleu.score
        except Exception as e:
            logger.error(f"BLEU calculation failed: {e}")
            return 0.0
    
    def evaluate_single_translation(
        self,
        source_text: str,
        translated_text: str,
        reference_text: str,
        source_lang: str,
        target_lang: str,
        domain: str,
        item_id: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single translation.
        
        Args:
            source_text: Original source text
            translated_text: Generated translation
            reference_text: Reference translation
            source_lang: Source language code
            target_lang: Target language code
            domain: Domain category
            item_id: Unique identifier
            
        Returns:
            Evaluation result dictionary
        """
        bleu_score = self.auto_sentence_bleu(translated_text, reference_text, target_lang)
        
        result = {
            'id': item_id,
            'source_lang': source_lang,
            'target_lang': target_lang,
            'domain': domain,
            'source_text': source_text,
            'translated_text': translated_text,
            'reference_text': reference_text,
            'bleu_score': bleu_score,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.debug(f"Evaluated {item_id}: BLEU={bleu_score:.2f}")
        
        return result
    
    def evaluate_translation_function(
        self,
        source_lang: str,
        target_lang: str,
        translate_func: Callable[[str], str],
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a translation function.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            translate_func: Translation function
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Evaluation results
        """
        if self.data is None:
            self.load_evaluation_data()
        
        logger.info(f"Starting evaluation: {source_lang} -> {target_lang}")
        
        # Filter applicable data
        applicable_data = []
        for item in self.data:
            if source_lang in item and target_lang in item:
                applicable_data.append(item)
        
        if not applicable_data:
            logger.warning(f"No data found for language pair: {source_lang} -> {target_lang}")
            return {}
        
        if max_samples:
            applicable_data = applicable_data[:max_samples]
        
        logger.info(f"Evaluating {len(applicable_data)} samples")
        
        # Initialize results structure
        domain_scores = {}
        individual_results = []
        
        # Evaluate each sample
        for item in tqdm(applicable_data, desc=f"Evaluating {source_lang}->{target_lang}"):
            try:
                source_text = item[source_lang]
                reference_text = item[target_lang]
                domain = item.get('domain', 'unknown')
                item_id = item.get('id', f"item_{len(individual_results)}")
                
                # Get translation
                start_time = time.time()
                translated_text = translate_func(source_text)
                translation_time = time.time() - start_time
                
                # Evaluate translation
                result = self.evaluate_single_translation(
                    source_text=source_text,
                    translated_text=translated_text,
                    reference_text=reference_text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    domain=domain,
                    item_id=item_id
                )
                
                result['translation_time'] = translation_time
                individual_results.append(result)
                
                # Group by domain
                if domain not in domain_scores:
                    domain_scores[domain] = []
                domain_scores[domain].append(result['bleu_score'])
                
            except Exception as e:
                logger.error(f"Error evaluating item {item.get('id', 'unknown')}: {e}")
                continue
        
        # Calculate statistics
        overall_scores = [r['bleu_score'] for r in individual_results]
        overall_stats = calculate_statistics(overall_scores)
        
        domain_stats = {}
        for domain, scores in domain_scores.items():
            domain_stats[domain] = calculate_statistics(scores)
        
        results = {
            'language_pair': f"{source_lang}->{target_lang}",
            'source_lang': source_lang,
            'target_lang': target_lang,
            'total_samples': len(individual_results),
            'overall_statistics': overall_stats,
            'domain_statistics': domain_stats,
            'individual_results': individual_results,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Evaluation completed. Overall BLEU: {overall_stats.get('mean', 0):.2f}")
        
        return results
    
    def evaluate_translator(
        self,
        translator: BaseTranslator,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a translator instance.
        
        Args:
            translator: Translator instance
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Evaluation results
        """
        return self.evaluate_translation_function(
            source_lang=translator.source_lang,
            target_lang=translator.target_lang,
            translate_func=translator.translate,
            max_samples=max_samples
        )
    
    def evaluate_multiple_language_pairs(
        self,
        language_pairs: List[Tuple[str, str]],
        translate_func: Callable[[str, str, str], str],  # (text, source_lang, target_lang) -> str
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple language pairs.
        
        Args:
            language_pairs: List of (source_lang, target_lang) tuples
            translate_func: Translation function that accepts source and target languages
            max_samples: Maximum number of samples per language pair
            
        Returns:
            Combined evaluation results
        """
        all_results = {}
        combined_individual_results = []
        
        for source_lang, target_lang in language_pairs:
            logger.info(f"Evaluating language pair: {source_lang} -> {target_lang}")
            
            # Create wrapper function for this language pair
            def pair_translate_func(text: str) -> str:
                return translate_func(text, source_lang, target_lang)
            
            # Evaluate this language pair
            pair_results = self.evaluate_translation_function(
                source_lang=source_lang,
                target_lang=target_lang,
                translate_func=pair_translate_func,
                max_samples=max_samples
            )
            
            if pair_results:
                pair_key = f"{source_lang}->{target_lang}"
                all_results[pair_key] = pair_results
                combined_individual_results.extend(pair_results.get('individual_results', []))
        
        # Calculate overall statistics
        if combined_individual_results:
            overall_scores = [r['bleu_score'] for r in combined_individual_results]
            overall_stats = calculate_statistics(overall_scores)
        else:
            overall_stats = {}
        
        combined_results = {
            'language_pairs': [f"{s}->{t}" for s, t in language_pairs],
            'total_samples': len(combined_individual_results),
            'overall_statistics': overall_stats,
            'pair_results': all_results,
            'individual_results': combined_individual_results,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        return combined_results
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_prefix: str = "evaluation_results"
    ) -> None:
        """
        Save evaluation results to files.
        
        Args:
            results: Evaluation results dictionary
            output_prefix: Output file prefix
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = RESULTS_DIR / f"{output_prefix}_{timestamp}"
        
        # Save summary
        create_evaluation_summary(results, base_path)
        
        # Save detailed individual results if available
        if 'individual_results' in results and results['individual_results']:
            csv_path = base_path.with_name(f"{base_path.name}_detailed.csv")
            save_to_csv(results['individual_results'], csv_path)
        
        logger.info(f"Results saved with prefix: {base_path}")


def get_all_language_pairs(languages: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """
    Get all possible language pairs.
    
    Args:
        languages: List of language codes (defaults to supported languages)
        
    Returns:
        List of (source, target) language pairs
    """
    if languages is None:
        languages = Config.SUPPORTED_LANGUAGES
    
    return list(itertools.combinations(languages, 2))
