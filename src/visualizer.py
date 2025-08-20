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

"""Visualization module for translation evaluation results."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from .config import Config, PLOTS_DIR, SUB_PLOTS_DIR
from .logger import get_logger
from .utils import calculate_statistics

logger = get_logger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


class EvaluationVisualizer:
    """Class for creating evaluation visualizations."""
    
    def __init__(self, save_plots: bool = True, show_plots: bool = False):
        """
        Initialize visualizer.
        
        Args:
            save_plots: Whether to save plots to files
            show_plots: Whether to display plots
        """
        self.save_plots = save_plots or Config.SAVE_PLOTS
        self.show_plots = show_plots or Config.SHOW_PLOTS
        
        # Ensure directories exist
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        SUB_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized visualizer (save: {self.save_plots}, show: {self.show_plots})")
    
    def plot_domain_scores(
        self,
        domain_scores: Dict[str, List[float]],
        source_lang: str,
        target_lang: str,
        title_suffix: str = ""
    ) -> Optional[Path]:
        """
        Create box plot for domain-wise BLEU scores.
        
        Args:
            domain_scores: Dictionary mapping domains to score lists
            source_lang: Source language code
            target_lang: Target language code
            title_suffix: Additional title text
            
        Returns:
            Path to saved plot file if saved, None otherwise
        """
        if not domain_scores:
            logger.warning("No domain scores to plot")
            return None
        
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting
        domains = list(domain_scores.keys())
        scores = list(domain_scores.values())
        
        # Create box plot
        box_plot = plt.boxplot(
            scores,
            tick_labels=domains,
            showmeans=True,
            meanline=True,
            patch_artist=True
        )
        
        # Customize colors
        colors = sns.color_palette("husl", len(domains))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Calculate and display overall average
        all_scores = [score for score_list in scores for score in score_list]
        overall_avg = np.mean(all_scores) if all_scores else 0
        
        plt.axhline(y=overall_avg, color='red', linestyle='--', alpha=0.7, 
                   label=f'Overall Average: {overall_avg:.2f}')
        
        # Formatting
        plt.xlabel('Domain', fontsize=12)
        plt.ylabel('BLEU Score', fontsize=12)
        plt.title(f'BLEU Scores by Domain: {source_lang} → {target_lang}{title_suffix}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Add statistics text
        stats_text = f"Samples: {len(all_scores)}\nMean: {overall_avg:.2f}\nStd: {np.std(all_scores):.2f}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        plot_path = None
        if self.save_plots:
            plot_path = SUB_PLOTS_DIR / f"{source_lang}_{target_lang}_domains.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Domain scores plot saved: {plot_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        return plot_path
    
    def plot_language_pair_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        title_suffix: str = ""
    ) -> Optional[Path]:
        """
        Create comparison plot for multiple language pairs.
        
        Args:
            results: Dictionary mapping language pairs to results
            title_suffix: Additional title text
            
        Returns:
            Path to saved plot file if saved, None otherwise
        """
        if not results:
            logger.warning("No results to plot")
            return None
        
        # Prepare data
        language_pairs = []
        mean_scores = []
        all_scores_by_pair = {}
        
        for pair, result in results.items():
            if 'overall_statistics' in result and 'mean' in result['overall_statistics']:
                language_pairs.append(pair)
                mean_scores.append(result['overall_statistics']['mean'])
                
                # Collect individual scores for box plot
                if 'individual_results' in result:
                    scores = [r['bleu_score'] for r in result['individual_results']]
                    all_scores_by_pair[pair] = scores
        
        if not language_pairs:
            logger.warning("No valid language pair data to plot")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Bar plot of mean scores
        bars = ax1.bar(language_pairs, mean_scores, alpha=0.7, 
                      color=sns.color_palette("husl", len(language_pairs)))
        ax1.set_xlabel('Language Pair', fontsize=12)
        ax1.set_ylabel('Mean BLEU Score', fontsize=12)
        ax1.set_title(f'Mean BLEU Scores by Language Pair{title_suffix}', fontsize=14)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, mean_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{score:.1f}', ha='center', va='bottom')
        
        # Box plot of score distributions
        if all_scores_by_pair:
            box_data = [all_scores_by_pair.get(pair, []) for pair in language_pairs]
            box_plot = ax2.boxplot(box_data, tick_labels=language_pairs, 
                                  showmeans=True, patch_artist=True)
            
            # Color the boxes
            for patch, color in zip(box_plot['boxes'], sns.color_palette("husl", len(language_pairs))):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax2.set_xlabel('Language Pair', fontsize=12)
        ax2.set_ylabel('BLEU Score Distribution', fontsize=12)
        ax2.set_title(f'BLEU Score Distributions by Language Pair{title_suffix}', fontsize=14)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = None
        if self.save_plots:
            plot_path = PLOTS_DIR / f"language_pairs_comparison{title_suffix.replace(' ', '_')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Language pair comparison plot saved: {plot_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        return plot_path
    
    def plot_score_distribution(
        self,
        scores: List[float],
        title: str = "BLEU Score Distribution",
        bins: int = 30
    ) -> Optional[Path]:
        """
        Create histogram of score distribution.
        
        Args:
            scores: List of scores
            title: Plot title
            bins: Number of histogram bins
            
        Returns:
            Path to saved plot file if saved, None otherwise
        """
        if not scores:
            logger.warning("No scores to plot")
            return None
        
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        plt.hist(scores, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add statistics lines
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        
        plt.axvline(mean_score, color='red', linestyle='--', 
                   label=f'Mean: {mean_score:.2f}')
        plt.axvline(median_score, color='green', linestyle='--', 
                   label=f'Median: {median_score:.2f}')
        
        # Formatting
        plt.xlabel('BLEU Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats = calculate_statistics(scores)
        stats_text = f"Count: {stats['count']}\nStd: {stats['std_dev']:.2f}\nMin: {stats['min']:.2f}\nMax: {stats['max']:.2f}"
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = None
        if self.save_plots:
            safe_title = title.replace(' ', '_').replace(':', '').lower()
            plot_path = PLOTS_DIR / f"{safe_title}_distribution.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Score distribution plot saved: {plot_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        return plot_path
    
    def create_evaluation_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Create comprehensive evaluation report with multiple visualizations.
        
        Args:
            results: Evaluation results dictionary
            output_path: Output directory path
            
        Returns:
            Path to report directory
        """
        if output_path is None:
            output_path = PLOTS_DIR / "evaluation_report"
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating evaluation report in: {output_path}")
        
        # Overall score distribution
        if 'individual_results' in results:
            all_scores = [r['bleu_score'] for r in results['individual_results']]
            if all_scores:
                self.plot_score_distribution(
                    scores=all_scores,
                    title="Overall BLEU Score Distribution"
                )
        
        # Domain-wise analysis if available
        if 'individual_results' in results:
            domain_scores = {}
            for result in results['individual_results']:
                domain = result.get('domain', 'unknown')
                if domain not in domain_scores:
                    domain_scores[domain] = []
                domain_scores[domain].append(result['bleu_score'])
            
            if domain_scores:
                source_lang = results.get('source_lang', 'unknown')
                target_lang = results.get('target_lang', 'unknown')
                self.plot_domain_scores(domain_scores, source_lang, target_lang)
        
        # Language pair comparison if multiple pairs
        if 'pair_results' in results:
            self.plot_language_pair_comparison(results['pair_results'])
        
        logger.info("Evaluation report created successfully")
        return output_path


# Legacy function compatibility
def plot_bleu_score(
    bleu_scores: Dict[str, List[float]],
    source_lang: str,
    target_lang: str
) -> None:
    """
    Legacy function for backward compatibility.
    
    Args:
        bleu_scores: Dictionary mapping domains to score lists
        source_lang: Source language code
        target_lang: Target language code
    """
    visualizer = EvaluationVisualizer()
    visualizer.plot_domain_scores(bleu_scores, source_lang, target_lang)
