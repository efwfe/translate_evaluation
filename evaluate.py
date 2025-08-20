from config import DATA_DIR
from config import PLOTS_DIR
from utils import load_data, save_data
from typing import Callable
import sacrebleu
import matplotlib.pyplot as plt
import itertools

def evaluate_translation(source_lang, target_lang, translate:Callable):
    data = load_data(DATA_DIR / "translation.json")
    domains = set([item['domain'] for item in data])
    bleu_scores = {}
    for domain in domains:
        bleu_scores[domain] = []

    for item in data:
        source_text = item[source_lang]
        translation_text = translate(source_text)
        target_text = item[target_lang]
        bleu = sacrebleu.sentence_bleu(translation_text, [target_text])
        bleu_scores[item['domain']].append(bleu.score)
    return bleu_scores

def get_avg_score(bleu_scores):
    avg_scores = [sum(scores) / len(scores) for scores in bleu_scores.values()]
    return sum(avg_scores) / len(avg_scores)


def get_all_domains():
    langs = ['zh', 'en', 'ja', 'fr', 'it', 'es', 'pt']
    return list(itertools.combinations(langs, 2))

def plot_bleu_score(bleu_scores, source_lang, target_lang):
    plt.figure(figsize=(12, 6))
    plt.boxplot(bleu_scores.values(), tick_labels=bleu_scores.keys(), showmeans=True, medianprops=dict(color='red'))
    plt.xlabel('Domain')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score by Domain')
    avg_score = get_avg_score(bleu_scores)
    plt.text(0.5, 0.95, f'Average BLEU Score: {avg_score:.2f}', transform=plt.gca().transAxes, ha='center', va='top')
    plt.savefig(PLOTS_DIR / f"{source_lang}_{target_lang}.png")
    plt.show()
    plt.close()

def plot_models_bleu_scores(models_scores):
    plt.figure(figsize=(36, 8))
 
    plt.boxplot(models_scores.values(), tick_labels=models_scores.keys(), showmeans=True, medianprops=dict(color='red'), meanprops={"marker":"^","markerfacecolor":"green","markeredgecolor":"black"})

    plt.xlabel('Mode')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score by Model')
    plt.savefig(PLOTS_DIR / "modes_bleu_scores.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    domains = get_all_domains()
    domain_scores = {}
    for domain in domains:
        bleu_scores = evaluate_translation(domain[0], domain[1], lambda x: x)
        domain_scores['->'.join(domain)] = list(itertools.chain(*bleu_scores.values()))
    print(domain_scores)
    plot_models_bleu_scores(domain_scores)