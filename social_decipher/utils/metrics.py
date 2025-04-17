from typing import Any

import nltk
import torch
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

nltk.download("punkt")


def compute_bleu(reference: str, hypothesis: str) -> float:
    try:
        reference_tokens = nltk.word_tokenize(reference.lower())
        hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
        smoothie = SmoothingFunction().method4
        bleu_score = sentence_bleu(
            [reference_tokens], hypothesis_tokens, smoothing_function=smoothie
        )
        return float(bleu_score)
    except Exception as e:
        print(f"Error computing BLEU score: {e}")
        return 0.0


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        rouge_l_f1 = scores["rougeL"].fmeasure
        return float(rouge_l_f1)
    except Exception as e:
        print(f"Error computing ROUGE-L score: {e}")
        return 0.0


def compute_bertscore(reference: str, hypothesis: str) -> float:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        P, R, F1 = bert_score(
            [hypothesis],
            [reference],
            lang="en",
            rescale_with_baseline=True,
            device=device,
        )
        return float(F1.mean().item())
    except Exception as e:
        print(f"Error computing BERTScore: {e}")
        return 0.0


def compute_gpt_metric(
    reference: str, hypothesis: str, template: dict[str, str], client: Any, model: str
) -> float:
    criteria = template["LLM_ToM_Score"].format(
        true_reason=reference,
        predicted_reason=hypothesis,
    )

    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": criteria}], temperature=0
        )
        result = response.choices[0].message.content.strip()
        return result

    except Exception as e:
        print("LLM evaluation failed:", e)
        return False
