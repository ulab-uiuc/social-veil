import numpy as np
from typing import Dict, List, Tuple, Union, Any 

import nltk
import torch
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

nltk.download("punkt")

# Confidence binning configuration
CONFIDENCE_BINS = {
    "Low": (0.0, 0.4),
    "Medium": (0.4, 0.7),
    "High": (0.7, 1.0)
}

def get_confidence_bin(confidence_score: float) -> str:
    """
    Convert a continuous confidence score to a discrete bin category.

    Args:
        confidence_score: Float between 0 and 1

    Returns:
        String: "Low", "Medium", or "High"
    """
    if not 0 <= confidence_score <= 1:
        raise ValueError(f"Confidence score must be between 0 and 1, got {confidence_score}")

    for bin_name, (lower, upper) in CONFIDENCE_BINS.items():
        if lower <= confidence_score < upper:
            return bin_name

    # Handle edge case where confidence_score == 1.0
    if confidence_score == 1.0:
        return "High"

    raise ValueError(f"Could not assign confidence score {confidence_score} to any bin")

def get_confidence_bin_boundaries() -> Dict[str, Tuple[float, float]]:
    """
    Get the confidence bin boundaries for use in prompts or analysis.

    Returns:
        Dictionary mapping bin names to (lower, upper) boundaries
    """
    return CONFIDENCE_BINS.copy()

def get_confidence_bin_center(bin_name: str) -> float:
    """
    Get the center value of a confidence bin.

    Args:
        bin_name: "Low", "Medium", or "High"

    Returns:
        Center value of the bin
    """
    if bin_name not in CONFIDENCE_BINS:
        raise ValueError(f"Invalid bin name: {bin_name}. Must be one of {list(CONFIDENCE_BINS.keys())}")

    lower, upper = CONFIDENCE_BINS[bin_name]
    return (lower + upper) / 2

def analyze_confidence_distribution(confidence_scores: List[float]) -> Dict[str, Union[int, float]]:
    """
    Analyze the distribution of confidence scores across bins.

    Args:
        confidence_scores: List of confidence scores (0-1)

    Returns:
        Dictionary with bin counts and percentages
    """
    bin_counts = {"Low": 0, "Medium": 0, "High": 0}
    total_scores = len(confidence_scores)

    for score in confidence_scores:
        bin_name = get_confidence_bin(score)
        bin_counts[bin_name] += 1

    # Calculate percentages
    bin_percentages = {bin_name: count / total_scores * 100 for bin_name, count in bin_counts.items()}

    return {
        "counts": bin_counts,
        "percentages": bin_percentages,
        "total_scores": total_scores,
        "mean_confidence": np.mean(confidence_scores),
        "std_confidence": np.std(confidence_scores)
    }

def validate_confidence_consistency(predicted_value: float, predicted_class: str) -> bool:
    """
    Validate that a predicted confidence value is consistent with its predicted class.

    Args:
        predicted_value: Continuous confidence score (0-1)
        predicted_class: Predicted bin class ("Low", "Medium", "High")

    Returns:
        True if consistent, False otherwise
    """
    actual_bin = get_confidence_bin(predicted_value)
    return actual_bin == predicted_class

def get_confidence_guidelines_text() -> str:
    """
    Get formatted confidence guidelines text for use in prompts.

    Returns:
        Formatted string with confidence guidelines
    """
    return """**Confidence Scoring Guidelines:**
- **Low Confidence (0.0-0.4)**: Very uncertain, limited evidence, conflicting signals
- **Medium Confidence (0.4-0.7)**: Some evidence, reasonable inference, moderate certainty
- **High Confidence (0.7-1.0)**: Strong evidence, clear patterns, high certainty"""


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
