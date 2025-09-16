import os
import json
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import ttest_ind
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import argparse
from scipy.stats import pearsonr

# --- Setup: Download necessary NLTK data ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')


# --- Feature Extraction Functions ---

def analyze_sentiment(text):
    """Calculates the average sentiment polarity of the text."""
    sid = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(text)
    if not sentences:
        return 0
    
    polarity_scores = [sid.polarity_scores(sentence)['compound'] for sentence in sentences]
    return np.mean(polarity_scores)

def get_agent_a_name(transcript_text):
    """Dynamically identifies the name of the first speaker (Agent A)."""
    for line in transcript_text.strip().split('\n'):
        clean_line = line.strip()
        if ':' in clean_line:
            # The first speaker found is assumed to be Agent A
            return clean_line.split(':', 1)[0]
    return None

def count_reference_pronouns(text):
    """Counts the occurrences of ambiguous/impersonal reference pronouns."""
    tokens = word_tokenize(text.lower())
    # This set is designed to capture referential ambiguity, as per the user's description.
    reference_pronouns = {'it', 'that', 'thing', 'things', 'something', 'anything', 'everything'}
    return sum(1 for token in tokens if token in reference_pronouns)

def count_hedging_words(text):
    """Counts hedging words indicating uncertainty or politeness."""
    tokens = word_tokenize(text.lower())
    hedging_words = {
        'maybe', 'perhaps', 'could', 'might', 'possibly', 'appears', 'seems', 
        'sort of', 'kind of', 'a bit', 'slightly'
    }
    return sum(1 for token in tokens if token in hedging_words)

def count_first_person_pronouns(text):
    """Counts first-person pronouns to measure self-focus."""
    tokens = word_tokenize(text.lower())
    first_person_pronouns = {'i', 'me', 'my', 'mine', 'myself'}
    return sum(1 for token in tokens if token in first_person_pronouns)

def extract_features_from_transcript(transcript_text, agent_a_name):
    """Extracts all linguistic features from a single conversation transcript for a given agent."""
    if not agent_a_name:
        return {}
        
    agent_a_text = ""
    agent_a_name_lower = agent_a_name.lower()
    
    for line in transcript_text.strip().split('\n'):
        # Strip leading/trailing whitespace from each line before checking the name
        clean_line = line.strip()
        if clean_line.lower().startswith(f'{agent_a_name_lower}:'):
            agent_a_text += clean_line.split(':', 1)[1]

    if not agent_a_text:
        return {}

    word_count = len(word_tokenize(agent_a_text.lower()))
    if word_count == 0:
        return {}

    features = {
        'senti_polarity': analyze_sentiment(agent_a_text),
        'refer_pronoun': (count_reference_pronouns(agent_a_text) / word_count) * 100,
        'hedging': (count_hedging_words(agent_a_text) / word_count) * 100,
        'self_focus': (count_first_person_pronouns(agent_a_text) / word_count) * 100,
    }
    return features

# --- Data Loading and Processing ---

def load_data(base_dir):
    """Loads all conversation logs and evaluation results with robust error handling."""
    all_data = []
    modes = ['mode_baseline', 'mode_semantic', 'mode_cultural', 'mode_emotional']
    
    for mode in modes:
        mode_path = os.path.join(base_dir, mode)
        scenario_paths = glob.glob(os.path.join(mode_path, 'scenario_*'))
        
        for scenario_path in scenario_paths:
            row = {'mode': mode.replace('mode_', '')}
            try:
                # --- Load Transcript ---
                transcript_path = os.path.join(scenario_path, 'conversation_log.txt')
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read()

                # Dynamically determine Agent A's name from the transcript
                agent_a_name = get_agent_a_name(transcript)
                if not agent_a_name:
                    print(f"Warning: Could not determine Agent A's name in {transcript_path}. Skipping.")
                    continue
                
                linguistic_features = extract_features_from_transcript(transcript, agent_a_name)
                if not linguistic_features:
                    # This warning is now more specific
                    print(f"Warning: No features extracted from {transcript_path} for agent '{agent_a_name}'.")
                    continue
                row.update(linguistic_features)

                # --- Load Eval Results ---
                eval_path = os.path.join(scenario_path, 'eval_result.json')
                with open(eval_path, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)

                # Robustly extract nested scores
                episode_scores = eval_data.get('aggregated_scores', {}).get('episode_level', {})
                agent1_scores = eval_data.get('aggregated_scores', {}).get('agent_1', {})
                agent2_scores = eval_data.get('aggregated_scores', {}).get('agent_2', {})

                outcome_scores = {
                    'Confus.': episode_scores.get('unresolved_confusion'),
                    'Mutual': episode_scores.get('mutual_understanding'),
                    'Goal': agent2_scores.get('goal_completion'),
                    'Rel': agent2_scores.get('relationship'),
                    'Kno': agent2_scores.get('knowledge'),
                }
                
                # Handle cases where scores might be nested dicts, e.g., {"score": 5}
                for key, value in outcome_scores.items():
                    if isinstance(value, dict) and 'score' in value:
                        outcome_scores[key] = value['score']
                
                row.update(outcome_scores)
                all_data.append(row)

            except FileNotFoundError as e:
                print(f"Warning: Skipping {scenario_path}. Missing file: {e.filename}")
            except json.JSONDecodeError:
                print(f"Warning: Skipping {scenario_path}. Invalid JSON in {eval_path}")
            except KeyError as e:
                print(f"Warning: Skipping {scenario_path}. Missing key in JSON: {e}")
            except Exception as e:
                print(f"Warning: Skipping {scenario_path} due to an unexpected error: {e}")
                
    return pd.DataFrame(all_data)

# --- Main Analysis and Plotting ---

def main():
    parser = argparse.ArgumentParser(description="Analyze linguistic features of conversation transcripts.")
    parser.add_argument(
        '--results_dir', 
        type=str, 
        default='results/exp_qwen3-4b-instruct_episode_all_neutralized',
        help='Path to the base directory containing the experiment results.'
    )
    args = parser.parse_args()

    # --- Set professional plot style inspired by the user's example ---
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.titleweight': 'bold'
    })

    df = load_data(args.results_dir)
    
    if df.empty:
        print("No data loaded. Please check the results directory path.")
        return

    # --- Figure 2: Correlation Heatmap with Significance (using Matplotlib) ---
    linguistic_features = ['refer_pronoun', 'hedging', 'self_focus', 'senti_polarity']
    outcome_metrics = [
        'Confus.', 'Mutual', 'Goal', 
        'Rel', 'Kno'
    ]
    
    # Calculate correlation and p-value matrices
    corr_matrix = df[linguistic_features + outcome_metrics].corr().loc[linguistic_features, outcome_metrics]
    
    p_values = pd.DataFrame(index=linguistic_features, columns=outcome_metrics, dtype=float)
    for feature in linguistic_features:
        for outcome in outcome_metrics:
            clean_df = df[[feature, outcome]].dropna()
            if len(clean_df) > 1:
                _, p_value = pearsonr(clean_df[feature], clean_df[outcome])
                p_values.loc[feature, outcome] = p_value
            else:
                p_values.loc[feature, outcome] = 1.0

    # Create annotations based on p-values
    annot_matrix = p_values.apply(lambda s: s.apply(lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    
    # Use a diverging norm to center the colormap at 0
    norm = mcolors.TwoSlopeNorm(vmin=corr_matrix.min().min(), vcenter=0, vmax=corr_matrix.max().max())
    im = ax.imshow(corr_matrix, cmap='RdBu_r', norm=norm)

    # Create colorbar with proper sizing
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom", fontsize=12)

    # Add the significance annotations
    for i in range(len(linguistic_features)):
        for j in range(len(outcome_metrics)):
            text = ax.text(j, i, annot_matrix.iloc[i, j],
                           ha="center", va="center", color="black", fontsize=14)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(outcome_metrics)))
    ax.set_yticks(np.arange(len(linguistic_features)))
    ax.set_xticklabels([label.replace('_', ' ').title() for label in outcome_metrics], fontsize=14, fontweight='bold')
    ax.set_yticklabels([label.replace('_', ' ').title() for label in linguistic_features], fontsize=14, fontweight='bold')
    
    
    # Remove grid lines but keep tick marks
    ax.grid(False)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_title('Qwen3-4B', fontsize=18)

    fig.tight_layout()
    plt.savefig('analysis/figure2_correlation_heatmap.png', dpi=500, bbox_inches='tight')
    print("Saved Correlation Heatmap with Significance")




    # --- Figure 3: Linguistic Signatures vs Barrier Types (delta vs baseline with significance) ---
    barrier_types = ['semantic', 'cultural', 'emotional']
    features = ['refer_pronoun', 'hedging', 'self_focus', 'senti_polarity']

    # Prepare matrices: value = mean(barrier) - mean(baseline); p-values from Welch t-test
    baseline_df = df[df['mode'] == 'baseline']
    delta_matrix = pd.DataFrame(index=features, columns=barrier_types, dtype=float)
    pval_matrix = pd.DataFrame(index=features, columns=barrier_types, dtype=float)

    for bt in barrier_types:
        bdf = df[df['mode'] == bt]
        for feat in features:
            base_vals = baseline_df[feat].dropna()
            barrier_vals = bdf[feat].dropna()
            # Mean difference (barrier - baseline)
            delta = barrier_vals.mean() - base_vals.mean()
            delta_matrix.loc[feat, bt] = delta
            # Welch's t-test
            if len(base_vals) > 1 and len(barrier_vals) > 1:
                try:
                    tstat, pval = ttest_ind(barrier_vals, base_vals, equal_var=False, nan_policy='omit')
                except Exception:
                    pval = 1.0
            else:
                pval = 1.0
            pval_matrix.loc[feat, bt] = pval

    # Significance annotations
    annot_b = pval_matrix.apply(lambda s: s.apply(lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''))

    fig, ax = plt.subplots(figsize=(7.5, 6))

    # Center colormap at 0 for deltas – match Figure 2 style
    norm_b = mcolors.TwoSlopeNorm(vmin=delta_matrix.min().min(), vcenter=0, vmax=delta_matrix.max().max())
    im = ax.imshow(delta_matrix, cmap='RdBu_r', norm=norm_b)

    # Colorbar – match Figure 2 sizing
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.ax.set_ylabel("Δ Feature vs Baseline", rotation=-90, va="bottom", fontsize=12)

    # Annotations
    for i in range(len(features)):
        for j in range(len(barrier_types)):
            ax.text(j, i, annot_b.iloc[i, j], ha="center", va="center", color="black", fontsize=14)

    # Ticks and labels (bold) – match Figure 2
    ax.set_xticks(np.arange(len(barrier_types)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels([bt.title() for bt in barrier_types], fontsize=14, fontweight='bold')
    ax.set_yticklabels([label.replace('_', ' ').title() for label in features], fontsize=14, fontweight='bold')
    ax.grid(False)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title('Qwen2.5_7B', fontsize=18)
    fig.tight_layout()
    plt.savefig('analysis/figure3_signature_vs_barrier.png', dpi=500, bbox_inches='tight')
    print("Saved Figure 3: Signatures vs Barrier Types")

if __name__ == '__main__':
    main()