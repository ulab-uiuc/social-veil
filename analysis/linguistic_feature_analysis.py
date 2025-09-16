import os
import json
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import argparse

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

def extract_features_from_transcript(transcript_text):
    """Extracts all linguistic features from a single conversation transcript."""
    # We only analyze Agent A's (Rafael's) speech, as they are the one with the barrier
    agent_a_text = ""
    for line in transcript_text.strip().split('\n'):
        if line.lower().startswith('rafael:'):
            agent_a_text += line.split(':', 1)[1]

    if not agent_a_text:
        return {}

    word_count = len(word_tokenize(agent_a_text.lower()))
    if word_count == 0:
        return {}

    features = {
        'sentiment_polarity': analyze_sentiment(agent_a_text),
        'reference_pronoun_rate': (count_reference_pronouns(agent_a_text) / word_count) * 100,
        'hedging_rate': (count_hedging_words(agent_a_text) / word_count) * 100,
        'self_focus_rate': (count_first_person_pronouns(agent_a_text) / word_count) * 100,
    }
    return features

# --- Data Loading and Processing ---

def load_data(base_dir):
    """Loads all conversation logs and evaluation results."""
    all_data = []
    modes = ['mode_baseline', 'mode_semantic', 'mode_cultural', 'mode_emotional']
    
    for mode in modes:
        mode_path = os.path.join(base_dir, mode)
        scenario_paths = glob.glob(os.path.join(mode_path, 'scenario_*'))
        
        for scenario_path in scenario_paths:
            try:
                transcript_path = os.path.join(scenario_path, 'conversation_log.txt')
                eval_path = os.path.join(scenario_path, 'eval_result.json')

                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read()
                
                with open(eval_path, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)

                linguistic_features = extract_features_from_transcript(transcript)
                
                outcome_scores = {
                    'unresolved_confusion': eval_data['aggregated_scores']['episode_level']['unresolved_confusion'],
                    'mutual_understanding': eval_data['aggregated_scores']['episode_level']['mutual_understanding'],
                    'agent_a_goal_completion': eval_data['aggregated_scores']['agent_1']['goal_completion'],
                    'agent_b_goal_completion': eval_data['aggregated_scores']['agent_2']['goal_completion'],
                }

                # Handle cases where scores might be nested
                for key, value in outcome_scores.items():
                    if isinstance(value, dict) and 'score' in value:
                        outcome_scores[key] = value['score']

                row = {
                    'mode': mode.replace('mode_', ''),
                    **linguistic_features,
                    **outcome_scores
                }
                all_data.append(row)

            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                print(f"Skipping {scenario_path} due to error: {e}")
                continue
                
    return pd.DataFrame(all_data)

# --- Main Analysis and Plotting ---

def main():
    parser = argparse.ArgumentParser(description="Analyze linguistic features of conversation transcripts.")
    parser.add_argument(
        '--results_dir', 
        type=str, 
        default='results/exp_qwen2.5-7b-instruct_episode_all_neutralized',
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

    # --- Figure 1: Radar Chart for Linguistic Signatures ---
    features_for_radar = ['reference_pronoun_rate', 'hedging_rate', 'self_focus_rate', 'sentiment_polarity']
    radar_palette = sns.color_palette("muted", len(df['mode'].unique()))
    
    # Normalize data for radar chart (scale each feature 0-1)
    df_radar = df.groupby('mode')[features_for_radar].mean().reset_index()
    for feature in features_for_radar:
        min_val = df_radar[feature].min()
        max_val = df_radar[feature].max()
        df_radar[feature] = (df_radar[feature] - min_val) / (max_val - min_val)

    labels = df_radar.columns[1:]
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for i, (index, row) in enumerate(df_radar.iterrows()):
        data = row.drop('mode').tolist()
        data += data[:1]
        ax.plot(angles, data, label=row['mode'], color=radar_palette[i], linewidth=2)
        ax.fill(angles, data, alpha=0.2, color=radar_palette[i])

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label.replace('_', ' ').title() for label in labels], size=12)
    plt.title('Figure 1: Linguistic Signatures of Communication Barriers', size=18, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig('analysis/figure1_linguistic_signatures.png', dpi=300, bbox_inches='tight')
    print("Saved Figure 1: Linguistic Signatures Radar Chart")

    # --- Figure 2: Correlation Heatmap (Focused View) ---
    linguistic_features = ['reference_pronoun_rate', 'hedging_rate', 'self_focus_rate', 'sentiment_polarity']
    outcome_metrics = ['unresolved_confusion', 'mutual_understanding', 'agent_a_goal_completion', 'agent_b_goal_completion']
    
    # Ensure all columns exist in the dataframe before correlation
    all_cols = [col for col in linguistic_features + outcome_metrics if col in df.columns]
    df_corr = df[all_cols].corr()
    
    # Select only the cross-correlation between features and outcomes
    focused_corr = df_corr.loc[linguistic_features, outcome_metrics]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        focused_corr, 
        annot=True, 
        cmap='RdBu_r', 
        fmt=".2f", 
        linewidths=.5, 
        linecolor='white',
        vmin=-1, vmax=1
    )
    plt.title('Figure 2: Linguistic Features vs. Conversational Outcomes', size=18)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('analysis/figure2_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved Figure 2: Correlation Heatmap")


if __name__ == '__main__':
    main()