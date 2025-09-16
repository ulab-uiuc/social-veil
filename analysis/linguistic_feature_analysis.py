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
        'sentiment_polarity': analyze_sentiment(agent_a_text),
        'reference_pronoun_rate': (count_reference_pronouns(agent_a_text) / word_count) * 100,
        'hedging_rate': (count_hedging_words(agent_a_text) / word_count) * 100,
        'self_focus_rate': (count_first_person_pronouns(agent_a_text) / word_count) * 100,
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
                    'unresolved_confusion': episode_scores.get('unresolved_confusion'),
                    'mutual_understanding': episode_scores.get('mutual_understanding'),
                    'agent_a_goal_completion': agent1_scores.get('goal_completion'),
                    'agent_b_goal_completion': agent2_scores.get('goal_completion'),
                    'agent_b_relationship': agent2_scores.get('relationship'),
                    'agent_b_knowledge': agent2_scores.get('knowledge'),
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

    # --- Figure 2: Correlation Heatmap with Significance (using Matplotlib) ---
    linguistic_features = ['reference_pronoun_rate', 'hedging_rate', 'self_focus_rate', 'sentiment_polarity']
    outcome_metrics = [
        'unresolved_confusion', 'mutual_understanding', 'agent_b_goal_completion', 
        'agent_b_relationship', 'agent_b_knowledge'
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

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use a diverging norm to center the colormap at 0
    norm = mcolors.TwoSlopeNorm(vmin=corr_matrix.min().min(), vcenter=0, vmax=corr_matrix.max().max())
    im = ax.imshow(corr_matrix, cmap='RdBu_r', norm=norm)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")

    # Add the significance annotations
    for i in range(len(linguistic_features)):
        for j in range(len(outcome_metrics)):
            text = ax.text(j, i, annot_matrix.iloc[i, j],
                           ha="center", va="center", color="black", fontsize=14)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(outcome_metrics)))
    ax.set_yticks(np.arange(len(linguistic_features)))
    ax.set_xticklabels([label.replace('_', ' ').title() for label in outcome_metrics])
    ax.set_yticklabels([label.replace('_', ' ').title() for label in linguistic_features])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_title('Linguistic Features vs. Conversational Outcomes (Significance)', size=18)
    fig.tight_layout()
    plt.savefig('analysis/figure2_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved Correlation Heatmap with Significance")


if __name__ == '__main__':
    main()