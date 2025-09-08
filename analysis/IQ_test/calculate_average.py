import json

file = 'results/incremental_results.jsonl'

gsm8k_scores = {}
aqua_scores = {}


def calculate_average(file):
    total = 0
    count = 0
    
    with open(file, 'r') as f:
        for line in f:
            data = json.loads(line)
     
    for entry in data:
        if entry['source'] == 'gsm8k':
            if entry['barrier_type'] == 'cultural_style':
                pass
            elif entry['barrier_type'] == 'cognitive_style':
                pass
            elif entry['barrier_type'] == 'social_style':
                pass
            elif entry['barrier_type'] == None:
                pass

        elif entry['source'] == 'aqua':
            pass

    
