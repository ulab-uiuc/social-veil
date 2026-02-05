import json
import os
import sys

import yaml

SOTOPIA_HARD_ENVS = ["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]


# Original Data 
with open("./processed_sotopia/sotopia_cleaned.jsonl") as f:
    sotopia = [json.loads(l) for l in f]

# check number of unique environment_id
unique_envs = set(ep["environment_id"] for ep in sotopia)

with open("./episode_all.jsonl") as f:
    all_episodes = [json.loads(l) for l in f]

with open("./episode_hard.jsonl") as f:
    hard_episodes = [json.loads(l) for l in f]

print(len(unique_envs))
print(len(sotopia))
print(len(all_episodes))
print(len(hard_episodes))

