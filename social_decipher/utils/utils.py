import json


def load_env(env_data_path):
    with open(env_data_path, "r") as f:
        episodes = [json.loads(line) for line in f]

        return episodes

