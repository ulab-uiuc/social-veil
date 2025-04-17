# generate_data.py
import json
import time
from pathlib import Path

import yaml
from openai import OpenAI

# Load prompts from YAML file
with open("../configs/data_generation.yaml") as f:
    generation_prompts = yaml.safe_load(f)


def generate_unique_completions(prompt: str, n: int, output_file: str):
    unique_data = set()
    results = []
    client = OpenAI()
    output_path = Path(output_file)

    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
            for item in existing:
                json_key = json.dumps(item, sort_keys=True)
                unique_data.add(json_key)
                results.append(item)
        except Exception as e:
            print(f"⚠️ Failed to read existing data: {e}")

    while len(results) < n:
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.9,
            )
            content = response.choices[0].message.content.strip()

            parsed = json.loads(content)
            json_key = json.dumps(parsed, sort_keys=True)
            print(json_key)
            if json_key not in unique_data:
                unique_data.add(json_key)
                results.append(parsed)
                with output_path.open("w") as f:
                    json.dump(results, f, indent=2)
                print(f"✅ Generated and saved {len(results)}/{n}")

        except json.JSONDecodeError:
            print("❌ Failed to parse JSON, skipping.")
        except Exception as e:
            print(f"❌ Error during generation: {e}")
        time.sleep(1)


def generate_agent_profiles(num: int, output_file="agent_profiles.json"):
    prompt = generation_prompts["Profile"]
    generate_unique_completions(prompt, num, output_file)
    print(f"✅ Saved {num} unique agent profiles to {output_file}")


def generate_environment_profiles(num: int, output_file="environment_profiles.json"):
    prompt = generation_prompts["Environment"]
    generate_unique_completions(prompt, num, output_file)
    print(f"✅ Saved {num} unique environment profiles to {output_file}")


if __name__ == "__main__":
    generate_agent_profiles(num=5)
    generate_environment_profiles(num=5)
