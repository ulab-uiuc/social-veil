import random

# Define the alphabet, vowels, and consonants
alphabet = list("abcdefghijklmnopqrstuvwxyz")
vowels = list("aeiou")
consonants = list("bcdfghjklmnpqrstvwxyz")

# Generate all possible consonant-vowel combinations
combinations = [c + v for c in consonants for v in vowels]

# Randomly select 26 unique combinations (one for each letter)
selected_combinations = random.sample(combinations, len(alphabet))

# Create the mapping dictionary
mapping = dict(zip(alphabet, selected_combinations, strict=False))

# Print the mapping
for letter, combo in mapping.items():
    print(f"{letter}: {combo}")
