import random
from typing import Dict, MutableMapping, KT, VT
from .encryption import BaseEncryption
from openai import OpenAI

def LLM_inference(prompt: str) -> str:
    client = OpenAI()

    response = client.responses.create(
        model="gpt-4o",
        input=prompt
    )
    return response.output_text

class MappingEncryption(BaseEncryption):
    def __init__(self, key):
        random.seed(key)
        alphabet = list("abcdefghijklmnopqrstuvwxyz")
        vowels = list("aeiou")
        consonants = list("bcdfghjklmnpqrstvwxyz")
        combinations = [c + v for c in consonants for v in vowels]
        # Randomly select 26 unique combinations (one for each letter)
        selected_combinations = random.sample(combinations, len(alphabet))

        # Create the mapping dictionary
        self.mapping = dict(zip(alphabet, selected_combinations))
        # add the upcase letters
        self.mapping.update({k.upper(): v.upper() for k, v in self.mapping.items()})

        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
        self.alphabet = alphabet

    def _simple_mapping(self, message: str) -> str:
        mapped_message = ""
        for letter in message:
            if letter in self.mapping:
                mapped_message += self.mapping[letter.lower()]
            else:
                mapped_message += letter
        return mapped_message
    
    def _simple_reverse_mapping(self, message: str) -> str:
        mapped_message = ""
        i = 0
        while i < len(message):
            combine = message[i:i+2].lower()
            if combine in self.reverse_mapping:
                mapped_message += self.reverse_mapping[combine]
                i += 2
            else:
                mapped_message += message[i]
                i += 1
        return mapped_message

    def __call__(self, message: str) -> str:
        return self.encrypt(message)

    def run(self, message: str) -> str:
        pass

    def encrypt(self, message: str) -> str:
        related_vocab = self.rag(message)
        related_vocab_str = "[Dictionary start] Original word -> target word\n" + '\n'.join([f"{k} -> {v}" for k, v in related_vocab.items()]) + "\n[Dictionary end]"
        prompt =  (f"You are a language expert good at translating messages based on given word mapping. The following is the word mapping you have:\n {related_vocab_str}"
                    f"\n\n Now please translate the following message based on the word mapping above:\n {message}"
                    "Respond with the translated message only."
                    "\n\nTranslated message:"
                    )
        #return self._simple_mapping(message)
        return LLM_inference(prompt)
    
    def decrypt(self, message: str) -> str:
        return self._simple_reverse_mapping(message)
    
    # should randomly generate a word mapping set
    def random_word_mapping(self) -> Dict:
        pass

    def rag(self, message: str) -> dict:
        related_vocab = set(message.split())
        mapped_vocab = {}
        for word in related_vocab:
            mapped_word = ""
            for letter in word:
                if letter in self.mapping:
                    mapped_word += self.mapping[letter.lower()]
                else:
                    mapped_word += letter
            mapped_vocab[word] = mapped_word
        return mapped_vocab