import random
import re
import string
from typing import Any, Dict, Optional, Union

from openai import OpenAI

from .encryption import BaseEncryption

class MappingEncryption(BaseEncryption):
    """
    Rule-based encryption that applies a consistent character mapping.
    This provides a deterministic, reversible transformation of text.
    """
    
    def __init__(
        self, 
        key: int = 42,
        mode: str = "substitution",  # "caesar", "substitution", or "word_shuffle"
        preserve_structure: bool = True,
        encrypt_keywords: bool = False,
    ):
        """
        Initialize mapping encryption.
        """
        self.key = key
        self.mode = mode
        self.preserve_structure = preserve_structure
        self.encrypt_keywords = encrypt_keywords
        
        # Initialize encryption mapping based on mode
        self._initialize_mapping()
    
    def _initialize_mapping(self):
        """Initialize encryption mapping based on the selected mode"""
        if self.mode == "caesar":
            # Caesar cipher - shift characters by key
            shift = self.key % 26  # Ensure shift is within alphabet range
            self.mapping = {
                c: chr((ord(c) - ord('a') + shift) % 26 + ord('a')) 
                for c in string.ascii_lowercase
            }
            # Add uppercase mapping
            self.mapping.update({
                c: chr((ord(c) - ord('A') + shift) % 26 + ord('A'))
                for c in string.ascii_uppercase
            })
            
        elif self.mode == "substitution":
            # Substitution cipher - random mapping
            random.seed(self.key)
            
            # Create random permutation of lowercase letters
            lowercase_chars = list(string.ascii_lowercase)
            shuffled_lowercase = lowercase_chars.copy()
            random.shuffle(shuffled_lowercase)
            
            # Create random permutation of uppercase letters
            uppercase_chars = list(string.ascii_uppercase)
            shuffled_uppercase = uppercase_chars.copy()
            random.shuffle(shuffled_uppercase)
            
            # Create mapping
            self.mapping = {}
            for i, c in enumerate(lowercase_chars):
                self.mapping[c] = shuffled_lowercase[i]
            for i, c in enumerate(uppercase_chars):
                self.mapping[c] = shuffled_uppercase[i]
                
        elif self.mode == "word_shuffle":
            # Word shuffle - initialize basic mapping for short words
            shift = self.key % 26
            self.mapping = {
                c: chr((ord(c) - ord('a') + shift) % 26 + ord('a')) 
                for c in string.ascii_lowercase
            }
            self.mapping.update({
                c: chr((ord(c) - ord('A') + shift) % 26 + ord('A'))
                for c in string.ascii_uppercase
            })
        else:
            raise ValueError(f"Unsupported encryption mode: {self.mode}")
            
        # Add digits mapping if needed
        if not self.preserve_structure:
            digits = string.digits
            self.mapping.update({
                d: str((int(d) + self.key) % 10) for d in digits
            })
    
    def __call__(self, text: Any) -> Any:
        """
        Encrypt text using the specified mapping.
        
        Args:
            text: Text to encrypt (string) or a dict with an 'argument' field
            
        Returns:
            Encrypted text in the same format
        """
        if isinstance(text, dict) and "argument" in text:
            # Handle action-based communication (dict with 'argument')
            argument = text["argument"]
            
            # Encrypt the argument field
            encrypted_text = ""
            if self.mode == "word_shuffle":
                encrypted_text = self._word_shuffle_encrypt(argument)
            else:
                encrypted_text = self._char_mapping_encrypt(argument)
            
            text["argument"] = encrypted_text
            return text
        else:
            # Handle regular text
            print(f"Encrypting text: {text}")
    
            if not isinstance(text, str):
                text = str(text)
            
            # Encrypt text
            encrypted_text = ""
            if self.mode == "word_shuffle":
                encrypted_text = self._word_shuffle_encrypt(text)
            else:
                encrypted_text = self._char_mapping_encrypt(text)
            
            print(f"Encrypted text: {encrypted_text}")
            return encrypted_text
    
    def _char_mapping_encrypt(self, text: str) -> str:
        """
        Encrypt text using character mapping (Caesar or substitution).
        
        Args:
            text: Text to encrypt
            
        Returns:
            Encrypted text
        """
        result = []
        
        # Check if we should avoid encrypting goal-related terms
        if not self.encrypt_keywords:
            # List of common goal-related terms to preserve
            keywords = ["want", "need", "goal", "objective", "aim", "target", "purpose",
                      "plan", "hope", "wish", "desire", "intention", "aspiration"]
            
            # Create regex pattern to match these words (case-insensitive, whole words only)
            pattern = r'\b(' + '|'.join(keywords) + r')\b'
            keyword_matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            # Keep track of which parts should not be encrypted
            preserve_ranges = [(m.start(), m.end()) for m in keyword_matches]
            
            for i, char in enumerate(text):
                # Check if this character is in a preserved range
                in_preserved_range = any(start <= i < end for start, end in preserve_ranges)
                
                if in_preserved_range:
                    result.append(char)
                elif char in self.mapping:
                    result.append(self.mapping[char])
                elif not self.preserve_structure:
                    result.append(self.mapping.get(char, '_'))
                else:
                    result.append(char)
        else:
            # Encrypt everything based on mapping
            for char in text:
                if char in self.mapping:
                    result.append(self.mapping[char])
                elif not self.preserve_structure:
                    result.append('_')
                else:
                    result.append(char)
        
        return ''.join(result)
    
    def _word_shuffle_encrypt(self, text: str) -> str:
        """
        Encrypt text by shuffling words while preserving word length.
        
        Args:
            text: Text to encrypt
            
        Returns:
            Encrypted text with shuffled words
        """
        random.seed(self.key)
        
        # Split text into words and non-words (whitespace, punctuation)
        pattern = r'(\w+|\s+|[^\w\s]+)'
        tokens = re.findall(pattern, text)
        
        result = []
        for token in tokens:
            if token.strip() and token.isalpha():
                # Shuffle characters in the word
                chars = list(token)
                word_len = len(chars)
                
                # If word is long enough to shuffle meaningfully
                if word_len > 3:
                    # Keep first and last letter, shuffle the middle
                    middle = chars[1:-1]
                    random.shuffle(middle)
                    shuffled = [chars[0]] + middle + [chars[-1]]
                    result.append(''.join(shuffled))
                else:
                    # For short words, apply character mapping
                    new_word = ''.join(self.mapping.get(c, c) for c in token)
                    result.append(new_word)
            else:
                # Preserve non-word tokens
                result.append(token)
        
        return ''.join(result)