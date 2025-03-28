import random
from typing import Dict, MutableMapping, KT, VT


class BaseEncryption:
    def __init__(self, key):
        self.key = key

    def run(self, message: str) -> str:
        raise NotImplementedError

    def encrypt(self, message: str) -> str:
        raise NotImplementedError
    
    # should randomly generate a word mapping set
    def random_word_mapping(self) -> Dict:
        raise NotImplementedError
    

    def rag(self) -> str:
        raise NotImplementedError  