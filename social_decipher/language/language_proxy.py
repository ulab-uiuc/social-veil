from language.encryption import Encryptor

class LanguageProxy:
    def __init__(self, native_language: str, conlang: str):
        self.native_language = native_language
        self.conlang = conlang
        self.encryptor = Encryptor()
    
    def encrypt(self, text: str) -> str:
        return self.encryptor.apply_encryption(text, self.conlang)
    
    def decrypt(self, text: str) -> str:
        return self.encryptor.reverse_encryption(text, self.conlang)


# language/encryption.py
class Encryptor:
    def apply_encryption(self, text: str, conlang: str) -> str:
        translation_table = str.maketrans("abcdefghijklmnopqrstuvwxyz", "tvaqzxmsbkpyfjncdwrghuleoi")
        return text.translate(translation_table)
    
    def reverse_encryption(self, text: str, conlang: str) -> str:
        translation_table = str.maketrans("tvaqzxmsbkpyfjncdwrghuleoi", "abcdefghijklmnopqrstuvwxyz")
        return text.translate(translation_table)
