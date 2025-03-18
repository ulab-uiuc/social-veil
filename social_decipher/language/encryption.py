class Encryptor:
    def apply_encryption(self, text: str, conlang: str) -> str:
        translation_table = str.maketrans("abcdefghijklmnopqrstuvwxyz", "tvaqzxmsbkpyfjncdwrghuleoi")
        return text.translate(translation_table)
    
    def reverse_encryption(self, text: str, conlang: str) -> str:
        translation_table = str.maketrans("tvaqzxmsbkpyfjncdwrghuleoi", "abcdefghijklmnopqrstuvwxyz")
        return text.translate(translation_table)