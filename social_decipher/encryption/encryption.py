class BaseEncryption:
    def __init__(self, key):
        self.key = key

    def __call_(self, message: str) -> str:
        pass

    def encrypt(self, message: str) -> str:
        pass

    def decrypt(self, message: str) -> str:
        pass

    # should randomly generate a word mapping set
    def random_word_mapping(self) -> dict:
        pass

    def rag(self, message: str) -> str:
        pass
