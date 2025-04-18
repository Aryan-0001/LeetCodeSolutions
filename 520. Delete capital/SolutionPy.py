class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        return word==word.upper() or word==word.capitalize() or word==word.lower()
