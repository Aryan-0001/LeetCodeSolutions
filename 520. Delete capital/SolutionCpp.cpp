class Solution {
public:
    bool detectCapitalUse(string word) {
        return all_of(word.begin(),word.end(), ::isupper) || all_of(word.begin(),word.end(), ::islower) || !word.empty() && isupper(word[0]) &&
           all_of(word.begin() + 1, word.end(), ::islower);
    }
};
