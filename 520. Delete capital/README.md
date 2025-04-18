# 520. Detect Capital ✅Easy - 781
We define the usage of capitals in a word to be right when one of the following cases holds:
- All letters in this word are capitals, like "USA".
- All letters in this word are not capitals, like "leetcode".
- Only the first letter in this word is capital, like "Google".
- Given a string word, return true if the usage of capitals in it is right.
<br>

### Example 1:
  > Input: word = "USA" <br>
    Output: true
<br>

### Example 2:
  > Input: word = "FlaG" <br>
    Output: false
 
### Constraints:

- 1 <= ```word.length``` <= 100
- word consists of lowercase and uppercase English letters.

---
Solution:
# Python
```python3
class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        return word==word.upper() or word==word.capitalize() or word==word.lower()
```
# Java
```java
class Solution {
    public boolean detectCapitalUse(String word) {
        return word.equals(word.toUpperCase()) || word.equals(word.toLowerCase()) || word.equals(word.substring(0, 1).toUpperCase() + word.substring(1).toLowerCase()); 
    }
}
```

# C++
```C++
class Solution {
public:
    bool detectCapitalUse(string word) {
        return all_of(word.begin(),word.end(), ::isupper) || all_of(word.begin(),word.end(), ::islower) || !word.empty() && isupper(word[0]) &&
           all_of(word.begin() + 1, word.end(), ::islower);
    }
};
```
# C
```c
#include<stdio.h>
#include<string.h>
bool detectCapitalUse(char* word) {
    int wordLen = strlen(word);
    if(wordLen==1) return true;
    bool isUpper=true, isLower=true , isCap=true;
    for(int i=0;i<wordLen;i++){
        if( !(word[i]>='A'&& word[i]<='Z') ) isUpper=false;
        if( !(word[i]>='a'&& word[i]<='z') ) isLower=false;
        if( i>0 && (word[i]>='A'&& word[i]<='Z')) isCap = false;
    }
    return isUpper||isLower||isCap;
}
```

## *If it helped pls upvote it here ⬆️: [link](https://leetcode.com/problems/detect-capital/solutions/6352787/1-liner-solution-by-kingz_0101-63bv)*
