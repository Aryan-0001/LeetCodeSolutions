### Check out : https://leetcode.com/problems/regular-expression-matching/solutions/4923717/easy-solution-in-5-lines-java-by-kingz_0-rl28
## Problem    ```🔴HARD```
Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:

- '.' Matches any single character.​​​​
- '*' Matches zero or more of the preceding element.

The matching should cover the entire input string (not partial).

- ### Example 1:
    
      Input: s = "aa", p = "a"
      Output: false
      Explanation: "a" does not match the entire string "aa".

- ### Example 2:

      Input: s = "aa", p = "a*"
      Output: true
      Explanation: '*' means zero or more of the preceding element, 'a'. Therefore, by repeating 'a' once, it becomes "aa".
- ### Example 3:
      
      Input: s = "ab", p = ".*"
      Output: true
      Explanation: ".*" means "zero or more (*) of any character (.)".
