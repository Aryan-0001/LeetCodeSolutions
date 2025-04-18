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
