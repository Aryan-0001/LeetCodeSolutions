class Solution:
    def dayOfYear(self, date: str) -> int:
        year,month,day = map(int, date.split('-'))
        if (year%400==0 or (year%100!=0 and year%4==0)) and month>2: 
            day+=1
        months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        for i in range(1, month):
            day+=months[i-1]
        return day
