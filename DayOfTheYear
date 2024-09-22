class Solution {
    public int dayOfYear(String date) {
        String[] dateArr = date.split("-");
        int year = Integer.parseInt(dateArr[0]);
        int month = Integer.parseInt(dateArr[1]);
        int day = Integer.parseInt(dateArr[2]);
        boolean isLeapYear = (year % 400 == 0) || (year % 100 != 0 && year % 4 == 0);
        int[] months = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        if (isLeapYear){
            months[1] = 29;
        }
        for(int i=1;i<month;i++){
            day+=months[i-1];
        }
        return day;
    }
}
