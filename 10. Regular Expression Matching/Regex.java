import java.util.regex.Pattern;
import java.util.regex.Matcher;
class Solution {
    public boolean isMatch(String s, String p) {
        p = p.replaceAll("\\*+", "*"); // replaces all consecutive *'s with single *
        Pattern p1 = Pattern.compile(p);
        Matcher m1 = p1.matcher(s);
        return m1.matches();
    }
}
