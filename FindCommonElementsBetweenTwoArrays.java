import java.util.ArrayList;
class Solution {
    public int[] findIntersectionValues(int[] nums1, int[] nums2) {
        int ans1 = 0, ans2 = 0;
        ArrayList<Integer> arr = new ArrayList<>();
        ArrayList<Integer> list2 = new ArrayList<>();
        for(int n:nums2){
            list2.add(n);
        }
        for(int i=0;i<nums1.length;i++){
            if(list2.contains(nums1[i])){
                ans1++;
            }
        }
        ArrayList<Integer> list1 = new ArrayList<>();
        for(int n:nums1){
            list1.add(n);
        }
        for(int i=0;i<nums2.length;i++){
            if(list1.contains(nums2[i])){
                ans2++;
            }
        }
        arr.add(ans1); arr.add(ans2);
        int[] result = new int[arr.size()];
        for(int i=0;i<arr.size();i++){
            result[i] = arr.get(i);
        }
        return result;
    }
}
