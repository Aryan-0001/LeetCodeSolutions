class Solution {
    public int removeElement(int[] nums, int val) {
        ArrayList<Integer> numArr = new ArrayList<>();
        int k = nums.length;
        for (int i = 0; i < k; i++) {
            if (nums[i] != val) {
                numArr.add(nums[i]);
            }
        }
        for (int i = 0; i < numArr.size(); i++) {
            nums[i] = numArr.get(i);
        }
        return numArr.size();
    }
}
