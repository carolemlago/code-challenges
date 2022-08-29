# Time complexity O(n*m)
# Space complexity O(n*m)
def findLength(self, nums1: List[int], nums2: List[int]) -> int:
    N, M = len(nums1), len(nums2) # len of lists
    dp = [[0 for _ in range(M+1)] for _ in range(N+1)]  # dp arr initialize with extra empty spot to handle exceptions
    output = 0  # count of subarray len
    for i in range(1, N+1): # loop through both arrays starting at index 1 since we had one extra in the dp
        for j in range(1, M+1):
            if nums1[i-1] == nums2[j-1]:    # if we have a match, add 1 to whatever number we have in the prev row and col
                dp[i][j] = 1 + dp[i-1][j-1] # increment the amount in our dp table of matching nums from both arrays
            output = max(output, dp[i][j])
    return output
