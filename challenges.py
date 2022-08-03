# Leet Code


# Missing Number
# Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.
def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
        nums_set = set(nums)
        for i in range (n+1):
            if i not in nums_set:
                return i



# Contains Duplicate
# Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.
def containsDuplicate(self, nums: List[int]) -> bool:
        to_check = set(nums)
        if len(to_check) == len(nums):
            return False
        else:
            return True

# Contains Duplicate II
# Given an integer array nums and an integer k, return true if there are two distinct indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.
def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
       
        to_visit = {}
        
        for i in range(len(nums)):
            if nums[i] in to_visit and abs(i- to_visit[nums[i]] <= k):
                return True
            
            to_visit[nums[i]] = i
            
        return False

# Find All Numbers Disappeared in an Array
# Given an array nums of n integers where nums[i] is in the range [1, n], return an array of all the integers in the range [1, n] that do not appear in nums.
 def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        n = len(nums)
        
        # Put in a set to optimize runtime 
        nums_set = set(nums)
        
        # Use a stack to store disappeared nums
        missing_lst = []
        
        for i in range (1, (n+1)):
            if i not in nums_set:
                missing_lst.append(i)
        
        return missing_lst

# Single Number
# Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.
# You must implement a solution with a linear runtime complexity and use only constant extra space.
def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            res = num ^ res
        
        return res

# Climbing Stairs
# You are climbing a staircase. It takes n steps to reach the top.
# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
def climbStairs(self, n: int) -> int:
        one, two = 1, 1
        
        for i in range (n-1):
            temp = one
            one = one + two
            two = temp
            
        return one

# Best Time to Buy and Sell Stock
""" You are given an array prices where prices[i] is the price of a given stock on the ith day.
You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0."""
def maxProfit(self, prices: List[int]) -> int:
        left, right = 0, 1
        max_profit = 0
        
        while right < len(prices):
            if prices[left] < prices[right]:
                profit = prices[right] - prices[left]
                max_profit = max(max_profit, profit)
            else:
                left = right
            
            right += 1
            
        return max_profit

# Running Sum of 1d Array
""" Given an array nums. We define a running sum of an array as runningSum[i] = sum(nums[0]…nums[i]).
Return the running sum of nums.
"""
def runningSum(self, nums: List[int]) -> List[int]:
    results = [0] * len(nums)
    results[0] = nums[0]
    for i in range(1, len(nums)):
        results[i] = results[i-1] + nums[i]
    return results
            
            
# Find Pivot Index
""" Given an array of integers nums, calculate the pivot index of this array.
The pivot index is the index where the sum of all the numbers strictly to the left of the index is equal to the sum of all the numbers strictly to the index's right.
If the index is on the left edge of the array, then the left sum is 0 because there are no elements to the left. This also applies to the right edge of the array.
Return the leftmost pivot index. If no such index exists, return -1."""

def pivotIndex(self, nums: List[int]) -> int:
    total = sum(nums)
    left_sum = 0
    for i in range(len(nums)):
        right_sum = total - nums[i] - left_sum
        if left_sum == right_sum:
            return i
        left_sum += nums[i]
    return -1
            

# Codewars
# The two oldest ages function/method needs to be completed. It should take an array of numbers as its argument and return the two highest numbers within the array. The returned value should be an array in the format [second oldest age,  oldest age].
# The order of the numbers passed in could be any order. The array will always include at least 2 items. If there are two or more oldest age, then return both of them in array format.

def two_oldest_ages(ages):
    ages.sort()
    
    return ages[-2::]


# Convert string to camel case
"""Complete the method/function so that it converts dash/underscore delimited words into camel casing. The first word within the output should be capitalized only if the original word was capitalized (known as Upper Camel Case, also often referred to as Pascal case)."""

def to_camel_case(text):
    #your code here

    words = text.replace("-", " ").replace("_", " ") 
    str = words.split(" ") 
    if len(text) == 0:
        return text
    return str[0] + ''.join(i.capitalize() for i in str[1:])
    
# Unique In Order
"""Implement the function unique_in_order which takes as argument a sequence and returns a list of items without any elements with the same value next to each other and preserving the original order of elements."""

def unique_in_order(iterable):
    list_unique = []
    letter = None
    for current in iterable:
        if current != letter:
            list_unique.append(current)
            letter = current
    return list_unique

# The highest profit wins!
"""Ben has a very simple idea to make some profit: he buys something and sells it again. Of course, this wouldn't give him any profit at all if he was simply to buy and sell it at the same price. Instead, he's going to buy it for the lowest possible price and sell it at the highest."""
def min_max(lst):
  return [min(lst), max(lst)]

# or 

def min_max(lst):
    min, max = lst[0], lst[0]
    for num in lst:
        if num < min:
            min = num
        elif num > max:
            max = num
    
    return [min, max]
    