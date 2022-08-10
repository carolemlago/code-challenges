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

# Range Sum Query - Immutable
""" Given an integer array nums, handle multiple queries of the following type:

Calculate the sum of the elements of nums between indices left and right inclusive where left <= right.
Implement the NumArray class:

NumArray(int[] nums) Initializes the object with the integer array nums.
int sumRange(int left, int right) Returns the sum of the elements of nums between indices left and right inclusive (i.e. nums[left] + nums[left + 1] + ... + nums[right]).""" 

class NumArray:

    def __init__(self, nums: List[int]):
        self.cumulative = [0] + list(accumulate(nums))
        

    def sumRange(self, left: int, right: int) -> int:
        return self.cumulative[right+1]-self.cumulative[left]

# Isomorphic Strings

""" Given two strings s and t, determine if they are isomorphic.
Two strings s and t are isomorphic if the characters in s can be replaced to get t.
All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character, but a character may map to itself. """
  
  def isIsomorphic(self, s: str, t: str) -> bool:
        map_letters = {}
        if len(s) != len(t):
            return False
        else:
            for i in range(len(s)):
                if s[i] in map_letters.keys():
                    if map_letters[s[i]] == t[i]:
                        pass  
                    else:
                        return False
                else:
                    if t[i] in map_letters.values():
                        return False
                    else:
                        map_letters[s[i]] = t[i]
            return True

# Is Subsequence
"""Given two strings s and t, return true if s is a subsequence of t, or false otherwise. 
A subsequence of a string is a new string that is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (i.e., "ace" is a subsequence of "abcde" while "aec" is not)"""

def isSubsequence(self, s: str, t: str) -> bool:
    if not s:
        return True
    i = 0
    for char in t:
        if char == s[i]:
            i += 1
        if i == len(s):
            break
    return i == len(s)

# Merge Two Sorted Lists

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = output = ListNode()
    
        while list1 and list2:
            if list1.val < list2.val:
                output.next = list1
                list1 = list1.next
            else:
                output.next = list2
                list2 = list2.next
            output = output.next
        if list1:
            output.next = list1
        elif list2:
            output.next = list2
        return dummy.next
        
# Reverse Linked List
""" Given the head of a singly linked list, reverse the list, and return the reversed list. """

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, current = None, head
        
        while current:
            temp = current.next
            current.next = prev
            prev = current
            current = temp
            
        return prev

# 1. Two Sum

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        answer = {}
        for i, elem in enumerate(nums):
            if target - elem in answer:
                return answer[target - elem], i
            answer[elem] = i


# Same Tree
""" Given the roots of two binary trees p and q, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value. """

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True

        if not p or not q or p.val != q.val:
            return False

        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

# Middle of the Linked List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        if head != None:
            while fast and fast.next:
                fast = fast.next.next
                slow = slow.next
        return slow
        

# Hacker Rank

# There is a large pile of socks that must be paired by color. Given an array of integers representing the color of each sock, determine how many pairs of socks with matching colors there are.
# Example
# There is one pair of color  and one of color . There are three odd socks left, one of each color. The number of pairs is .

def sockMerchant(n, ar):
    pair_stack = []
    count = 0
    for i in ar:
        if i not in pair_stack:
            pair_stack.append(i)
        else:
            pair_stack.remove(i)
            count += 1
            
    return count

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

# Simple Pig Latin
""" Move the first letter of each word to the end of it, then add "ay" to the end of the word. Leave punctuation marks untouched """

def pig_it(text):
    words = text.split()
    pig_list = []
    for word in words:
        if word in '!.%&?':
            pig_list.append(word)
        else:
            word = word[1:] + word[0] + "ay"
            pig_list.append(word)
    return ' '.join(pig_list)

    # or

def pig_it(text):
    words = text.split()
    pig_list = []
    for word in words:
        if word.isalpha():
            word = word[1:] + word[0] + "ay"
            pig_list.append(word)
        else:
            pig_list.append(word)
    return ' '.join(pig_list)

# Calculating with Functions
""" This time we want to write calculations using functions and get the results. Let's have a look at some examples:

seven(times(five())) # must return 35
four(plus(nine())) # must return 13
eight(minus(three())) # must return 5
six(divided_by(two())) # must return 3 

"""

def zero(f=None): return 0 if not f else f(0)
def one(f=None): return 1 if not f else f(1)
def two(f=None): return 2 if not f else f(2)
def three(f=None): return 3 if not f else f(3)
def four(f=None): return 4 if not f else f(4)
def five(f=None): return 5 if not f else f(5)
def six(f=None): return 6 if not f else f(6)
def seven(f=None): return 7 if not f else f(7)
def eight(f=None): return 8 if not f else f(8)
def nine(f=None): return 9 if not f else f(9)

def plus(y): return lambda x: x + y
def minus(y): return lambda x: x - y
def times(y): return lambda x: x * y
def divided_by(y): return lambda x: x // y

# Human Readable Time
""" Write a function, which takes a non-negative integer (seconds) as input and returns the time in a human-readable format (HH:MM:SS)

HH = hours, padded to 2 digits, range: 00 - 99
MM = minutes, padded to 2 digits, range: 00 - 59
SS = seconds, padded to 2 digits, range: 00 - 59
The maximum time never exceeds 359999 (99:59:59) """

def make_readable(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return "%02d:%02d:%02d" % (hour, min, sec)


# Valid Braces
""" Write a function that takes a string of braces, and determines if the order of the braces is valid. It should return true if the string is valid, and false if it's invalid.

This Kata is similar to the Valid Parentheses Kata, but introduces new characters: brackets [], and curly braces {}. Thanks to @arnedag for the idea!

All input strings will be nonempty, and will only consist of parentheses, brackets and curly braces: ()[]{}.

What is considered Valid?
A string of braces is considered valid if all braces are matched with the correct brace. """

def valid_braces(string):
    parens_dict = {")":"(","]":"[","}":"{"}
    
    to_visit = []
    for char in string:
        if char in parens_dict.values():
            to_visit.append(char)
        elif to_visit and parens_dict[char] == to_visit[-1]:
            to_visit.pop()
        else:
            return False
    return to_visit == []