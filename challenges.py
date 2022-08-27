# Leet Code

#  Valid Palindrome

"""A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise."""
class Solution:
    def isPalindrome(self, s: str) -> bool:
        newString = ""
        
        for c in s:
            if c.isalnum():
                newString += c.lower()
        return newString == newString[::-1]


# Valid Anagram

""" Given two strings s and t, return true if t is an anagram of s, and false otherwise.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
"""
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        countS = {}
        countT = {}
        
        for i in range(len(s)):
            countS[s[i]] = 1 + countS.get(s[i], 0)
            countT[t[i]] = 1 + countT.get(t[i], 0)
        for c in countS:
            if countS[c] != countT[c]:
                return False
        return True
                

# 1209. Remove All Adjacent Duplicates in String II

""" You are given a string s and an integer k, a k duplicate removal consists of choosing k adjacent and equal letters from s and removing them, causing the left and the right side of the deleted substring to concatenate together.

We repeatedly make k duplicate removals on s until we no longer can.

Return the final string after all such duplicate removals have been made. It is guaranteed that the answer is unique. """

class Solution:
    def removeDuplicates(self, s: str, k: int) -> str:
        dup = [] # [char, count]
        for c in s:
            if dup and dup[-1][0] == c:
                dup[-1][1] += 1
            else:
                dup.append([c, 1])
            if dup[-1][1] == k:
                dup.pop()
        ans = ""
        for char, count in dup:
            ans += (char * count)
        return ans


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
                return [answer[target - elem], i]
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
        
# Valid Parentheses
""" Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order. """

class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) % 2 != 0:
            return False
  
        par_dict = {'(':')','{':'}','[':']'}
        stack = []
        for char in s:
            if char in par_dict.keys():
                stack.append(char)
            else:
                if stack == []:
                    return False
                open_brac = stack.pop()
                if char != par_dict[open_brac]:
                    return False
        return stack == []
        
# Counting Bits
""" Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i. """
class Solution:
    def countBits(self, n: int) -> List[int]:
        dp = [0] * (n + 1)
        offset = 1
        for i in range(1, n + 1):
            if offset * 2 == i:
                offset = i
            dp[i] = 1 + dp[i-offset]
        return dp

# 141. Linked List Cycle       
""" Given head, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

Return true if there is a cycle in the linked list. Otherwise, return false. """ 
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        # use fast and slow pointers, and if they meet at some point, then it's a cycle
        slow = fast = head
        while slow and fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False

        #or

def hasCycle(self, head: Optional[ListNode]) -> bool:
    seen = set() # runtime and spacetime = O(n)
    current = head
    while current:
        if current in seen:
            return True
        seen.add(current)
        current = current.next
    return False
# 234. Palindrome Linked List

""" Given the head of a singly linked list, return true if it is a palindrome."""
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        reverse = []
        curr = head
        while curr:
            reverse.append(curr.val)
            curr = curr.next
        return reverse == reverse[::-1]

#  203. Remove Linked List Elements

""" Given the head of a linked list and an integer val, remove all the nodes of the linked list that has Node.val == val, and return the new head. """
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
              
        curr = head
        if head is None:
            return head  
        while curr.next:
            if curr.next.val == val:
                curr.next = curr.next.next
            else:
                curr = curr.next
        if head.val == val:
            head = head.next
        return head

# 83. Remove Duplicates from Sorted List

""" Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.
"""

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head
        if head is None:
            return head
        while cur and cur.next:
            if cur.val != cur.next.val:
                cur = cur.next
            else:
                cur.next = cur.next.next
        return head

# L142. Linked List Cycle II
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        node_list = set()

        while head:
            if head in node_list:
                return head
            
            node_list.add(head)
            head = head.next
        
        return None

# Hacker Rank


#  Caesar Cypher
"""Julius Caesar protected his confidential information by encrypting it using a cipher. Caesar's cipher shifts each letter by a number of letters. If the shift takes you past the end of the alphabet, just rotate back to the front of the alphabet. In the case of a rotation by 3, w, x, y and z would map to z, a, b and c.

Original alphabet:      abcdefghijklmnopqrstuvwxyz
Alphabet rotated +3:    defghijklmnopqrstuvwxyzabc
Example

The alphabet is rotated by , matching the mapping above. The encrypted string is .
Note: The cipher only encrypts letters; symbols, such as -, remain unencrypted.
Function Description
Complete the caesarCipher function in the editor below.
caesarCipher has the following parameter(s):
string s: cleartext
int k: the alphabet rotation factor
Returns

string: the encrypted string
 """
def caesarCipher(s, k):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    rotated = alphabet[k%26:] + alphabet[:k%26] #rotated alphabet by +k
    
    encrypted_alphabet = {} #alphabet to keep track of original letter and the corresponding rotated one
    for i in range(len(alphabet)): 
        encrypted_alphabet[alphabet[i]] = rotated[i]
        
    message = ""
    for letter in s: # loop to add letter to message
        if letter.isupper():
            if letter.lower() in encrypted_alphabet:
                message += encrypted_alphabet[letter.lower()].upper()
        
        elif letter.islower():
            if letter in encrypted_alphabet:
                message += encrypted_alphabet[letter]
        else:
            message += letter
    return message


# Sock Merchant
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

#!/bin/python3

# import math
# import os
# import random
# import re
# import sys

#
# Complete the 'countingValleys' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER steps
#  2. STRING path
#

def countingValleys(steps, path):
    cur_level = 0
    valleys = 0
    for p in path:
        if p == "U":
            cur_level += 1
            if cur_level == 0:
                valleys += 1
        else:
            cur_level -= 1
    return valleys


#!/bin/python3

# import math
# import os
# import random
# import re
# import sys

#
# Complete the 'jumpingOnClouds' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY c as parameter.
#

def jumpingOnClouds(c):
    jump = 0
    i = 0
    while i < n - 1:
        if i + 2 < n and c[i+2] == 0:
            jump += 1
            i += 2
        elif i + 1 < n and c[i+1] == 0:
            jump += 1
            i += 1
    return jump

#!/bin/python3

# import math
# import os
# import random
# import re
# import sys

#
# Complete the 'repeatedString' function below.
#
# The function is expected to return a LONG_INTEGER.
# The function accepts following parameters:
#  1. STRING s
#  2. LONG_INTEGER n
#

def repeatedString(s, n):
   n1 = (n//len(s))
   x = s.count("a")
   x1 = n1 * x
   x2 = s[:n%(len(s))].count("a")
   return x1 + x2  

# Arrays: Left Rotation

""" A left rotation operation on an array shifts each of the array's elements  unit to the left. For example, if  left rotations are performed on array , then the array would become . Note that the lowest index item moves to the highest index in a rotation. This is called a circular array.

Given an array  of  integers and a number, , perform  left rotations on the array. Return the updated array to be printed as a single line of space-separated integers. """

def rotLeft(a, d):
    d = d % n # if it's not divisible by len(a), rotate only the mod
    l1 = a[0:d] # slice from array up to that index = d
    l2 = a[d:n] # remainder slice
    return l2 + l1 # inverted
# New Year Chaos

""" It is New Year's Day and people are in line for the Wonderland rollercoaster ride. Each person wears a sticker indicating their initial position in the queue from  to . Any person can bribe the person directly in front of them to swap positions, but they still wear their original sticker. One person can bribe at most two others.

Determine the minimum number of bribes that took place to get to a given queue order. Print the number of bribes, or, if anyone has bribed more than two people, print Too chaotic.

 """
def minimumBribes(q):
    # create variable to count moves
    # compare expected index with q[i] starting from the end.
    # increment moves depending on if i == i - 1 or i == i-2
    moves = 0
    for i in range(len(q)-1, 0, -1):
        if q[i] != i+1:
            if q[i-1] == i+1:
                moves += 1
                q[i-1], q[i] = q[i], q[i-1]
            elif q[i-2] == i+1:
                moves += 2
                q[i-2], q[i-1], q[i] = q[i-1], q[i], q[i-2]
            else:
                print("Too chaotic")
                return
    print(moves)


# Library fine
"""Your local library needs your help! Given the expected and actual return dates for a library book, create a program that calculates the fine (if any). The fee structure is as follows:

If the book is returned on or before the expected return date, no fine will be charged (i.e.: .
If the book is returned after the expected return day but still within the same calendar month and year as the expected return date, .
If the book is returned after the expected return month but still within the same calendar year as the expected return date, the .
If the book is returned after the calendar year in which it was expected, there is a fixed fine of .
Charges are based only on the least precise measure of lateness. For example, whether a book is due January 1, 2017 or December 31, 2017, if it is returned January 1, 2018, that is a year late and the fine would be .

Example


The first values are the return date and the second are the due date. The years are the same and the months are the same. The book is  days late. Return .

Function Description

Complete the libraryFine function in the editor below.

libraryFine has the following parameter(s):

d1, m1, y1: returned date day, month and year, each an integer
d2, m2, y2: due date day, month and year, each an integer
Returns

int: the amount of the fine or  if there is none """
def libraryFine(d1, m1, y1, d2, m2, y2):
    
    if y1 > y2:
        fine = 10000
    elif y1 == y2 and m1 > m2:
        fine = (m1-m2) * 500
    elif y1 == y2 and m1 == m2 and d1 > d2:
        fine = (d1-d2) * 15
    else:
        fine =0
    
    return fine
    
          
        
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


def digital_root(n):
    # iterate through n
#     sum of  x in n  keep add until len(res) = 1
    dig = str(n)
    res = 0
    i = 0
    while i < len(dig):
        for n in dig:
            res += int(n)
            i += 1
    if len(str(res)) > 1:
        return digital_root(res)
    else:
        return res

# First non-repeating character
""" Write a function named first_non_repeating_letter that takes a string input, and returns the first character that is not repeated anywhere in the string.

For example, if given the input 'stress', the function should return 't', since the letter t only occurs once in the string, and occurs first in the string.

As an added challenge, upper- and lowercase letters are considered the same character, but the function should return the correct case for the initial letter. For example, the input 'sTreSS' should return 'T'.

If a string contains all repeating characters, it should return an empty string ("") or None -- see sample tests. """

def first_non_repeating_letter(string):
    str_lst = [i.lower() for i in string]
    for i in range(len(str_lst)):
        if str_lst.count(str_lst[i]) == 1:
            return string[i]
    return ""
                
# Apple and Orange

""" Function Description

Complete the countApplesAndOranges function in the editor below. It should print the number of apples and oranges that land on Sam's house, each on a separate line.

countApplesAndOranges has the following parameter(s):

s: integer, starting point of Sam's house location.
t: integer, ending location of Sam's house location.
a: integer, location of the Apple tree.
b: integer, location of the Orange tree.
apples: integer array, distances at which each apple falls from the tree.
oranges: integer array, distances at which each orange falls from the tree. """

def countApplesAndOranges(s, t, a, b, apples, oranges):

    # rangeHouse = [s, t] 
    # a  # prefixA => position of tree Apples   
    # b  # prefixB=> position of tree Oranges
    # apples # => array where apples fell
    # oranges # => array where oranges fell
    # ansA = s <= (apples[i] + a) <= t
    # ansB = s <= (oranges[i] + b) <= t
    
    countA = 0
    countO = 0
    
  
    for i in range(len(apples)):
        if (apples[i] + a) >= s and (apples[i] + a) <= t:
            countA += 1
    for i in range(len(oranges)):
        if (oranges[i] + b) >= s and (oranges[i] + b) <= t:
            countO += 1

    print(countA)
    print(countO) 
