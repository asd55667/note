#### 1.Two Sum 
>Brute Force
```
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        idxs = []
        for i in range(len(nums)):
            for j in range(len(nums)-i-1):
                if nums[i] + nums[i+j+1] == target:
                    idxs.append((i, i+j+1))
        return list(idxs[0])
```
>One-pass Hash Table    
```        
    def twoSum(self, nums, target):
        dic = {}
        for i, n in enumerate(nums): 
            if n in dic:
                return [dic[n], i]
            #给元素的匹配数创立键值, 当出现目标时返回双方索引    
            dic[target-n] = i  
```
# 2.Add Two Numbers
```
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

    def addTwoNumbers(self, l1, l2):
        dummy = cur = ListNode(0)  
        carry = 0
        while l1 or l2 or carry:
            if l1:
                carry += l1.val
                l1 = l1.next
            if l2:
                carry += l2.val
                l2 = l2.next
            cur.next = ListNode(carry%10)
            cur = cur.next 
            # 进位 0或1, 参与下一个位数的运算
            carry //= 10   
        return dummy.next #dummy与cur地址相同
```

# 3.Longest Substring Without Repeating Characters
```
    def lengthOfLongestSubstring(self, s):
        start = maxLength = 0
        usedChar = {}        
        for i in range(len(s)):
            # 字符重复且键值未重置时, 重置键值
            if s[i] in usedChar and start <= usedChar[s[i]]: 
                start = usedChar[s[i]] + 1
            else:
                maxLength = max(maxLength, i - start + 1)
            usedChar[s[i]] = i
        return maxLength
```

# 4. Median of Two Sorted Arrays
```
def findMedianSortedArrays(self, A, B):
    l = len(A) + len(B)
    if l % 2 == 1:
        return self.kth(A, B, l // 2)
    else:
        return (self.kth(A, B, l // 2) + self.kth(A, B, l // 2 - 1)) / 2.   
    
def kth(self, a, b, k):
    if not a:
        return b[k]
    if not b:
        return a[k]
    ia, ib = len(a) // 2 , len(b) // 2
    ma, mb = a[ia], b[ib]
    
    if ia + ib < k:
        if ma > mb:
            return self.kth(a, b[ib + 1:], k - ib - 1)
        else:
            return self.kth(a[ia + 1:], b, k - ia - 1)
    else:
        if ma > mb:
            return self.kth(a[:ia], b, k)
        else:
            return self.kth(a, b[:ib], k)
```
将数组A,B以中位数划分为A1,A2, B1,B2, 当A的中位数比B的大时, A2一定在数组C={A,B}的最右侧, C的中位数左侧的数只会在A1,B1,B2之中, 递归去掉两端的数

# 5. Longest Palindromic Substring