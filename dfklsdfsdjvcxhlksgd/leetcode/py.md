## 1.Two Sum 
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
            dic[target-n] = i    
```
## 2.Add Two Numbers
```
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

    def addTwoNumbers(self, l1, l2):
        carry = 0                       
        root = n = ListNode(0)          
        while l1 or l2 or carry:
            v1 = v2 = 0                 
            if l1:
                v1 = l1.var             
                l1 = l1.next            
            if l2:
                v2 = l2.var     
                l2 = l2.next            
            carry, var = divmod(v1+v1+carry, 10)
            n.next = ListNode(var)              
            n = n.next
        return root.next
``` 
    def addTwoNumbers(self, l1, l2):
        dummy = cur = ListNode(0)
        carry = 0
        while l1 or l2 or carry:
            if l1:
                carry += l1.var
                l1 = l1.next
            if l2:
                carry += l2.var
                l2 = l2.next
            cur.next = ListNode(carry%10)
            cur = cur.next
            carry //= 10
        return dummy.next

