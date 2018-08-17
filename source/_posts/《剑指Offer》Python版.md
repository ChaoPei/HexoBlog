---
title: 《剑指Offer》Python版
date: 2018-07-29 15:02:01
tags: [算法]
categories: 算法
mathjax: true
---

### 1.二维数组中的查找

**题目：** 在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

**思路：**遍历每一行，查找该元素是否在该行之中。

```python
# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        for line in array:
            if target in line:
                return True
        return False

if __name__=='__main__':
    target=2
    array=[[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]]
    solution=Solution()
    ans=solution.Find(target,array)
    print(ans)
```

### 2.替换空格

**题目：** 请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

**思路：**利用字符串中的replace直接替换即可。

```python
# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        temp = s.replace(" ", "%20")
        return temp

if __name__=='__main__':
    s='We Are Happy'
    solution=Solution()
    ans=solution.replaceSpace(s)
    print(ans)
```

### 3.从尾到头打印链表

**题目：**输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。

**思路：**将链表中的值记录到list之中，然后进行翻转list。

```python
# -*- coding:utf-8 -*-
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        l=[]
        while listNode:
            l.append(listNode.val)
            listNode=listNode.next
        return l[::-1]

if __name__=='__main__':
    A1 = ListNode(1)
    A2 = ListNode(2)
    A3 = ListNode(3)
    A4 = ListNode(4)
    A5 = ListNode(5)

    A1.next=A2
    A2.next=A3
    A3.next=A4
    A4.next=A5

    solution=Solution()
    ans=solution.printListFromTailToHead(A1)
    print(ans)
```

### 4.重建二叉树

**题目：**输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

**题解：**首先前序遍历的第一个元素为二叉树的根结点，那么便能够在中序遍历之中找到根节点，那么在根结点左侧则是左子树，假设长度为M.在根结点右侧，便是右子树,假设长度为N。然后在前序遍历根节点后面M长度的便是左子树的前序遍历序列，再后面的N个长度便是右子树的后序遍历的长度。

```python
# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if len(pre)==0:
            return None
        if len(pre)==1:
            return TreeNode(pre[0])
        else:
            flag=TreeNode(pre[0])
            flag.left=self.reConstructBinaryTree(pre[1:tin.index(pre[0])+1],tin[:tin.index(pre[0])])
            flag.right=self.reConstructBinaryTree(pre[tin.index(pre[0])+1:],tin[tin.index(pre[0])+1:])
        return flag

if __name__=='__main__':
    solution=Solution()
    pre=list(map(int,input().split(',')))
    tin=list(map(int,input().split(',')))
    ans=solution.reConstructBinaryTree(pre,tin)
    print(ans.val)
```

### 5.用两个栈实现队列

**题目：**用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。

**题解：**申请两个栈Stack1和Stack2，Stack1当作输入，Stack2当作pop。当Stack2空的时候，将Stack1进行反转，并且输入到Stack2。

```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.Stack1=[]
        self.Stack2=[]
    def push(self, node):
        # write code here
        self.Stack1.append(node)
    def pop(self):
        # return xx
        if self.Stack2==[]:
            while self.Stack1:
                self.Stack2.append(self.Stack1.pop())
            return self.Stack2.pop()
        return self.Stack2.pop()

if __name__=='__main__':
    solution = Solution()
    solution.push(1)
    solution.push(2)
    solution.push(3)
    print(solution.pop())
```

### 6.旋转数组的最小数字

**题目：**把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

**题解：**遍历数组寻找数组最小值。

```python
# -*- coding:utf-8 -*-
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        minnum=999999
        for i in range(0,len(rotateArray)):
            if minnum>rotateArray[i]:
                minnum=rotateArray[i]
        if minnum:
            return minnum
        else:
            return 0

if __name__=='__main__':
    solution=Solution()
    rotateArray=list(map(int,input().split(',')))
    ans=solution.minNumberInRotateArray(rotateArray)
    print(ans)
```

### 7.斐波那契数列

**题目：**大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项。n<=39。

**题解：**递归和非递归方法。

```python
# -*- coding:utf-8 -*-
class Solution:
    def Fibonacci(self, n):
        # write code here
        if n==0:
            return 0
        if n==1:
            return 1
        Fib=[0 for i in range(0,n+1)]
        Fib[0],Fib[1]=0,1
        for i in range(2,n+1):
            Fib[i]=Fib[i-1]+Fib[i-2]
        return Fib[n]
    def Fibonacci1(self,n):
        if n==0:
            return 0
        if n==1 or n==2:
            return 1
        else:
            return self.Fibonacci1(n-1)+self.Fibonacci1(n-2)

if __name__=='__main__':
    solution=Solution()
    n=int(input())
    ans=solution.Fibonacci1(n)
    print(ans)
```

### 8.跳台阶

**题目：**一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

**题解：**ans[n]=ans[n-1]+ans[n-2]

```python
class Solution:
    def jumpFloor(self, number):
        # write code here
        if number==0:
            return 0
        if number==1:
            return 1
        if number==2:
            return 2
        ans=[0 for i in range(0,number+1)]
        ans[1],ans[2]=1,2
        for i in range(3,number+1):
            ans[i]=ans[i-1]+ans[i-2]
        return ans[number]


if __name__ == '__main__':
    solution = Solution()
    n=int(input())
    ans=solution.jumpFloor(n)
    print(ans)
```

### 9.变态跳台阶

**题目：**一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

**题解：**ans[n]=ans[n-1]+ans[n-2]+ans[n-3]+...+ans[n-n]，ans[n-1]=ans[n-2]+ans[n-3]+...+ans[n-n]，ans[n]=2*ans[n-1]。

```python
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloorII(self, number):
        # write code here
        if number==1:
            return 1
        if number==2:
            return 2
        return 2*self.jumpFloorII(number-1)

if __name__=='__main__':
    solution=Solution()
    n=int(input())
    ans=solution.jumpFloorII(n)
    print(ans)
```

### 10.矩形覆盖

**题目：**我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

**题解：**新增加的小矩阵竖着放，则方法与n-1时相同，新增加的小矩阵横着放，则方法与n-2时相同，于是f(n)=f(n-1)+f(n-2)。

```python
# -*- coding:utf-8 -*-
class Solution:
    def rectCover(self, number):
        # write code here
        if number==0:
            return 0
        if number==1:
            return 1
        Fib=[0 for i in range(0,number+1)]
        Fib[1],Fib[2]=1,2
        for i in range(3,number+1):
            Fib[i]=Fib[i-1]+Fib[i-2]
        return Fib[number]

if __name__=='__main__':
    solution=Solution()
    n=int(input())
    ans=solution.rectCover(n)
    print(ans)
```

### 11.二进制中1的个数

**题目：**输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。

**题解：**每次进行左移一位，然后与1进行相与，如果是1则进行加1。

```python
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1(self, n):
        # write code here
        count = 0
        for i in range(32):
            count += (n >> i) & 1
        return count

if __name__=='__main__':
    solution=Solution()
    n=int(input())
    ans=solution.NumberOf1(n)
    print(ans)
```

### 12.数值的整次方

**题目：**给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。

```python
# -*- coding:utf-8 -*-
class Solution:
    def Power(self, base, exponent):
        # write code here
        ans=1
        for i in range(0,abs(exponent)):
            ans=ans*base
        if exponent>0:
            return ans
        else:
            return 1/ans

if __name__=='__main__':
    solution=Solution()
    base=float(input())
    exponent=int(input())
    ans=solution.Power(base,exponent)
    print(ans)
```

### 13.调整数组顺序使奇数位于偶数前面

**题目：**输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

**题解：**申请奇数数组和偶数数组，分别存放奇数值和偶数值，数组相加便为结果。

```python
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        # write code here
        array1=[]#奇数
        array2=[]#偶数

        for i in range(0,len(array)):
            if array[i]%2!=0:
                array1.append(array[i])
            else:
                array2.append(array[i])
        ans=array1+array2
        return ans

if __name__=='__main__':
    solution=Solution()
    array=list(map(int,input().split(',')))
    ans=solution.reOrderArray(array)
    print(ans)
```

### 14.链表中倒数第K个节点

**题目：**输入一个链表，输出该链表中倒数第k个结点。

**题解：**反转链表，寻找第K个节点。

```python
# -*- coding:utf-8 -*-
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        #反转链表
        if head is None or head.next is None:
            return head
        pre=None #指向上一个节点
        while head:
            #先用temp保存当前节点的下一个节点信息
            temp=head.next
            #保存好next之后，便可以指向上一个节点
            head.next=pre
            #让pre,head指向下一个移动的节点
            pre=head
            head=temp
        # 寻找第K个元素的位置
        for i in range(1,k):
            pre=pre.next
        temp=pre
        return temp

if __name__=='__main__':
    solution=Solution()
    k=3
    p1=ListNode(1)
    p2=ListNode(2)
    p3=ListNode(3)
    p4=ListNode(4)
    p1.next=p2
    p2.next=p3
    p3.next=p4

    ans=solution.FindKthToTail(p1,k)
    print(ans.val)
```

### 15.反转链表

**题目：**输入一个链表，反转链表后，输出新链表的表头。

```python
# -*- coding:utf-8 -*-
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        if pHead is None or pHead.next is None:
            return pHead
        pre=None
        while pHead:
            #暂存当前节点的下一个节点信息
            temp=pHead.next
            #反转节点
            pHead.next=pre
            #进行下一个节点
            pre = pHead
            pHead=temp
        return pre

if __name__=='__main__':
    solution=Solution()
    p1=ListNode(1)
    p2=ListNode(2)
    p3=ListNode(3)
    p1.next=p2
    p2.next=p3
    ans=solution.ReverseList(p1)
    print(ans.val)
```

### 16.合并两个排序的列表

**题目：**输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

**题解：**将两个链表之中的数值转换到列表之中，并进行排序，将排序后的列表构造成链表。

```python
# -*- coding:utf-8 -*-
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    # 返回合并后列表
    def Merge(self,pHead1,pHead2):
        # write code here
        if pHead1 is None and pHead2 is None:
            return None
        num1,num2=[],[]
        while pHead1:
            num1.append(pHead1.val)
            pHead1=pHead1.next
        while pHead2:
            num2.append(pHead2.val)
            pHead2=pHead2.next
        ans=num1+num2
        ans.sort()
        head=ListNode(ans[0])
        pre=head
        for i in range(1,len(ans)):
            node=ListNode(ans[i])
            pre.next=node
            pre=pre.next
        return head

if __name__=='__main__':
    solution=Solution()
    pHead1_1 = ListNode(1)
    pHead1_2 = ListNode(3)
    pHead1_3 = ListNode(5)
    pHead1_1.next=pHead1_2
    pHead1_2.next=pHead1_3

    pHead2_1 = ListNode(2)
    pHead2_2 = ListNode(4)
    pHead2_3 = ListNode(6)
    pHead2_1.next=pHead2_2
    pHead2_2.next=pHead2_3
    ans=solution.Merge(pHead1_1,pHead2_1)
    print(ans)
```

### 17.树的子结构

**题目：**输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）。

**题解：**将树转变为中序序列，然后转变为str类型，最后判断是否包含。

```python
# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        if pRoot1 is None or pRoot2 is None:
            return False
        pRoot1_result,pRoot2_result=[],[]
        self.order_traversal(pRoot1,pRoot1_result)
        self.order_traversal(pRoot2,pRoot2_result)
        str1=''.join(str(i) for i in pRoot1_result)
        str2=''.join(str(i) for i in pRoot2_result)
        print(str1,str2)
        if str2 in str1:
            return True
        else:
            return False

    def order_traversal(self,root,result):
        if not root:
            return
        self.order_traversal(root.left,result)
        result.append(root.val)
        self.order_traversal(root.right,result)

if __name__=='__main__':
    solution=Solution()
    pRootA1 = TreeNode(1)
    pRootA2 = TreeNode(2)
    pRootA3 = TreeNode(3)
    pRootA4 = TreeNode(4)
    pRootA5 = TreeNode(5)
    pRootA1.left=pRootA2
    pRootA1.right=pRootA3
    pRootA2.left=pRootA4
    pRootA2.right=pRootA5

    pRootB2 = TreeNode(2)
    pRootB4 = TreeNode(4)
    pRootB5 = TreeNode(5)
    pRootB2.left=pRootB4
    pRootB2.right = pRootB5
    ans=solution.HasSubtree(pRootA1,pRootB2)
    print(ans)
```

### 18.二叉树的镜像

**题目：** 操作给定的二叉树，将其变换为源二叉树的镜像。

**输入描述：**

​	源二叉树
          8
         /  \
        6   10
       / \  / \
      5  7 9 11
      镜像二叉树
          8
         /  \
        10   6
       / \  / \
      11 9 7  5

**思路：**递归实现反转每个子节点

```python
# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        # write code here
        # A1_order_result=[]
        # self.order_traversal(A1,A1_order_result)
        if root is None:
            return
        if root.left is None and root.right is None:
            return
        temp=root.left
        root.left=root.right
        root.right=temp

        if root is not None:
            self.Mirror(root.left)
        if root is not None:
            self.Mirror(root.right)

    def order_traversal(self,root,result):
        if not root:
            return
        self.order_traversal(root.left,result)
        result.append(root.val)
        self.order_traversal(root.right,result)

if __name__=='__main__':
    A1 = TreeNode(8)
    A2 = TreeNode(6)
    A3 = TreeNode(10)
    A4 = TreeNode(5)
    A5 = TreeNode(7)
    A6 = TreeNode(9)
    A7 = TreeNode(11)
    A1.left=A2
    A1.right=A3
    A2.left=A4
    A2.right=A5
    A3.left=A6
    A3.right=A7

    temp1=[]
    solution=Solution()
    solution.order_traversal(A1,temp1)
    print(temp1)
    solution.Mirror(A1)
    solution.order_traversal(A1,temp1)
    print(temp1)
```

### 19.顺时针打印矩阵

**题目：**

> ```python
> 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，
> 例如，如果输入如下矩阵：
>  1 2 3 4
>  5 6 7 8
>  9 10 11 12
>  13 14 15 16
> 则依次打印出数字
> 1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
> 
> 
> ```

**思路：**每次打印圈，但要判断最后一次是打印横还是竖，另外判断数据是否已存在。

```python
# -*- coding:utf-8 -*-
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        # write code here
        m,n=len(matrix),len(matrix[0])
        res = []
        if n==1 and m==1:
            res.append(matrix[0][0])
            return res
        for k in range(0,(min(m,n)+1)//2):
            [res.append(matrix[k][i]) for i in range(k, n - k)]
            [res.append(matrix[j][n-k-1]) for j in range(k,m-k) if matrix[j][n-k-1] not in res]
            [res.append(matrix[m-k-1][j]) for j in range(n-k-1,k-1,-1) if matrix[m-k-1][j] not in res]
            [res.append(matrix[j][k]) for j in range(m-1-k,k-1,-1) if matrix[j][k] not in res]
        return res

if __name__=='__main__':
    solution=Solution()
    m,n=1,5
    matrix=[]
    for i in range(0,m):
        matrix.append(list(map(int,input().split(' '))))
    print(matrix)
    ans=solution.printMatrix(matrix)
    print(ans)
```

### 20.包含Min函数的栈

**题目：**定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数。

```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.num=[]
    def push(self, node):
        # write code here
        self.num.append(node)
    def pop(self):
        # write code here
        self.num.pop()
    def top(self):
        # write code here
        numlen = len(self.num)
        return self.num[numlen-1]
    def min(self):
        # write code here
        return min(self.num)

if __name__=='__main__':
    solution = Solution()
    solution.push(1)
    solution.push(2)
    solution.push(3)
    solution.push(4)
    solution.pop()
    print(solution.top())
    print(solution.min())			
```

### 21.栈的压入弹出序列

**题目：**输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）。

**题解：**新构建一个中间栈，来模拟栈的输入和栈的输出，比对输入结果和输出结果是否相等。

```python
# -*- coding:utf-8 -*-
class Solution:
    def IsPopOrder(self, pushV, popV):
        # write code here
        if len(pushV)==1 and len(popV)==1 and pushV[0]!=popV[0]:
            return False

        helpV=[]
        pushV.reverse()
        popV.reverse()
        #模拟给定栈的压入和压出
        helpV.append(pushV[len(pushV)-1])
        pushV.pop()
        while True:
            if helpV[len(helpV)-1]!=popV[len(popV)-1]:
                helpV.append(pushV[len(pushV)-1])
                pushV.pop()

            if helpV[len(helpV)-1]==popV[len(popV)-1]:
                helpV.pop()
                popV.pop()

            if pushV==[] and popV==[] and helpV==[]:
                return True

            if pushV==[] and popV[len(popV)-1]!=helpV[len(helpV)-1]:
                return False


if __name__=='__main__':
    solution=Solution()
    push=list(map(int,input().split(' ')))
    pop=list(map(int,input().split(' ')))
    ans=solution.IsPopOrder(push,pop)
    print(ans)
```

### 22.从上往下打印二叉树

**题目：**从上往下打印出二叉树的每个节点，同层节点从左至右打印。

**思路：**递归，每次将左子树结果和右子树结果存到结果集之中。

```python
# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        if root is None:
            return []
        ans=[]
        ans.append(root.val)
        self.orderans(root,ans)
        return ans

    def orderans(self,root,ans):
        if not root:
            return
        if root.left:
            ans.append(root.left.val)
        if root.right:
            ans.append(root.right.val)

        self.orderans(root.left, ans)
        self.orderans(root.right,ans)

if __name__=='__main__':
    solution=Solution()
    A1 = TreeNode(1)
    A2 = TreeNode(2)
    A3 = TreeNode(3)
    A4 = TreeNode(4)
    A5 = TreeNode(5)

    A1.left=A2
    A1.right=A3
    A2.left=A4
    A2.right=A5
    ans=solution.PrintFromTopToBottom(A1)
    print(ans)
```

### 23.二叉树的后续遍历序列

**题目：**输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

**思路：**二叉搜索树的特性是所有左子树值都小于中节点，所有右子树的值都大于中节点，递归遍历左子树和右子树的值。

```python
# -*- coding:utf-8 -*-
class Solution:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if not sequence:
            return False
        if len(sequence)==1:
            return True
        i=0
        while sequence[i]<sequence[-1]:
            i=i+1
        k=i
        for j in range(i,len(sequence)-1):
            if sequence[j]<sequence[-1]:
                return False
            
        leftsequence=sequence[:k]
        rightsequence=sequence[k:len(sequence)-1]

        leftans=True
        rightans=True

        if len(leftsequence)>0:
            self.VerifySquenceOfBST(leftsequence)
        if len(rightsequence)>0:
            self.VerifySquenceOfBST(rightsequence)

        return leftans and rightans

if __name__=='__main__':
    solution=Solution()
    num=list(map(int,input().split(' ')))
    ans=solution.VerifySquenceOfBST(num)
    print(ans)
```

### 24.二叉树中和为某一值的路径

**题目：**输入一颗二叉树的根节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)。

**思路：**利用递归的方法，计算加左子树和右子树之后的值，当参数较多是，可以将结果添加到函数变量之中。

```python
# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        # write code here
        if not root:
            return []
        ans=[]
        path=[]
        self.dfs(root,expectNumber,ans,path)
        ans.sort()
        return ans

    def dfs(self,root,target,ans,path):
        if not root:
            return

        path.append(root.val)
        if root.left is None and root.right is None and target==root.val:
            ans.append(path[:])

        if root.left:
            self.dfs(root.left,target-root.val,ans,path)
        if root.right:
            self.dfs(root.right,target-root.val,ans,path)

        path.pop()


if __name__=='__main__':
    A1=TreeNode(10)
    A2=TreeNode(8)
    A3=TreeNode(12)
    A4=TreeNode(4)
    A5=TreeNode(2)
    A6=TreeNode(2)

    A1.left=A2
    A1.right=A3
    A2.left=A4
    A2.right=A5
    A5.left=A6

    expectNumber=22
    solution=Solution()
    ans=solution.FindPath(A1,expectNumber)
    print(ans)
```

### 25.复杂链表的复制

**题目：**输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）。

**思路：**将大问题转变为小问题，每次都进行复制头部节点，然后进行递归，每次同样处理头部节点。

```python
# -*- coding:utf-8 -*-
class RandomListNode:
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None

class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        # write code here
        # 复制头部节点
        if pHead is None:
            return None

        newHead=RandomListNode(pHead.label)
        newHead.next=pHead.next
        newHead.random=pHead.random

        # 递归其他节点
        newHead.next=self.Clone(pHead.next)

        return newHead


if __name__=='__main__':
    A1=RandomListNode(2)
    A2=RandomListNode(3)
    A3=RandomListNode(4)
    A4=RandomListNode(5)
    A5=RandomListNode(6)

    A1.next=A2
    A1.random=A3

    A2.next=A3
    A2.random=A4

    A3.next=A4
    A3.random=A5

    A4.next=A5
    A4.random=A3

    solution=Solution()
    ans=solution.Clone(A1)
```

### 26.二叉搜索树与双向列表

**题目：**输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

**思路：**递归将根结点和左子树的最右节点和右子树的最左节点进行连接起来。

```python
# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def Convert(self, pRootOfTree):
        # write code here
        if pRootOfTree is None:
            return pRootOfTree
        if pRootOfTree.left is None and pRootOfTree.right is None:
            return pRootOfTree

        #处理左子树
        self.Convert(pRootOfTree.left)
        left=pRootOfTree.left

        if left:
            while left.right:
                left=left.right
            pRootOfTree.left,left.right=left,pRootOfTree

        #处理右子树
        self.Convert(pRootOfTree.right)
        right=pRootOfTree.right

        if right:
            while right.left:
                right=right.left
            pRootOfTree.right,right.left=right,pRootOfTree

        while pRootOfTree.left:
            pRootOfTree=pRootOfTree.left
        return pRootOfTree


if __name__=='__main__':
    A1 = TreeNode(7)
    A2 = TreeNode(5)
    A3 = TreeNode(15)
    A4 = TreeNode(2)
    A5 = TreeNode(6)
    A6 = TreeNode(8)
    A7 = TreeNode(19)
    A8 = TreeNode(24)

    A1.left=A2
    A1.right=A3
    A2.left=A4
    A2.right=A5
    A3.left=A6
    A3.right=A7
    A7.right=A8

    solution=Solution()
    solution.Convert(A1)
```

### 27.字符串的排列

**题目：**输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。

**输入：**输入一个字符串,长度不超过9(可能有字符重复),字符只包括大小写字母。

**思路：**通过将第k位的字符提取到最前面，然后进行和后面的每个字符进行交换，得到所有结果集。

```python
# -*- coding:utf-8 -*-
class Solution:
    def Permutation(self, ss):
        # write code here
        if not ss:
            return []
        res=[]
        self.helper(ss,res,'')
        return sorted(list(set(res)))

    def helper(self,ss,res,path):
        if not ss:
            res.append(path)
        else:
            for i in range(0,len(ss)):
                self.helper(ss[:i]+ss[i+1:],res,path+ss[i])

if __name__=='__main__':
    str='abbcDeefg'
    str1='abbc'
    solution=Solution()
    ans=solution.Permutation(str1)
    print(ans)
```

### 28.数组中出现次数超过一般的数字

**题目：**数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0

**题解：**利用list列表来存放每个数出现的次数ans[numbers[i]]=ans[numbers[i]]+1。

```python
# -*- coding:utf-8 -*-
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        numlen=len(numbers)
        halflen=numlen//2
        maxans=0
        ans=[0 for i in range(0,1000)]
        for i in range(0,len(numbers)):
            ans[numbers[i]]=ans[numbers[i]]+1
            if ans[numbers[i]]>maxans:
                maxans=numbers[i]
        ans.sort()
        ans.reverse()
        res=ans[0]
        if res>halflen:
            return maxans
        else:
            return 0


if __name__=='__main__':
    num=list(map(int,input().split(',')))
    solution=Solution()
    ans=solution.MoreThanHalfNum_Solution(num)
    print(ans)
```

### 29.最小的K个数

**题目：**输入n个整数，找出其中最小的K个数，例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4。

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        if k>len(tinput):
            return []
        tinput.sort()
        return tinput[:k]

if __name__=='__main__':
    num=list(map(int,input().split(',')))
    k=int(input())
    solution=Solution()
    ans=solution.GetLeastNumbers_Solution(num,k)
    print(ans)
```

### 30.连续子数组的最大和

**题目：**HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。你会不会被他忽悠住？(子向量的长度至少是1)

```python
# -*- coding:utf-8 -*-
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        # write code here
        maxsum,tempsum=array[0],array[0]
        for i in range(1,len(array)):
            if tempsum<0:
                tempsum=array[i]
            else:
                tempsum = tempsum + array[i]
            if tempsum>maxsum:
                maxsum=tempsum
        return maxsum

if __name__=='__main__':
    array=list(map(int,input().split(',')))
    solution=Solution()
    ans=solution.FindGreatestSumOfSubArray(array)
    print(ans)
```

### 31.整数中1出现的次数

**题目：**求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

**思路：**对每个数字的每位进行分解，含有1则结果加1。

```python
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        # write code here
        ans=0
        for i in range(1,n+1):
            tempans=0
            while i!=0:
                eachnum=i%10
                i=i//10
                if eachnum==1:
                    tempans=tempans+1
            ans=ans+tempans
        return ans

if __name__=='__main__':
    n=130
    solution=Solution()
    ans=solution.NumberOf1Between1AndN_Solution(n)
    print(ans)
```

### 32.把数组排成最小的数

**题目：**输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

**思路：**将数组转换成字符串之后，进行两两比较字符串的大小，比如3,32的大小由332和323确定，即3+32和32+3确定。

```python
# -*- coding:utf-8 -*-
class Solution:
    def PrintMinNumber(self, numbers):
        # write code here
        if not numbers:
            return ""
        num = map(str, numbers)
        for i in range(0,len(numbers)):
            for j in range(i,len(numbers)):
                if int(str(numbers[i])+str(numbers[j]))>int(str(numbers[j])+str(numbers[i])):
                    numbers[i],numbers[j]=numbers[j],numbers[i]
        ans=''
        for i in range(0,len(numbers)):
            ans=ans+str(numbers[i])
        return ans

if __name__=='__main__':
    numbers=[3,32,321]
    solution=Solution()
    ans=solution.PrintMinNumber(numbers)
    print(ans)
```

### 33.丑数

**题目：**把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

**思路：**每一个丑数必然是由之前的某个丑数与2，3或5的乘积得到的，这样下一个丑数就用之前的丑数分别乘以2，3，5，找出这三这种最小的并且大于当前最大丑数的值，即为下一个要求的丑数。

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetUglyNumber_Solution(self, index):
        # write code here
        if (index <= 0):
            return 0
        uglyList = [1]
        indexTwo = 0
        indexThree = 0
        indexFive = 0
        for i in range(index-1):
            newUgly = min(uglyList[indexTwo]*2, uglyList[indexThree]*3, uglyList[indexFive]*5)
            uglyList.append(newUgly)
            if (newUgly % 2 == 0):
                indexTwo += 1
            if (newUgly % 3 == 0):
                indexThree += 1
            if (newUgly % 5 == 0):
                indexFive += 1
        return uglyList[-1]

if __name__=='__main__':
    solution=Solution()
    index=200
    ans=solution.GetUglyNumber_Solution(index)
    print(ans)
```

### 34.第一个只出现一次的字符

**题目：**在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1。

**思路：**找出所有出现一次的字符，然后进行遍历找到第一次出现字符的位置。

```python
# -*- coding:utf-8 -*-
class Solution:
    def FirstNotRepeatingChar(self, s):
        # write code here
        if not s:
            return -1
        sset=set(s)
        dict={}
        for c in sset:
            dict[c]=0
        for i in range(0,len(s)):
            dict[s[i]]=dict[s[i]]+1
        onetime=[]
        for c in dict:
            if dict[c]==1:
                onetime.append(c)

        if onetime is None:
            return -1
        else:
            index=0
            for i in range(0,len(s)):
                if s[i] in onetime:
                    index=i
                    break
            return index

if __name__=='__main__':
    s='abbddebbac'
    solution=Solution()
    ans=solution.FirstNotRepeatingChar(s)
    print(ans)
```

### 35.数组中的逆序对

**题目描述：**在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007。

**输入描述：**题目保证输入的数组中没有的相同的数字。

**数据范围：**
   对于%50的数据,size<=10^4
   对于%75的数据,size<=10^5
   对于%100的数据,size<=2*10^5

> 示例1
>
> 输入 1,2,3,4,5,6,7,0
>
> 输出 7

```python
# -*- coding:utf-8 -*-
class Solution:
    def InversePairs(self, data):
        # write code here
        global count
        count = 0

        def A(array):
            global count
            if len(array) <= 1:
                return array
            k = int(len(array) / 2)
            left = A(array[:k])
            right = A(array[k:])
            l = 0
            r = 0
            result = []
            while l < len(left) and r < len(right):
                if left[l] < right[r]:
                    result.append(left[l])
                    l += 1
                else:
                    result.append(right[r])
                    r += 1
                    count += len(left) - l
            result += left[l:]
            result += right[r:]
            return result

        A(data)
        return count % 1000000007

if __name__=='__main__':
    data=[1,2,3,4,5,6,7,0]
    solution=Solution()
    ans=solution.InversePairs(data)
    print(ans)
```

### 36.两个链表的第一个公共节点

**题目：**输入两个链表，找出它们的第一个公共结点。

```python
# -*- coding:utf-8 -*-
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        list1 = []
        list2 = []
        node1 = pHead1
        node2 = pHead2
        while node1:
            list1.append(node1.val)
            node1 = node1.next
        while node2:
            if node2.val in list1:
                return node2
            else:
                node2 = node2.next

if __name__=='__main__':
    A1 = ListNode(1)
    A2 = ListNode(2)
    A3 = ListNode(3)
    A1.next=A2
    A2.next=A3

    B4 = ListNode(4)
    B5 = ListNode(5)
    B4.next=B5

    C6=ListNode(6)
    C7=ListNode(7)

    A3.next=C6
    B5.next=C6
    C6.next=C7

    solution=Solution()
    ans=solution.FindFirstCommonNode(A1,B4)
    print(ans.val)
```

### 37.数字在排序数组中出现的次数

**题目：**统计一个数字在排序数组中出现的次数。

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        ans=0
        for i in range(0,len(data)):
            if data[i]==k:
                ans=ans+1
            if data[i]>k:
                break
        return ans

if __name__=='__main__':
    data=[1,2,3,3,3,4,4,5]
    k=3
    solution=Solution()
    ans=solution.GetNumberOfK(data,k)
    print(ans)
```

### 38.二叉树的深度

**题目：**输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

```python
# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def TreeDepth(self, pRoot):
        # write code here
        if pRoot is None:
            return 0
        left=self.TreeDepth(pRoot.left)
        right=self.TreeDepth(pRoot.right)
        print(left,right)
        return max(left,right)+1

if __name__=='__main__':
    A1 = TreeNode(1)
    A2 = TreeNode(2)
    A3 = TreeNode(3)
    A4 = TreeNode(4)
    A5 = TreeNode(5)
    A6 = TreeNode(6)

    A1.left=A2
    A1.right=A3
    A2.left=A4
    A2.right=A5
    A4.left=A6

    solution=Solution()
    ans=solution.TreeDepth(A1)
    print('ans=',ans)
```

### 39.平衡二叉树

**题目：**输入一棵二叉树，判断该二叉树是否是平衡二叉树。

**题解：**平衡二叉树是左右子数的距离不能大于1，因此递归左右子树，判断子树距离是否大于1。

```python
# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def IsBalanced_Solution(self, pRoot):
        # write code here
        if pRoot is None:
            return True
        if abs(self.TreeDepth(pRoot.left)-self.TreeDepth(pRoot.right))>1:
            return False
        return self.IsBalanced_Solution(pRoot.left) and self.IsBalanced_Solution(pRoot.right)

    def TreeDepth(self,root):
        if root is None:
            return 0
        left=self.TreeDepth(root.left)
        right=self.TreeDepth(root.right)
        return max(left+1,right+1)

if __name__=='__main__':
    A1 = TreeNode(1)
    A2 = TreeNode(2)
    A3 = TreeNode(3)
    A4 = TreeNode(4)
    A5 = TreeNode(5)
    A6 = TreeNode(6)

    A1.left=A2
    A1.right=A3
    A2.left=A4
    A2.right=A5
    #A4.left=A6

    solution=Solution()
    ans=solution.IsBalanced_Solution(A1)
    print(ans)
```

### 40.数组中只出现一次的数字

**题目：**一个整型数组里除了两个数字之外，其他的数字都出现了偶数次。请写程序找出这两个只出现一次的数字。

**题解：**将数组中数转到set之中，然后利用dict存储每个数字出现的次数。

```python
# -*- coding:utf-8 -*-
class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
        # write code here
        arrayset=set(array)
        dict={}
        for num in arrayset:
            dict[num]=0
        for i in range(0,len(array)):
            dict[array[i]]=dict[array[i]]+1
        ans=[]
        for num in arrayset:
            if dict[num]==1:
                ans.append(num)
        return ans


if __name__=='__main__':
    array=[1,1,2,2,3,3,4,5,5,6,7,7]
    solution=Solution()
    ans=solution.FindNumsAppearOnce(array)
    print(ans)
```
### 41.和为S的连续正整数序列

**题目：**小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!

**输出描述：**输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序。

**思路：**首项加尾项*2等于和，那么只要遍历项的开始和长度即可。

```python
# -*- coding:utf-8 -*-
class Solution:
    def FindContinuousSequence(self, tsum):
        # write code here
        ans=[]
        for i in range(1,tsum//2+1):
            oneans=[]
            for k in range(1,tsum):
                tempsum=((i+i+k-1)*k)//2
                if tempsum==tsum:
                    for j in range(i,i+k):
                        oneans.append(j)
                    break
            if oneans !=[]:
                ans.append(oneans)
        return ans

if __name__=='__main__':
    tsum=15
    solution=Solution()
    ans=solution.FindContinuousSequence(tsum)
    print(ans)
```

### 42.和为S的两个数字

**题目：**输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

**输出描述：**对应每个测试案例，输出两个数，小的先输出。

**思路：**利用i和j从后面进行扫描结果，选取最小的乘积放入到结果集之中。

```python
# -*- coding:utf-8 -*-
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        ans=[]
        i,j,minres=0,len(array)-1,1000000
        for i in range(0,len(array)-1):
            j=len(array)-1
            while True:
                tempsum = array[i] + array[j]
                if tempsum == tsum:
                    if array[i]*array[j]<minres:
                        ans=[]
                        ans.append(array[i])
                        ans.append(array[j])
                        minres=array[i]*array[j]
                    break
                else:
                    j = j - 1
                if tempsum<tsum:
                    break
                if j<=i:
                    break
        return ans

if __name__=='__main__':
    array=[1,2,4,7,11,15]
    tsum=15
    solution=Solution()
    ans=solution.FindNumbersWithSum(array,tsum)
    print(ans)
```

### 43.左旋字符子串

**题目：**汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！

```python
# -*- coding:utf-8 -*-
class Solution:
    def LeftRotateString(self, s, n):
        # write code here
        if s=='' and n==0:
            return ''
        ans=''
        ans=s[n:]+s[0:n]
        return ans

if __name__=='__main__':
    s='abcdefg'
    n=2
    solution=Solution()
    ans=solution.LeftRotateString(s,n)
    print(ans)
```

### 44.反转单词顺序

**题目：**牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？

```python
# -*- coding:utf-8 -*-
class Solution:
    def ReverseSentence(self, s):
        # write code here
        ans,word=[],''
        for i in range(0,len(s)):
            word = word + s[i]
            if s[i]==' ':
                ans.append(word)
                word=''
            if i==len(s)-1:
                word=word+' '
                ans.append(word)
        ans.reverse()
        res=''
        for c in ans:
            res=res+c
        return res[:len(res)-1]

if __name__=='__main__':
    solution=Solution()
    s='I am a student.'
    ans=solution.ReverseSentence(s)
    print(ans)
```

### 45.扑克牌顺序

**题目：**LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0。

```python
# -*- coding:utf-8 -*-
class Solution:
    def IsContinuous(self, numbers):
        # write code here
        if numbers==[]:
            return False
        numbers.sort()
        zero=0
        for i in range(0,len(numbers)):
            if numbers[i]==0:
                zero=zero+1
        for i in range(zero+1,len(numbers)):
            if numbers[i]==numbers[i-1]:
                return False
            if numbers[i]-numbers[i-1]==1:
                continue
            else:
                diff=numbers[i]-numbers[i-1]-1
                zero=zero-diff

        if zero<0:
            return False
        return True
    
if __name__=='__main__':
    numbers=[1,0,0,1,0]
    solution=Solution()
    ans=solution.IsContinuous(numbers)
    print(ans)
```

### 46.孩子们的圈圈(圈圈中最后剩下的数)

**题目：**每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)。

**思路：**约瑟夫环问题。

```python
# 题目
# 每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。
# 其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。
# 每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,
# 从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,
# 并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)

# 思路
# 约瑟夫环问题

# -*- coding:utf-8 -*-
class Solution:
    def LastRemaining_Solution(self, n, m):
        # write code here
        if n<1 or m<1:
            return -1
        last=0
        for i in range(2,n+1):
            last=(last+m)%i
        return last

if __name__=='__main__':
    n,m=8,4
    solution=Solution()
    ans=solution.LastRemaining_Solution(n,m)
    print(ans)
```

### 47.求1+2+3+...+n

**题目：**求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

**思路：**利用递归当作计算结果。

```python
# -*- coding:utf-8 -*-
class Solution:
    def Sum_Solution(self, n):
        # write code here
        if n==0:
            return 0
        return self.Sum_Solution(n-1)+n

if __name__=='__main__':
    n=6
    solution=Solution()
    ans=solution.Sum_Solution(n)
    print(ans)
```

### 48.不用加减乘除做加法

**题目：**写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。

**思路：**二进制异或进位。

```python
# -*- coding:utf-8 -*-
class Solution:
    def Add(self, num1, num2):
        # write code here
        while num2!=0:
            sum=num1^num2
            carry=(num1&num2)<<1
            num1=sum
            num2=carry
        return num1

if __name__=='__main__':
    num1,num2=10,500000
    solution=Solution()
    ans=solution.Add(num1,num2)
    print(ans)
```

### 49.把字符串转换成整数

**题目：**将一个字符串转换成一个整数(实现Integer.valueOf(string)的功能，但是string不符合数字要求时返回0)，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0。

**输入描述：**输入一个字符串,包括数字字母符号,可以为空输出描述:如果是合法的数值表达则返回该数字，否则返回0。

```python
示例
+2147483647
    1a33
2147483647
    0
```

```python
# -*- coding:utf-8 -*-
class Solution:
    def StrToInt(self, s):
        # write code here
        if len(s) == 0:
            return 0
        else:
            if s[0] > '9' or s[0] < '0':
                a = 0
            else:
                a = int(s[0]) * 10 ** (len(s) - 1)
            if len(s) > 1:
                for i in range(1, len(s)):
                    if s[i] >= '0' and s[i] <= '9':
                        a = a + int(s[i]) * 10 ** (len(s) - 1 - i)
                    else:
                        return 0
        if s[0] == '+':
            return a
        if s[0] == '-':
            return -a
        return a

if __name__=='__main__':
    s='115'
    solution=Solution()
    ans=solution.StrToInt(s)
    print(ans)
```

### 50.数组中重复的数字

**题目：**在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

**思路：**利用dict计算重复数字。

```python
# -*- coding:utf-8 -*-
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        # write code here
        numset=set(numbers)
        dict={}
        duplication.append(0)
        for val in numbers:
            dict[val]=0
        for i in range(0,len(numbers)):
            dict[numbers[i]]=dict[numbers[i]]+1
        for val in numset:
            if dict[val]>1:
                duplication[0]=val
                return True
        return False

if __name__=='__main__':
    numbers=[2,1,3,1,4]
    solution=Solution()
    duplication=[]
    ans=solution.duplicate(numbers,duplication)
    print(ans)
```

### 51.构建乘积数组

```python
# 题目
# 给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],
# 其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。

# 思路
# 审题仔细 没有A[i]

# -*- coding:utf-8 -*-
class Solution:
    def multiply(self, A):
        # write code here
        B=[]
        for i in range(0,len(A)):
            temp=1
            for j in range(0,len(A)):
                if j==i:
                    continue
                temp=temp*A[j]
            B.append(temp)
        return B

if __name__=='__main__':
    solution=Solution()
    A=[1,2,3,4,5]
    ans=solution.multiply(A)
    print(ans)
```

### 52.正则表达式匹配

**题目：**请实现一个函数用来匹配包括'.'和'\*'的正则表达式。模式中的字符'.'表示任意一个字符，而'\*'表示它前面的字符可以出现任意次（包含0次）。在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab\*ac\*a"匹配，但是与"aa.a"和"ab*a"均不匹配。

**思路：**

> 当模式中的第二个字符不是`*`时： 
>
> - 如果字符串第一个字符和模式中的第一个字符相匹配，那么字符串和模式都后移一个字符，然后匹配剩余的。 
> - 如果字符串第一个字符和模式中的第一个字符相不匹配，直接返回false。

> 当模式中的第二个字符是`*`时：
>
> + 如果字符串第一个字符跟模式第一个字符不匹配，则模式后移2个字符，继续匹配。
> + 如果字符串第一个字符跟模式第一个字符匹配，可以有3种匹配方式。
>   + 模式后移2字符，相当于`x*`被忽略。即模式串中*与他前面的字符和字符串匹配0次。 
>   +  字符串后移1字符，模式后移2字符。即模式串中*与他前面的字符和字符串匹配1次。
>   + 字符串后移1字符，模式不变，即继续匹配字符下一位，因为`*`可以匹配多位。即模式串中*与他前面的字符和字符串匹配多次。

```python
# -*- coding:utf-8 -*-
class Solution:
    # s, pattern都是字符串
    def match(self, s, pattern):
        if s == pattern:
            return True
        if not pattern:
            return False
        if len(pattern) > 1 and pattern[1] == '*':
            if (s and s[0] == pattern[0]) or (s and pattern[0] == '.'):
                return self.match(s, pattern[2:]) \
                       or self.match(s[1:], pattern) \
                       or self.match(s[1:], pattern[2:])
            else:
                return self.match(s, pattern[2:])
        elif s and (s[0] == pattern[0] or pattern[0] == '.'):
            return self.match(s[1:], pattern[1:])
        return False

if __name__=='__main__':
    solution=Solution()
    s='aaa'
    pattern='a*a.a'
    ans=solution.match(s,pattern)
    print(ans)
```

### 53.表示数值的字符串

**题目：**请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。

```python
# -*- coding:utf-8 -*-
class Solution:
    # s字符串
    def isNumeric(self, s):
        # write code here
        # 标记符号、小数点、e是否出现过
        sign,decimal,hasE=False,False,False
        for i in range(0,len(s)):
            if s[i]=='e' or s[i]=='E':
                if i==len(s)-1:# e后面一定要接数字
                    return False
                if hasE==True:# 不能出现两次e
                    return False
                hasE=True
            elif s[i]=='+' or s[i]=='-':
                #第二次出现+或-一定要在e之后
                if sign and s[i-1]!='e' and s[i-1]!='E':
                    return False
                # 第一次出现+或-，如果不是出现在字符最前面，那么就要出现在e或者E后面
                if sign==False and i>0 and s[i-1]!='e' and s[i-1]!='E':
                    return False
                sign=True
            elif s[i]=='.':
                # e后面不能出现小数点，小数点不能出现两次
                if decimal or hasE:
                    return False
                decimal=True
            elif s[i]>'9' or s[i]<'0':
                return False
        return True

if __name__=='__main__':
    solution=Solution()
    s='123e.1416'
    ans=solution.isNumeric(s)
    print(ans)
```

### 54.字符流中第一个不重复的字符

**题目：**请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。

**输出描述：**如果当前字符流没有存在出现一次的字符，返回#字符。

```python
# -*- coding:utf-8 -*-
class Solution:
    # 返回对应char
    def __init__(self):
        self.all={}
        self.ch=[]
    def FirstAppearingOnce(self):
        # write code here
        if self.all is None:
            return '#'
        for c in self.ch:
            if self.all[c]==1:
                return c
        return '#'

    def Insert(self, char):
        # write code here
        self.ch.append(char)
        if char in self.all:
            self.all[char]=self.all[char]+1
        else:
            self.all[char]=1

if __name__=='__main__':
    solution=Solution()
    solution.Insert('g')
    ans = solution.FirstAppearingOnce()
    print(ans)
    solution.Insert('o')
    ans = solution.FirstAppearingOnce()
    print(ans)
    solution.Insert('o')
    ans = solution.FirstAppearingOnce()
    print(ans)
    solution.Insert('g')
    ans = solution.FirstAppearingOnce()
    print(ans)
    solution.Insert('l')
    ans = solution.FirstAppearingOnce()
    print(ans)
    solution.Insert('e')
    ans = solution.FirstAppearingOnce()
    print(ans)
```

### 55.链表中环的入口节点

**题目：**给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。

**思路：**把链表中节点值放到dict数组中，并记录出现的次数，如果出现次数超过一次，则为环的入口节点。

```python
# -*- coding:utf-8 -*-
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    def EntryNodeOfLoop(self, pHead):
        # write code here
        if pHead is None:
            return None
        num,dict,flag=[],{},True
        tempans=0
        while pHead and flag==True:
            num.append(pHead.val)
            numset=set(num)
            for c in numset:
                dict[c]=0
            for c in num:
                dict[c]=dict[c]+1
            for c in num:
                if dict[c]>1:
                    flag=False
                    tempans=c
            pHead=pHead.next
        while pHead:
            if pHead.val==tempans:
                return pHead
            pHead=pHead.next
        return None

if __name__=='__main__':
    pHead1 = ListNode(1)
    pHead2 = ListNode(2)
    pHead3 = ListNode(3)
    pHead4 = ListNode(4)
    pHead5 = ListNode(5)

    pHead1.next=pHead2
    pHead2.next=pHead3
    pHead3.next=pHead4
    pHead4.next=pHead5
    pHead5.next=pHead1

    solution=Solution()
    ans=solution.EntryNodeOfLoop(pHead1)
    print(ans.val)
```

### 56.删除链表中重复的节点

**题目：**在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5。

**思路：**记录链表中出现的数字，然后构建新链表。

```python
# -*- coding:utf-8 -*-
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        num=[]
        tempnum1=pHead
        while tempnum1:
            num.append(tempnum1.val)
            tempnum1=tempnum1.next
        dict={}
        for c in num:
            dict[c]=0
        for c in num:
            dict[c]=dict[c]+1
        newnum=[]
        for c in num:
            if dict[c]==1:
                newnum.append(c)
        if newnum==[]:
            return None
        head=ListNode(newnum[0])
        temphead=head
        for i in range(1,len(newnum)):
            tempnode=ListNode(newnum[i])
            temphead.next=tempnode
            temphead=tempnode
        # while head:
        #     print(head.val)
        #     head=head.next
        return head

if __name__=='__main__':
    pHead1 = ListNode(1)
    pHead2 = ListNode(1)
    pHead3 = ListNode(1)
    pHead4 = ListNode(1)
    pHead5 = ListNode(1)
    pHead6 = ListNode(1)
    pHead7 = ListNode(1)

    pHead1.next=pHead2
    pHead2.next=pHead3
    pHead3.next=pHead4
    pHead4.next=pHead5
    pHead5.next=pHead6
    pHead6.next=pHead7

    solution=Solution()
    ans=solution.deleteDuplication(pHead1)
    print(ans)
```

### 57. 二叉树中的下一个节点

**题目：**给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

**思路：**分析二叉树的下一个节点，一共有以下情况：1.二叉树为空，则返回空；2.节点右孩子存在，则设置一个指针从该节点的右孩子出发，一直沿着指向左子结点的指针找到的叶子节点即为下一个节点；3.节点不是根节点。如果该节点是其父节点的左孩子，则返回父节点；否则继续向上遍历其父节点的父节点，重复之前的判断，返回结果。

```python
# -*- coding:utf-8 -*-
class TreeLinkNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.next = None
class Solution:
    def GetNext(self, pNode):
        # write code here
        if not pNode:
            return pNode
        if pNode.right:
            left1=pNode.right
            while left1.left:
                   left1=left1.left
            return left1

        while pNode.next:
            tmp=pNode.next
            if tmp.left==pNode:
                return tmp
            pNode=tmp

if __name__=='__main__':
    solution=Solution()
```

### 58.对称的二叉树

**题目：**请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

**思路：**采用递归的方法来判断两数是否相同。

```python
# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def isSymmetrical(self, pRoot):
        # write code here
        if not pRoot:
            return True
        result=self.same(pRoot,pRoot)
        return result
    def same(self,root1,root2):
        if not root1 and not root2:
            return True
        if root1 and not root2:
            return False
        if not root1 and root2:
            return False
        if root1.val!= root2.val:
            return False

        left=self.same(root1.left,root2.right)
        if not left:
            return False
        right=self.same(root1.right,root2.left)
        if not right:
            return False
        return True

if __name__=='__main__':

    A1 = TreeNode(1)
    A2 = TreeNode(2)
    A3 = TreeNode(2)
    A4 = TreeNode(3)
    A5 = TreeNode(4)
    A6 = TreeNode(4)
    A7 = TreeNode(3)

    A1.left=A2
    A1.right=A3
    A2.left=A4
    A2.right=A5
    A3.left=A6
    A3.right=A7


    solution = Solution()
    ans=solution.isSymmetrical(A1)
    print(ans)
```

### 59.按之字形顺序打印二叉树

**题目：**请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

**思路：** 把当前列结果存放到list之中，设置翻转变量，依次从左到右打印和从右到左打印。

```python
# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def Print(self, pRoot):
        # write code here
        root=pRoot
        if not root:
            return []
        level=[root]
        result=[]
        righttoleft=False
        while level:
            curvalues=[]
            nextlevel=[]
            for i in level:
                curvalues.append(i.val)
                if i.left:
                    nextlevel.append(i.left)
                if i.right:
                    nextlevel.append(i.right)
            if righttoleft:
                    curvalues.reverse()
            if curvalues:
                    result.append(curvalues)
            level = nextlevel
            righttoleft = not righttoleft
        return result

if __name__=='__main__':
    A1 = TreeNode(1)
    A2 = TreeNode(2)
    A3 = TreeNode(3)
    A4 = TreeNode(4)
    A5 = TreeNode(5)
    A6 = TreeNode(6)
    A7 = TreeNode(7)

    A1.left=A2
    A1.right=A3
    A2.left=A4
    A2.right=A5
    A3.left=A6
    A3.right=A7

    solution = Solution()
    ans=solution.Print(A1)
    print(ans)
```

### 60.把二叉树打印成多行

**题目：**从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。

```python
# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, pRoot):
        # write code here
        root=pRoot
        if not root:
            return []
        level=[root]
        result=[]
        while level:
            curvalues=[]
            nextlevel=[]
            for i in level:
                curvalues.append(i.val)
                if i.left:
                    nextlevel.append(i.left)
                if i.right:
                    nextlevel.append(i.right)
            if curvalues:
                    result.append(curvalues)
            level = nextlevel
        return result

if __name__=='__main__':
    A1 = TreeNode(1)
    A2 = TreeNode(2)
    A3 = TreeNode(3)
    A4 = TreeNode(4)
    A5 = TreeNode(5)
    A6 = TreeNode(6)
    A7 = TreeNode(7)

    A1.left=A2
    A1.right=A3
    A2.left=A4
    A2.right=A5
    A3.left=A6
    A3.right=A7

    solution = Solution()
    ans=solution.Print(A1)
    print(ans)
```

### 61.序列化二叉树

**题目：**请实现两个函数，分别用来序列化和反序列化二叉树。

**思路：**转变成前序遍历，空元素利用"#"代替，然后进行解序列。

```python
# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

import collections
class Solution:
    def Serialize(self, root):
        # write code here
        if not root:
            return None
        res=[]
        self.pre(root,res)
        return res

    def pre(self,root,res):
        if not root:
            return
        res.append(root.val)
        if root.left:
            self.pre(root.left, res)
        else:
            res.append('#')
        if root.right:
            self.pre(root.right,res)
        else:
            res.append('#')
    def Deserialize(self, s):
        if s=='':
            return None
        vals=[]
        for i in range(0,len(s)):
            vals.append(s[i])
        vals=collections.deque(vals)
        ans=self.build(vals)
        return ans

    def build(self,vals):
        if vals:
            val = vals.popleft()
            if val == '#':
                return None
            root = TreeNode(int(val))
            root.left = self.build(vals)
            root.right = self.build(vals)
            return root
        return self.build(vals)

# [1, ',', 2, ',', 4, ',', ',', ',', 5, ',', ',', ',', 3, ',', 6, ',', ',', ',', 7, ',', ',']
if __name__=="__main__":
    A1 = TreeNode(1)
    A2 = TreeNode(2)
    A3 = TreeNode(3)
    A4 = TreeNode(4)
    A5 = TreeNode(5)
    A6 = TreeNode(6)
    A7 = TreeNode(7)

    A1.left=A2
    A1.right=A3
    A2.left=A4
    A2.right=A5
    A3.left=A6
    A3.right=A7

    solution = Solution()
    ans=solution.Serialize(A1)
    print(ans)
    root=solution.Deserialize(ans)
    res=solution.Serialize(root)
    print(res)
```

### 62.二叉搜索树中的第K个节点

**题目：**给定一棵二叉搜索树，请找出其中的第k小的结点。例如（5，3，7，2，4，6，8）中，按结点数值大小顺序第三小结点的值为4。

**思路：**中序遍历后，返回第K个节点值。

```python
# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    # 返回对应节点TreeNode
    def KthNode(self, pRoot, k):
        # write code here
        res=[]
        if not pRoot:
            return None
        self.order(pRoot,res)
        if len(res)<k or k<=0:
            return None
        else:
            return res[k-1]

    def order(self,root,res):
        if not root:
            return
        self.order(root.left,res)
        res.append(root)
        self.order(root.right,res)

if __name__=='__main__':
    A1 = TreeNode(5)
    A2 = TreeNode(3)
    A3 = TreeNode(7)
    A4 = TreeNode(2)
    A5 = TreeNode(4)
    A6 = TreeNode(6)
    A7 = TreeNode(8)

    A1.left=A2
    A1.right=A3
    A2.left=A4
    A2.right=A5
    A3.left=A6
    A3.right=A7

    k=3
    solution = Solution()
    ans=solution.KthNode(A1,k)
    print(ans)
```

### 63.数据流中的中位数

**题目：**如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。

```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.data=[]
    def Insert(self, num):
        # write code here
        self.data.append(num)
        self.data.sort()
    def GetMedian(self):
        # write code here
        length=len(self.data)
        if length%2==0:
            return (self.data[length//2]+self.data[length//2-1])/2.0
        else:
            return self.data[int(length//2)]


if __name__=="__main__":
    solution=Solution()
    solution.Insert(5)
    ans = solution.GetMedian()
    print(ans)
    solution.Insert(2)
    ans = solution.GetMedian()
    print(ans)
    solution.Insert(3)
    ans = solution.GetMedian()
    print(ans)
    solution.Insert(4)
    ans = solution.GetMedian()
    print(ans)
    solution.Insert(1)
    ans = solution.GetMedian()
    print(ans)
```

### 64.滑动窗口的最大值

**题目：**给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}，{2,3,4,2,6,[2,5,1]}。

```python
# -*- coding:utf-8 -*-
class Solution:
    def maxInWindows(self, num, size):
        # write code here
        if size==0 or num==[]:
            return []
        res=[]
        for i in range(0,len(num)-size+1):
            tempnum=[]
            for j in range(i,i+size):
                tempnum.append(num[j])
            res.append(max(tempnum))
        return res

if __name__=="__main__":
    solution=Solution()
    num=[2,3,4,2,6,2,5,1]
    size=3
    ans=solution.maxInWindows(num,size)
    print(ans)
```

### 66.矩阵中的路径

**题目：**请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则之后不能再次进入这个格子。例如 a b c e s f c s a d e e 这样的3 X 4 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。

**思路：**当起点第一个字符相同时，开始进行递归搜索，设计搜索函数。

```python
# -*- coding:utf-8 -*-
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        # write code here
        for i in range(0,rows):
            for j in range(0,cols):
                if matrix[i*rows+j]==path[0]:
                    if self.find_path(list(matrix),rows,cols,path[1:],i,j):
                        return True
        return False

    def find_path(self,matrix,rows,cols,path,i,j):
        if not path:
            return True
        matrix[i*cols+j]=0
        if j+1<cols and matrix[i*cols+j+1]==path[0]:
            return self.find_path(matrix,rows,cols,path[1:],i,j+1)
        elif j-1>=0 and matrix[i*cols+j-1]==path[0]:
            return self.find_path(matrix, rows, cols, path[1:], i, j - 1)
        elif i+1<rows and matrix[(i+1)*cols+j]==path[0]:
            return self.find_path(matrix, rows, cols, path[1:], i+1, j)
        elif i-1>=0 and matrix[(i-1)*cols+j]==path[0]:
            return self.find_path(matrix, rows, cols, path[1:], i-1, j)
        else:
            return False

if __name__=='__main__':
    solution=Solution()
    matrix='ABCEHJIGSFCSLOPQADEEMNOEADIDEJFMVCEIFGGS'
    rows=5
    cols=8
    path='SGGFIECVAASABCEHJIGQEMS'
    ans=solution.hasPath(matrix,rows,cols,path)
    print(ans)
```

### 66.机器人的运动范围

**题目：**地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？

**思路：**对未走过的路径进行遍历，搜索所有的路径值。

```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.vis = {}

    def movingCount(self, threshold, rows, cols):
        # write code here
        return self.moving(threshold, rows, cols, 0, 0)

    def moving(self, threshold, rows, cols, row, col):
        rowans,colans=0,0
        rowtemp,coltemp=row,col
        while rowtemp>0:
            rowans=rowans+rowtemp%10
            rowtemp=rowtemp//10
        while coltemp>0:
            colans=colans+coltemp%10
            coltemp=coltemp//10

        if rowans+colans>threshold:
            return 0
        if row >= rows or col >= cols or row < 0 or col < 0:
            return 0
        if (row, col) in self.vis:
            return 0
        self.vis[(row, col)] = 1

        return 1 + self.moving(threshold, rows, cols, row - 1, col) +\
               self.moving(threshold, rows, cols, row + 1,col) + \
               self.moving(threshold, rows,cols, row,col - 1) + \
               self.moving(threshold, rows, cols, row, col + 1)


if __name__=='__main__':
    solution=Solution()
    threshold=10
    rows,cols=1,100
    ans=solution.movingCount(threshold,rows,cols)
    print(ans)
```
### 67.推广

更多内容请关注公众号**谓之小一**，若有疑问可在公众号后台提问，随时回答，欢迎关注，内容转载请注明出处。

![推广](《剑指Offer》Python版/推广.png)