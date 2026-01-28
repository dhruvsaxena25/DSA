# Linked List - Complete Interview Guide

## 📌 Table of Contents
1. [Core Concepts](#core-concepts)
2. [Types of Linked Lists](#types-of-linked-lists)
3. [Basic Operations](#basic-operations)
4. [Time & Space Complexity](#time--space-complexity)
5. [Common Interview Patterns](#common-interview-patterns)
6. [Top 20 Interview Questions](#top-20-interview-questions)
7. [Advanced Techniques](#advanced-techniques)
8. [Interview Tips & Tricks](#interview-tips--tricks)

---

## Core Concepts

### What is a Linked List?
- **Definition**: Linear data structure where elements (nodes) are connected via pointers/references
- **Memory**: Non-contiguous memory allocation (unlike arrays)
- **Access**: Sequential access only, no random access

### Node Structure
```python
class Node:
    def __init__(self, data):
        self.data = data      # Value stored
        self.next = None      # Pointer to next node
```

### Key Characteristics
✅ **Advantages**
- Dynamic size (grows/shrinks at runtime)
- Efficient insertions/deletions at beginning: O(1)
- No memory waste from pre-allocation
- Easy to implement stacks, queues, graphs

❌ **Disadvantages**
- No random access (must traverse from head)
- Extra memory for pointers
- Not cache-friendly (scattered memory)
- Cannot use binary search directly

---

## Types of Linked Lists

### 1. Singly Linked List
```
[Data|Next] -> [Data|Next] -> [Data|Next] -> None
```
- One pointer per node (forward only)
- Most common type

```python
class SinglyLinkedList:
    def __init__(self):
        self.head = None
```

### 2. Doubly Linked List
```
None <- [Prev|Data|Next] <-> [Prev|Data|Next] <-> [Prev|Data|Next] -> None
```
- Two pointers per node (bidirectional)
- Can traverse backward

```python
class DoublyNode:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None
```

### 3. Circular Linked List
```
[Data|Next] -> [Data|Next] -> [Data|Next] --+
   ^                                         |
   +-----------------------------------------+
```
- Last node points to first node
- No None/NULL pointer at end
- Used in round-robin scheduling

```python
# Last node: tail.next = head (not None)
```

---

## Basic Operations

### 1. Insertion Operations

#### Insert at Beginning - O(1)
```python
def insert_at_beginning(self, data):
    new_node = Node(data)
    new_node.next = self.head
    self.head = new_node
```
**Interview Tip**: Always the fastest insertion!

#### Insert at End - O(n)
```python
def insert_at_end(self, data):
    new_node = Node(data)
    
    if not self.head:
        self.head = new_node
        return
    
    current = self.head
    while current.next:
        current = current.next
    current.next = new_node
```
**Optimization**: Maintain a `tail` pointer → O(1)

#### Insert at Position - O(n)
```python
def insert_at_position(self, data, position):
    if position == 0:
        self.insert_at_beginning(data)
        return
    
    new_node = Node(data)
    current = self.head
    
    for _ in range(position - 1):
        if not current:
            raise IndexError("Position out of bounds")
        current = current.next
    
    new_node.next = current.next
    current.next = new_node
```

### 2. Deletion Operations

#### Delete at Beginning - O(1)
```python
def delete_at_beginning(self):
    if not self.head:
        return None
    
    deleted_data = self.head.data
    self.head = self.head.next
    return deleted_data
```

#### Delete at End - O(n)
```python
def delete_at_end(self):
    if not self.head:
        return None
    
    if not self.head.next:  # Single node
        deleted_data = self.head.data
        self.head = None
        return deleted_data
    
    current = self.head
    while current.next.next:
        current = current.next
    
    deleted_data = current.next.data
    current.next = None
    return deleted_data
```

#### Delete by Value - O(n)
```python
def delete_by_value(self, value):
    if not self.head:
        return False
    
    # If head node has the value
    if self.head.data == value:
        self.head = self.head.next
        return True
    
    current = self.head
    while current.next:
        if current.next.data == value:
            current.next = current.next.next
            return True
        current = current.next
    
    return False
```

### 3. Traversal & Search

#### Display/Print - O(n)
```python
def display(self):
    elements = []
    current = self.head
    
    while current:
        elements.append(str(current.data))
        current = current.next
    
    print(" -> ".join(elements) + " -> None")
```

#### Search - O(n)
```python
def search(self, value):
    current = self.head
    position = 0
    
    while current:
        if current.data == value:
            return position
        current = current.next
        position += 1
    
    return -1  # Not found
```

### 4. Reverse a Linked List - O(n)
```python
def reverse(self):
    prev = None
    current = self.head
    
    while current:
        next_node = current.next  # Save next
        current.next = prev       # Reverse pointer
        prev = current            # Move prev forward
        current = next_node       # Move current forward
    
    self.head = prev
```
**Interview Gold**: One of the most asked questions!

---

## Time & Space Complexity

| Operation | Array | Singly LL | Doubly LL |
|-----------|-------|-----------|-----------|
| Access by index | O(1) | O(n) | O(n) |
| Search | O(n) | O(n) | O(n) |
| Insert at beginning | O(n) | **O(1)** | **O(1)** |
| Insert at end | O(1)* | O(n) or O(1)** | O(1)*** |
| Insert at middle | O(n) | O(n) | O(n) |
| Delete at beginning | O(n) | **O(1)** | **O(1)** |
| Delete at end | O(1)* | O(n) or O(1)** | O(1)*** |
| Delete at middle | O(n) | O(n) | O(n) |

*Amortized for dynamic arrays  
**O(1) with tail pointer  
***With tail pointer

**Space Complexity**
- Singly LL: O(n) - one pointer per node
- Doubly LL: O(n) - two pointers per node
- Circular LL: O(n) - same as base type

---

## Common Interview Patterns

### 🎯 Pattern 1: Two Pointer Technique

#### Fast & Slow Pointers (Floyd's Cycle Detection)
```python
def has_cycle(head):
    """Detect if linked list has a cycle - O(n) time, O(1) space"""
    if not head:
        return False
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True
    
    return False
```

#### Find Middle Node
```python
def find_middle(head):
    """Find middle using slow/fast pointers"""
    if not head:
        return None
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow  # Slow is at middle when fast reaches end
```

#### Find Nth Node from End
```python
def find_nth_from_end(head, n):
    """Two pointers with n gap between them"""
    first = second = head
    
    # Move first pointer n steps ahead
    for _ in range(n):
        if not first:
            return None
        first = first.next
    
    # Move both until first reaches end
    while first:
        first = first.next
        second = second.next
    
    return second
```

### 🎯 Pattern 2: Dummy Node Technique

```python
def remove_elements(head, val):
    """Remove all nodes with given value using dummy node"""
    dummy = Node(0)
    dummy.next = head
    current = dummy
    
    while current.next:
        if current.next.data == val:
            current.next = current.next.next
        else:
            current = current.next
    
    return dummy.next
```

**Why use dummy?** Handles edge cases (head deletion) elegantly!

### 🎯 Pattern 3: Recursion

#### Reverse Linked List (Recursive)
```python
def reverse_recursive(head):
    """Reverse using recursion - O(n) time, O(n) stack space"""
    if not head or not head.next:
        return head
    
    new_head = reverse_recursive(head.next)
    head.next.next = head
    head.next = None
    
    return new_head
```

#### Print Reverse (Without modifying)
```python
def print_reverse(head):
    """Print in reverse without changing structure"""
    if not head:
        return
    
    print_reverse(head.next)
    print(head.data, end=" ")
```

---

## Top 20 Interview Questions

### ⭐ Easy Level

#### 1. Reverse a Linked List
```python
def reverse_list(head):
    prev = None
    current = head
    
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    return prev
```
**Companies**: Amazon, Microsoft, Google, Facebook  
**Difficulty**: Easy  
**Pattern**: Iterative traversal

#### 2. Detect Cycle in Linked List
```python
def has_cycle(head):
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    
    return False
```
**Companies**: Amazon, Microsoft, Adobe  
**Difficulty**: Easy  
**Pattern**: Floyd's algorithm (tortoise & hare)

#### 3. Merge Two Sorted Lists
```python
def merge_two_lists(l1, l2):
    dummy = Node(0)
    current = dummy
    
    while l1 and l2:
        if l1.data <= l2.data:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    current.next = l1 or l2
    return dummy.next
```
**Companies**: Facebook, Amazon, Microsoft  
**Difficulty**: Easy  
**Pattern**: Two pointers + dummy node

#### 4. Remove Duplicates from Sorted List
```python
def delete_duplicates(head):
    current = head
    
    while current and current.next:
        if current.data == current.next.data:
            current.next = current.next.next
        else:
            current = current.next
    
    return head
```
**Companies**: Adobe, Amazon  
**Difficulty**: Easy

#### 5. Find Middle of Linked List
```python
def find_middle(head):
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow
```
**Companies**: Microsoft, Amazon  
**Difficulty**: Easy  
**Pattern**: Slow/fast pointers

### ⭐⭐ Medium Level

#### 6. Remove Nth Node From End
```python
def remove_nth_from_end(head, n):
    dummy = Node(0)
    dummy.next = head
    first = second = dummy
    
    # Move first n+1 steps ahead
    for _ in range(n + 1):
        first = first.next
    
    # Move both until first reaches end
    while first:
        first = first.next
        second = second.next
    
    # Remove nth node
    second.next = second.next.next
    
    return dummy.next
```
**Companies**: Amazon, Facebook, Google  
**Difficulty**: Medium  
**Pattern**: Two pointers with gap

#### 7. Add Two Numbers (as Linked Lists)
```python
def add_two_numbers(l1, l2):
    """
    Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
    Output: 7 -> 0 -> 8
    Explanation: 342 + 465 = 807
    """
    dummy = Node(0)
    current = dummy
    carry = 0
    
    while l1 or l2 or carry:
        val1 = l1.data if l1 else 0
        val2 = l2.data if l2 else 0
        
        total = val1 + val2 + carry
        carry = total // 10
        current.next = Node(total % 10)
        
        current = current.next
        if l1: l1 = l1.next
        if l2: l2 = l2.next
    
    return dummy.next
```
**Companies**: Amazon, Microsoft, Adobe  
**Difficulty**: Medium

#### 8. Reorder List (L0→L1→...→Ln-1→Ln to L0→Ln→L1→Ln-1→...)
```python
def reorder_list(head):
    if not head or not head.next:
        return
    
    # Find middle
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse second half
    second = slow.next
    slow.next = None
    second = reverse_list(second)
    
    # Merge alternately
    first = head
    while second:
        temp1, temp2 = first.next, second.next
        first.next = second
        second.next = temp1
        first, second = temp1, temp2
```
**Companies**: Facebook, Amazon  
**Difficulty**: Medium  
**Pattern**: Find middle + Reverse + Merge

#### 9. Copy List with Random Pointer
```python
class NodeWithRandom:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.random = None

def copy_random_list(head):
    if not head:
        return None

    # Map original node -> copied node
    old_to_new = {}

    # Pass 1: create all copied nodes (no pointers yet)
    curr = head
    while curr:
        old_to_new[curr] = NodeWithRandom(curr.data)
        curr = curr.next

    # Pass 2: wire next/random using the map
    curr = head
    while curr:
        copy = old_to_new[curr]
        copy.next = old_to_new.get(curr.next)      # None if curr.next is None
        copy.random = old_to_new.get(curr.random)  # None if curr.random is None
        curr = curr.next

    return old_to_new[head]

```
**Companies**: Amazon, Facebook, Microsoft  
**Difficulty**: Medium  
**Pattern**: Three-pass algorithm

#### 10. Intersection of Two Linked Lists
```python
def get_intersection_node(headA, headB):
    if not headA or not headB:
        return None
    
    pA, pB = headA, headB
    
    # When pointer reaches end, redirect to other list's head
    while pA != pB:
        pA = pA.next if pA else headB
        pB = pB.next if pB else headA
    
    return pA  # Either intersection node or None
```
**Companies**: Amazon, Microsoft, Bloomberg  
**Difficulty**: Medium  
**Pattern**: Two pointers with list switching

#### 11. Palindrome Linked List
```python
def is_palindrome(head):
    # Find middle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse second half
    second = reverse_list(slow)
    
    # Compare
    first = head
    while second:
        if first.data != second.data:
            return False
        first = first.next
        second = second.next
    
    return True
```
**Companies**: Amazon, Facebook, Microsoft  
**Difficulty**: Medium

#### 12. Flatten a Multilevel Doubly Linked List
```python
def flatten(head):
    if not head:
        return None
    
    current = head
    
    while current:
        if current.child:
            # Find tail of child list
            child_tail = current.child
            while child_tail.next:
                child_tail = child_tail.next
            
            # Connect child to main list
            child_tail.next = current.next
            if current.next:
                current.next.prev = child_tail
            
            current.next = current.child
            current.child.prev = current
            current.child = None
        
        current = current.next
    
    return head
```
**Companies**: Amazon, Microsoft  
**Difficulty**: Medium

### ⭐⭐⭐ Hard Level

#### 13. Merge K Sorted Lists
```python
import heapq

def merge_k_lists(lists):
    """Using Min Heap - O(N log k) where N = total nodes, k = number of lists"""
    heap = []
    
    # Add first node of each list to heap
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.data, i, node))
    
    dummy = Node(0)
    current = dummy
    
    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        
        if node.next:
            heapq.heappush(heap, (node.next.data, i, node.next))
    
    return dummy.next
```
**Companies**: Google, Amazon, Facebook  
**Difficulty**: Hard  
**Pattern**: Min heap

#### 14. Reverse Nodes in K-Group
```python
def reverse_k_group(head, k):
    """
    Input: 1->2->3->4->5, k = 2
    Output: 2->1->4->3->5
    """
    # Check if there are k nodes remaining
    current = head
    count = 0
    while current and count < k:
        current = current.next
        count += 1
    
    if count < k:
        return head
    
    # Reverse first k nodes
    prev = None
    current = head
    for _ in range(k):
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    # Recursively reverse remaining
    head.next = reverse_k_group(current, k)
    
    return prev
```
**Companies**: Google, Facebook, Amazon  
**Difficulty**: Hard

#### 15. LRU Cache Implementation
```python
class LRUNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key -> node
        self.head = LRUNode()  # Dummy head
        self.tail = LRUNode()  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _remove(self, node):
        """Remove node from doubly linked list"""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _add_to_head(self, node):
        """Add node right after head"""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
    
    def get(self, key):
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add_to_head(node)
            return node.value
        return -1
    
    def put(self, key, value):
        if key in self.cache:
            self._remove(self.cache[key])
        
        node = LRUNode(key, value)
        self._add_to_head(node)
        self.cache[key] = node
        
        if len(self.cache) > self.capacity:
            # Remove least recently used (before tail)
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]
```
**Companies**: Amazon, Google, Microsoft, Facebook  
**Difficulty**: Hard  
**Pattern**: Hash map + Doubly linked list

#### 16. Sort Linked List (O(n log n))
```python
def sort_list(head):
    """Merge sort for linked list"""
    if not head or not head.next:
        return head
    
    # Find middle using slow/fast
    slow = fast = head
    prev = None
    
    while fast and fast.next:
        prev = slow
        slow = slow.next
        fast = fast.next.next
    
    # Split into two halves
    prev.next = None
    
    # Recursively sort both halves
    left = sort_list(head)
    right = sort_list(slow)
    
    # Merge sorted halves
    return merge_two_lists(left, right)
```
**Companies**: Microsoft, Amazon  
**Difficulty**: Hard  
**Pattern**: Merge sort

#### 17. Rotate List
```python
def rotate_right(head, k):
    """
    Input: 1->2->3->4->5, k = 2
    Output: 4->5->1->2->3
    """
    if not head or k == 0:
        return head
    
    # Find length and make circular
    length = 1
    tail = head
    while tail.next:
        tail = tail.next
        length += 1
    
    tail.next = head  # Make circular
    
    # Find new tail (length - k % length - 1)
    k = k % length
    steps_to_new_tail = length - k
    
    new_tail = head
    for _ in range(steps_to_new_tail - 1):
        new_tail = new_tail.next
    
    new_head = new_tail.next
    new_tail.next = None
    
    return new_head
```
**Companies**: Amazon, Microsoft  
**Difficulty**: Medium-Hard

#### 18. Clone Graph (uses linked list concepts)
```python
class GraphNode:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors else []

def clone_graph(node):
    if not node:
        return None
    
    visited = {}
    
    def dfs(node):
        if node in visited:
            return visited[node]
        
        clone = GraphNode(node.val)
        visited[node] = clone
        
        for neighbor in node.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)
```
**Companies**: Facebook, Google, Amazon  
**Difficulty**: Medium

#### 19. Swap Nodes in Pairs
```python
def swap_pairs(head):
    """
    Input: 1->2->3->4
    Output: 2->1->4->3
    """
    dummy = Node(0)
    dummy.next = head
    prev = dummy
    
    while prev.next and prev.next.next:
        first = prev.next
        second = prev.next.next
        
        # Swap
        prev.next = second
        first.next = second.next
        second.next = first
        
        prev = first
    
    return dummy.next
```
**Companies**: Microsoft, Amazon  
**Difficulty**: Medium

#### 20. Find Duplicate Number (Cycle Detection variant)
```python
def find_duplicate(nums):
    """
    Array with n+1 integers where each integer is between 1 and n
    Uses Floyd's algorithm treating array as linked list
    """
    # Phase 1: Find intersection
    slow = fast = nums[0]
    
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    
    # Phase 2: Find entrance to cycle
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    
    return slow
```
**Companies**: Amazon, Microsoft  
**Difficulty**: Medium  
**Pattern**: Floyd's algorithm variant

---

## Advanced Techniques

### 1. Skip List (Advanced Data Structure)
```python
import random

class SkipNode:
    def __init__(self, key, value, level):
        self.key = key
        self.value = value
        self.forward = [None] * (level + 1)

class SkipList:
    """Probabilistic data structure - O(log n) search/insert/delete"""
    def __init__(self, max_level=16, p=0.5):
        self.max_level = max_level
        self.p = p
        self.header = SkipNode(None, None, max_level)
        self.level = 0
    
    def random_level(self):
        level = 0
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level
    
    def search(self, key):
        current = self.header
        
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        
        current = current.forward[0]
        
        if current and current.key == key:
            return current.value
        return None
```

### 2. XOR Linked List (Memory Efficient)
```python
class XORNode:
    """Memory efficient doubly linked list using XOR"""
    def __init__(self, data):
        self.data = data
        self.both = 0  # XOR of prev and next addresses
    
# XOR of two addresses: prev XOR next
# To traverse: next = prev XOR current.both
```
**Note**: Not commonly used in Python due to garbage collection

### 3. Unrolled Linked List
```python
class UnrolledNode:
    """Stores multiple elements per node - cache friendly"""
    def __init__(self, capacity=16):
        self.capacity = capacity
        self.elements = []
        self.next = None
```

---

## Interview Tips & Tricks

### 🎯 Before You Code

1. **Ask Clarifying Questions**
   - Is it singly or doubly linked?
   - Can there be cycles?
   - Can I modify the original list?
   - What should I return if list is empty?
   - Are there duplicates?

2. **Analyze Edge Cases**
   ```python
   # Always test these:
   - Empty list (head = None)
   - Single node
   - Two nodes
   - Large list
   - All duplicates
   - Already sorted/reversed
   ```

3. **Draw It Out**
   ```
   Before: 1 -> 2 -> 3 -> 4 -> None
   After:  4 -> 3 -> 2 -> 1 -> None
   
   Step by step visualization helps!
   ```

### 🎯 Common Patterns to Recognize

| Problem Type | Pattern | Technique |
|-------------|---------|-----------|
| Cycle detection | Fast/slow pointers | Floyd's algorithm |
| Find middle | Fast/slow pointers | Slow moves 1, fast moves 2 |
| Nth from end | Two pointers | Gap of n between them |
| Merge lists | Dummy node | Avoid head edge cases |
| Reverse | Three pointers | prev, current, next |
| Palindrome | Find middle + reverse | Two phases |
| Intersection | Two pointers | Switch lists when reaching end |

### 🎯 Code Templates

#### Template 1: Dummy Node Pattern
```python
def solution(head):
    dummy = Node(0)
    dummy.next = head
    current = dummy
    
    # Process logic here
    while current.next:
        # ... operations
        current = current.next
    
    return dummy.next  # Handles head changes elegantly
```

#### Template 2: Two Pointer Pattern
```python
def solution(head):
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        # Check condition (cycle, middle, etc.)
        if slow == fast:
            return True
    
    return False
```

#### Template 3: Recursive Pattern
```python
def solution(head):
    # Base case
    if not head or not head.next:
        return head
    
    # Recursive case
    result = solution(head.next)
    
    # Process current node
    # ... logic
    
    return result
```

### 🎯 Time/Space Optimization Tricks

1. **Avoid Creating New List**
   - Modify pointers in place when possible
   - Space: O(n) → O(1)

2. **Use Dummy Node**
   - Eliminates special cases for head
   - Cleaner code, fewer bugs

3. **Fast/Slow Pointers**
   - Find middle: O(n) time, O(1) space
   - Alternative: Count length first (2 passes)

4. **Tail Pointer Optimization**
   ```python
   class LinkedList:
       def __init__(self):
           self.head = None
           self.tail = None  # Maintain tail
       
       def append(self, data):
           new_node = Node(data)
           if self.tail:
               self.tail.next = new_node
           else:
               self.head = new_node
           self.tail = new_node  # O(1) append!
   ```

### 🎯 Common Mistakes to Avoid

❌ **Mistake 1**: Losing reference to next node
```python
# WRONG
current.next = prev  # Lost reference to current.next!

# CORRECT
next_node = current.next  # Save first
current.next = prev
current = next_node
```

❌ **Mistake 2**: Not handling empty list
```python
# WRONG
def reverse(head):
    prev = None
    current = head
    # Crashes if head is None!

# CORRECT
def reverse(head):
    if not head:  # Check first
        return None
    prev = None
    current = head
```

❌ **Mistake 3**: Off-by-one errors
```python
# Finding nth from end
# WRONG: range(n) - off by one
# CORRECT: range(n + 1) for dummy node approach
```

❌ **Mistake 4**: Not checking fast.next before fast.next.next
```python
# WRONG
while fast:
    fast = fast.next.next  # Can crash!

# CORRECT
while fast and fast.next:
    fast = fast.next.next
```

### 🎯 Interview Communication Tips

1. **Think Aloud**
   - "I'll use two pointers here because..."
   - "Edge case: what if the list is empty?"
   - "Time complexity is O(n) because we traverse once"

2. **Start Simple**
   - Write iterative before recursive
   - Brute force first, then optimize
   - "Let me first solve this in O(n²), then optimize to O(n)"

3. **Test Your Code**
   ```python
   # Test cases to mention:
   print(reverse(None))           # Empty
   print(reverse(Node(1)))        # Single
   print(reverse(1->2))           # Two nodes
   print(reverse(1->2->3->4->5))  # Multiple
   ```

4. **Mention Alternatives**
   - "We could also use recursion, but it would use O(n) stack space"
   - "A hash map would solve this in O(n) space vs O(1) for two pointers"

### 🎯 Big O Cheat Sheet for Linked Lists

| Operation | Best | Average | Worst | Space |
|-----------|------|---------|-------|-------|
| Access | O(n) | O(n) | O(n) | O(1) |
| Search | O(1) | O(n) | O(n) | O(1) |
| Insert (beginning) | O(1) | O(1) | O(1) | O(1) |
| Insert (end) | O(1)* | O(n) | O(n) | O(1) |
| Delete (beginning) | O(1) | O(1) | O(1) | O(1) |
| Delete (end) | O(1)* | O(n) | O(n) | O(1) |

*With tail pointer

---

## Quick Reference: Problem → Solution Strategy

| Problem | First Thought | Technique |
|---------|--------------|-----------|
| Reverse | Three pointers | prev, curr, next |
| Cycle? | Two pointers | Floyd's algorithm |
| Middle? | Two pointers | Slow/fast |
| Merge sorted | Dummy + two pointers | Compare & link |
| Nth from end | Two pointers | n gap between |
| Palindrome | Middle + reverse | Two-phase |
| Remove duplicates | One pointer | Compare current/next |
| Clone with random | Three passes | Interleave → assign → separate |
| LRU Cache | Hash + DLL | O(1) get/put |
| Flatten multilevel | Stack or recursion | DFS traversal |


---

**Good luck! 🚀**

*Remember: Linked lists are fundamental. Master them, and you'll find many other concepts (stacks, queues, graphs) become easier!*
