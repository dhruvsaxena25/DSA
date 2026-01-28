# Queue Data Structure - Complete Interview Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Types of Queues](#types-of-queues)
4. [Implementation Methods](#implementation-methods)
5. [Common Operations & Time Complexity](#common-operations--time-complexity)
6. [Interview Problem Patterns](#interview-problem-patterns)
7. [Common Interview Questions](#common-interview-questions)
8. [Pitfalls & Edge Cases](#pitfalls--edge-cases)
9. [Quick Reference](#quick-reference)

---

## Introduction

### What is a Queue?
A **queue** is a linear data structure that follows the **FIFO (First-In-First-Out)** principle. The first element added is the first one to be removed - like a line at a coffee shop.

### Real-World Analogies
- **Checkout line** at a grocery store
- **Printer queue** - documents print in the order they were sent
- **Call center** - customers helped in order of arrival
- **Task scheduling** in operating systems
- **BFS traversal** in graphs and trees

### Why Queues Matter in Interviews
- Fundamental for BFS algorithms
- Essential for level-order tree traversal
- Used in system design (message queues, task schedulers)
- Tests understanding of FIFO ordering
- Common in streaming/window problems

---

## Core Concepts

### FIFO Principle
```
Enqueue (Add) →  [5, 3, 7, 2, 9]  → Dequeue (Remove)
                  ↑           ↑
                Front        Rear
```

### Key Terminology
- **Enqueue**: Add element to the rear/back
- **Dequeue**: Remove element from the front
- **Front/Head**: First element to be removed
- **Rear/Tail**: Last element, most recently added
- **Peek/Front**: View front element without removing
- **IsEmpty**: Check if queue is empty
- **Size**: Number of elements in queue

### Queue vs Stack
| Feature | Queue | Stack |
|---------|-------|-------|
| Order | FIFO | LIFO |
| Insertion | Rear | Top |
| Deletion | Front | Top |
| Use Case | BFS, Scheduling | DFS, Recursion |

---

## Types of Queues

### 1. Simple Queue (Linear Queue)
Standard FIFO queue with single-ended insertion and deletion.

```python
# Using list (inefficient for production)
queue = []
queue.append(1)  # enqueue
queue.append(2)
front = queue.pop(0)  # dequeue - O(n) operation!
```

### 2. Circular Queue
Rear wraps around to the front when reaching the end of the array.

```python
class CircularQueue:
    def __init__(self, k):
        """Initialize queue with fixed size k"""
        self.queue = [None] * k
        self.size = k
        self.front = -1
        self.rear = -1
        self.count = 0
    
    def enqueue(self, value):
        """Add element to rear - O(1)"""
        if self.is_full():
            return False
        
        # First element case
        if self.front == -1:
            self.front = 0
            self.rear = 0
        else:
            # Circular increment
            self.rear = (self.rear + 1) % self.size
        
        self.queue[self.rear] = value
        self.count += 1
        return True
    
    def dequeue(self):
        """Remove element from front - O(1)"""
        if self.is_empty():
            return -1
        
        value = self.queue[self.front]
        
        # Last element case
        if self.front == self.rear:
            self.front = -1
            self.rear = -1
        else:
            # Circular increment
            self.front = (self.front + 1) % self.size
        
        self.count -= 1
        return value
    
    def peek(self):
        """View front element - O(1)"""
        if self.is_empty():
            return -1
        return self.queue[self.front]
    
    def is_empty(self):
        """Check if empty - O(1)"""
        return self.count == 0
    
    def is_full(self):
        """Check if full - O(1)"""
        return self.count == self.size

# Example usage
cq = CircularQueue(5)
cq.enqueue(1)
cq.enqueue(2)
cq.enqueue(3)
print(cq.dequeue())  # 1
print(cq.peek())     # 2
```

### 3. Double-Ended Queue (Deque)
Insertion and deletion allowed at both ends.

```python
from collections import deque

# Standard deque operations
dq = deque()

# Add to rear
dq.append(1)
dq.append(2)

# Add to front
dq.appendleft(0)

# Remove from rear
dq.pop()

# Remove from front
dq.popleft()

# Access elements
front = dq[0]
rear = dq[-1]

# Example: Sliding window maximum
def max_sliding_window(nums, k):
    """Find maximum in each sliding window of size k"""
    result = []
    dq = deque()  # Store indices
    
    for i in range(len(nums)):
        # Remove elements outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements (they'll never be max)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add to result once window is full
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

# Test
print(max_sliding_window([1,3,-1,-3,5,3,6,7], 3))
# Output: [3,3,5,5,6,7]
```

### 4. Priority Queue
Elements dequeued based on priority, not insertion order.

```python
import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0  # For stable ordering
    
    def enqueue(self, item, priority):
        """Lower priority value = higher priority"""
        # Use counter for stable ordering of equal priorities
        heapq.heappush(self.heap, (priority, self.counter, item))
        self.counter += 1
    
    def dequeue(self):
        """Remove highest priority element"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return heapq.heappop(self.heap)[2]  # Return item only
    
    def peek(self):
        """View highest priority element"""
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self.heap[0][2]
    
    def is_empty(self):
        return len(self.heap) == 0
    
    def size(self):
        return len(self.heap)

# Example: Task scheduling
pq = PriorityQueue()
pq.enqueue("Low priority task", 3)
pq.enqueue("High priority task", 1)
pq.enqueue("Medium priority task", 2)

print(pq.dequeue())  # "High priority task"
print(pq.dequeue())  # "Medium priority task"
```

---

## Implementation Methods

### 1. Using Python List (❌ Not Recommended for Production)

```python
class QueueWithList:
    def __init__(self):
        self.queue = []
    
    def enqueue(self, item):
        """O(1) - amortized"""
        self.queue.append(item)
    
    def dequeue(self):
        """O(n) - THIS IS THE PROBLEM!"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self.queue.pop(0)  # Shifts all elements
    
    def is_empty(self):
        return len(self.queue) == 0

# Why this is bad:
# q = QueueWithList()
# for i in range(1000000):
#     q.enqueue(i)
# 
# # This will be VERY slow
# while not q.is_empty():
#     q.dequeue()  # Each call is O(n)
```

**Problems:**
- `pop(0)` shifts all remaining elements → O(n)
- Terrible performance for large queues
- Only use for small queues or learning

### 2. Using collections.deque (✅ RECOMMENDED)

```python
from collections import deque

class Queue:
    def __init__(self):
        self.queue = deque()
    
    def enqueue(self, item):
        """O(1)"""
        self.queue.append(item)
    
    def dequeue(self):
        """O(1)"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self.queue.popleft()
    
    def peek(self):
        """O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self.queue[0]
    
    def is_empty(self):
        """O(1)"""
        return len(self.queue) == 0
    
    def size(self):
        """O(1)"""
        return len(self.queue)

# Usage
q = Queue()
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
print(q.dequeue())  # 1
print(q.peek())     # 2
print(q.size())     # 2
```

### 3. Using Linked List

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedListQueue:
    def __init__(self):
        self.front = None
        self.rear = None
        self._size = 0
    
    def enqueue(self, item):
        """O(1) - Add to rear"""
        new_node = Node(item)
        
        # Empty queue case
        if self.rear is None:
            self.front = self.rear = new_node
        else:
            self.rear.next = new_node
            self.rear = new_node
        
        self._size += 1
    
    def dequeue(self):
        """O(1) - Remove from front"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        
        removed_data = self.front.data
        self.front = self.front.next
        
        # If queue becomes empty
        if self.front is None:
            self.rear = None
        
        self._size -= 1
        return removed_data
    
    def peek(self):
        """O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self.front.data
    
    def is_empty(self):
        """O(1)"""
        return self.front is None
    
    def size(self):
        """O(1)"""
        return self._size
    
    def display(self):
        """O(n) - For debugging"""
        if self.is_empty():
            print("Queue is empty")
            return
        
        current = self.front
        elements = []
        while current:
            elements.append(str(current.data))
            current = current.next
        print(" -> ".join(elements))

# Usage
llq = LinkedListQueue()
llq.enqueue(10)
llq.enqueue(20)
llq.enqueue(30)
llq.display()  # 10 -> 20 -> 30
print(llq.dequeue())  # 10
llq.display()  # 20 -> 30
```

### 4. Using Two Stacks (Interview Classic!)

```python
class QueueUsingStacks:
    """
    Implement queue using two stacks
    Stack 1: For enqueue operations
    Stack 2: For dequeue operations
    """
    def __init__(self):
        self.stack_in = []   # For enqueue
        self.stack_out = []  # For dequeue
    
    def enqueue(self, item):
        """O(1)"""
        self.stack_in.append(item)
    
    def dequeue(self):
        """Amortized O(1)"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        
        # If out stack is empty, transfer all from in stack
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        
        return self.stack_out.pop()
    
    def peek(self):
        """Amortized O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty queue")
        
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        
        return self.stack_out[-1]
    
    def is_empty(self):
        """O(1)"""
        return not self.stack_in and not self.stack_out
    
    def size(self):
        """O(1)"""
        return len(self.stack_in) + len(self.stack_out)

# Example walkthrough
qs = QueueUsingStacks()
qs.enqueue(1)  # stack_in: [1], stack_out: []
qs.enqueue(2)  # stack_in: [1,2], stack_out: []
qs.enqueue(3)  # stack_in: [1,2,3], stack_out: []

print(qs.dequeue())  # Transfers: stack_in: [], stack_out: [3,2,1]
                      # Returns: 1, stack_out: [3,2]

qs.enqueue(4)  # stack_in: [4], stack_out: [3,2]
print(qs.dequeue())  # Returns: 2, stack_out: [3]
```

---

## Common Operations & Time Complexity

### Time Complexity Summary

| Operation | List (pop(0)) | deque | Linked List | Two Stacks | Circular Array |
|-----------|---------------|-------|-------------|------------|----------------|
| Enqueue   | O(1)*         | O(1)  | O(1)        | O(1)       | O(1)           |
| Dequeue   | O(n) ❌       | O(1)  | O(1)        | O(1)**     | O(1)           |
| Peek      | O(1)          | O(1)  | O(1)        | O(1)**     | O(1)           |
| IsEmpty   | O(1)          | O(1)  | O(1)        | O(1)       | O(1)           |
| Size      | O(1)          | O(1)  | O(1)***     | O(1)       | O(1)           |

\* Amortized when array needs to resize  
\** Amortized due to occasional transfer  
\*** If maintaining size counter

### Space Complexity
- **deque**: O(n)
- **Linked List**: O(n) but with extra space for pointers
- **Two Stacks**: O(n)
- **Circular Array**: O(k) where k is fixed size

---

## Interview Problem Patterns

### Pattern 1: BFS/Level Order Traversal

```python
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def level_order_traversal(root):
    """
    Problem: Return level order traversal of binary tree
    Time: O(n), Space: O(w) where w is max width
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result

# More advanced: Zigzag level order
def zigzag_level_order(root):
    """Alternate between left-to-right and right-to-left"""
    if not root:
        return []
    
    result = []
    queue = deque([root])
    left_to_right = True
    
    while queue:
        level_size = len(queue)
        current_level = deque()
        
        for _ in range(level_size):
            node = queue.popleft()
            
            # Add to correct end based on direction
            if left_to_right:
                current_level.append(node.val)
            else:
                current_level.appendleft(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(list(current_level))
        left_to_right = not left_to_right
    
    return result
```

### Pattern 2: Sliding Window with Queue

```python
from collections import deque

def moving_average(stream, window_size):
    """
    Calculate moving average of last window_size elements
    Time: O(1) per element, Space: O(k)
    """
    queue = deque()
    total = 0
    
    def next_val(val):
        nonlocal total
        queue.append(val)
        total += val
        
        if len(queue) > window_size:
            removed = queue.popleft()
            total -= removed
        
        return total / len(queue)
    
    return next_val

# Example
ma = moving_average([], 3)
print(ma(1))    # 1.0
print(ma(10))   # 5.5
print(ma(3))    # 4.67
print(ma(5))    # 6.0
```

### Pattern 3: Monotonic Queue

```python
from collections import deque

def max_in_sliding_window(nums, k):
    """
    Find maximum in each sliding window
    Uses monotonic decreasing queue
    Time: O(n), Space: O(k)
    """
    result = []
    dq = deque()  # Store indices
    
    for i in range(len(nums)):
        # Remove elements outside current window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Maintain decreasing order - remove smaller elements
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Start adding to result once window is full
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

# Example
print(max_in_sliding_window([1,3,-1,-3,5,3,6,7], 3))
# Output: [3, 3, 5, 5, 6, 7]
```

### Pattern 4: Graph BFS

```python
from collections import deque

def shortest_path_binary_matrix(grid):
    """
    Find shortest path from top-left to bottom-right in binary matrix
    Can only move through 0s in 8 directions
    Time: O(n²), Space: O(n²)
    """
    n = len(grid)
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1
    
    # 8 directions: up, down, left, right, and 4 diagonals
    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    
    queue = deque([(0, 0, 1)])  # (row, col, distance)
    visited = {(0, 0)}
    
    while queue:
        row, col, dist = queue.popleft()
        
        # Reached destination
        if row == n - 1 and col == n - 1:
            return dist
        
        # Explore all 8 directions
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds and validity
            if (0 <= new_row < n and 
                0 <= new_col < n and 
                grid[new_row][new_col] == 0 and 
                (new_row, new_col) not in visited):
                
                queue.append((new_row, new_col, dist + 1))
                visited.add((new_row, new_col))
    
    return -1  # No path found
```

### Pattern 5: Task Scheduling/Processing

```python
from collections import deque

def time_to_complete_tasks(tasks, cooldown):
    """
    Calculate time to complete all tasks with cooldown period
    Same task can't run within cooldown interval
    Time: O(n), Space: O(k) where k is unique tasks
    """
    if not tasks:
        return 0
    
    task_count = {}
    for task in tasks:
        task_count[task] = task_count.get(task, 0) + 1
    
    # Max frequency determines minimum time
    max_freq = max(task_count.values())
    max_freq_count = sum(1 for count in task_count.values() if count == max_freq)
    
    # Calculate based on intervals
    intervals = max_freq - 1
    interval_length = cooldown + 1
    empty_slots = intervals * (interval_length - max_freq_count)
    remaining_tasks = len(tasks) - max_freq * max_freq_count
    idles = max(0, empty_slots - remaining_tasks)
    
    return len(tasks) + idles

# Example
print(time_to_complete_tasks(['A','A','A','B','B','B'], 2))
# Output: 8 (A -> B -> idle -> A -> B -> idle -> A -> B)
```

### Pattern 6: Recent Counter/Time-based Queue

```python
from collections import deque

class RecentCounter:
    """
    Count requests in last 3000ms
    LeetCode 933: Number of Recent Calls
    """
    def __init__(self):
        self.queue = deque()
    
    def ping(self, t):
        """
        Add new request and return count in [t-3000, t]
        Time: O(1) amortized, Space: O(w) where w is window size
        """
        self.queue.append(t)
        
        # Remove requests older than 3000ms
        while self.queue and self.queue[0] < t - 3000:
            self.queue.popleft()
        
        return len(self.queue)

# Example
rc = RecentCounter()
print(rc.ping(1))     # 1 (request at 1)
print(rc.ping(100))   # 2 (requests at 1, 100)
print(rc.ping(3001))  # 3 (requests at 1, 100, 3001)
print(rc.ping(3002))  # 3 (requests at 100, 3001, 3002; 1 is too old)
```

---

## Common Interview Questions

### Beginner Level

#### 1. Implement Queue using Stacks
```python
class MyQueue:
    """LeetCode 232"""
    def __init__(self):
        self.s1 = []  # enqueue
        self.s2 = []  # dequeue
    
    def push(self, x):
        """O(1)"""
        self.s1.append(x)
    
    def pop(self):
        """Amortized O(1)"""
        self._move()
        return self.s2.pop()
    
    def peek(self):
        """Amortized O(1)"""
        self._move()
        return self.s2[-1]
    
    def empty(self):
        """O(1)"""
        return not self.s1 and not self.s2
    
    def _move(self):
        """Transfer elements from s1 to s2 if s2 is empty"""
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
```

#### 2. Design Circular Queue
```python
class MyCircularQueue:
    """LeetCode 622"""
    def __init__(self, k):
        self.queue = [0] * k
        self.size = k
        self.front = 0
        self.rear = -1
        self.count = 0
    
    def enQueue(self, value):
        """O(1)"""
        if self.isFull():
            return False
        self.rear = (self.rear + 1) % self.size
        self.queue[self.rear] = value
        self.count += 1
        return True
    
    def deQueue(self):
        """O(1)"""
        if self.isEmpty():
            return False
        self.front = (self.front + 1) % self.size
        self.count -= 1
        return True
    
    def Front(self):
        """O(1)"""
        return -1 if self.isEmpty() else self.queue[self.front]
    
    def Rear(self):
        """O(1)"""
        return -1 if self.isEmpty() else self.queue[self.rear]
    
    def isEmpty(self):
        return self.count == 0
    
    def isFull(self):
        return self.count == self.size
```

### Intermediate Level

#### 3. Binary Tree Right Side View
```python
from collections import deque

def right_side_view(root):
    """
    LeetCode 199: Return values visible from right side
    Time: O(n), Space: O(w)
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        
        for i in range(level_size):
            node = queue.popleft()
            
            # Rightmost element of this level
            if i == level_size - 1:
                result.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return result
```

#### 4. Perfect Squares
```python
from collections import deque

def num_squares(n):
    """
    LeetCode 279: Minimum perfect squares that sum to n
    BFS approach - find shortest path
    Time: O(n√n), Space: O(n)
    """
    if n <= 0:
        return 0
    
    # Pre-compute perfect squares up to n
    squares = []
    i = 1
    while i * i <= n:
        squares.append(i * i)
        i += 1
    
    queue = deque([(n, 0)])  # (remaining, steps)
    visited = {n}
    
    while queue:
        remaining, steps = queue.popleft()
        
        # Try subtracting each perfect square
        for square in squares:
            next_remaining = remaining - square
            
            if next_remaining == 0:
                return steps + 1
            
            if next_remaining > 0 and next_remaining not in visited:
                queue.append((next_remaining, steps + 1))
                visited.add(next_remaining)
    
    return -1
```

#### 5. Rotting Oranges
```python
from collections import deque

def oranges_rotting(grid):
    """
    LeetCode 994: Multi-source BFS
    Time: O(m*n), Space: O(m*n)
    """
    if not grid:
        return -1
    
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh_count = 0
    
    # Find all rotten oranges and count fresh ones
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c, 0))  # (row, col, time)
            elif grid[r][c] == 1:
                fresh_count += 1
    
    if fresh_count == 0:
        return 0
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    max_time = 0
    
    while queue:
        row, col, time = queue.popleft()
        max_time = max(max_time, time)
        
        # Spread to adjacent cells
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < rows and 
                0 <= new_col < cols and 
                grid[new_row][new_col] == 1):
                
                grid[new_row][new_col] = 2  # Mark as rotten
                fresh_count -= 1
                queue.append((new_row, new_col, time + 1))
    
    return max_time if fresh_count == 0 else -1
```

### Advanced Level

#### 6. Shortest Path in Grid with Obstacles Elimination
```python
from collections import deque

def shortest_path(grid, k):
    """
    LeetCode 1293: BFS with state tracking
    Time: O(m*n*k), Space: O(m*n*k)
    """
    rows, cols = len(grid), len(grid[0])
    
    # State: (row, col, obstacles_remaining)
    queue = deque([(0, 0, k, 0)])  # row, col, k_left, steps
    visited = {(0, 0, k)}
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while queue:
        row, col, k_left, steps = queue.popleft()
        
        # Reached destination
        if row == rows - 1 and col == cols - 1:
            return steps
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if 0 <= new_row < rows and 0 <= new_col < cols:
                new_k = k_left - grid[new_row][new_col]
                
                if new_k >= 0 and (new_row, new_col, new_k) not in visited:
                    visited.add((new_row, new_col, new_k))
                    queue.append((new_row, new_col, new_k, steps + 1))
    
    return -1
```

#### 7. Jump Game with BFS
```python
from collections import deque

def min_jumps_to_reach_end(arr):
    """
    LeetCode 1345: Minimum jumps to reach last index
    Can jump to i±1 or any index with same value
    Time: O(n), Space: O(n)
    """
    n = len(arr)
    if n <= 1:
        return 0
    
    # Map value to indices
    value_indices = {}
    for i, val in enumerate(arr):
        if val not in value_indices:
            value_indices[val] = []
        value_indices[val].append(i)
    
    queue = deque([(0, 0)])  # (index, steps)
    visited = {0}
    
    while queue:
        idx, steps = queue.popleft()
        
        if idx == n - 1:
            return steps
        
        # Jump to adjacent indices
        for next_idx in [idx - 1, idx + 1]:
            if 0 <= next_idx < n and next_idx not in visited:
                visited.add(next_idx)
                queue.append((next_idx, steps + 1))
        
        # Jump to indices with same value
        if arr[idx] in value_indices:
            for next_idx in value_indices[arr[idx]]:
                if next_idx not in visited:
                    visited.add(next_idx)
                    queue.append((next_idx, steps + 1))
            # Clear to avoid revisiting
            del value_indices[arr[idx]]
    
    return -1
```

#### 8. Snakes and Ladders
```python
from collections import deque

def snakes_and_ladders(board):
    """
    LeetCode 909: Minimum moves to reach end
    Time: O(n²), Space: O(n²)
    """
    n = len(board)
    
    def get_position(square):
        """Convert square number to (row, col)"""
        square -= 1  # 0-indexed
        row = n - 1 - (square // n)
        col = square % n
        # Reverse column for odd rows (bottom-up)
        if (n - 1 - row) % 2 == 1:
            col = n - 1 - col
        return row, col
    
    queue = deque([(1, 0)])  # (square, moves)
    visited = {1}
    
    while queue:
        square, moves = queue.popleft()
        
        # Try all dice rolls (1-6)
        for dice in range(1, 7):
            next_square = square + dice
            
            if next_square > n * n:
                continue
            
            row, col = get_position(next_square)
            
            # Check for ladder or snake
            if board[row][col] != -1:
                next_square = board[row][col]
            
            # Reached end
            if next_square == n * n:
                return moves + 1
            
            if next_square not in visited:
                visited.add(next_square)
                queue.append((next_square, moves + 1))
    
    return -1
```

---

## Pitfalls & Edge Cases

### Common Mistakes

#### 1. Using list.pop(0) in Python
```python
# ❌ WRONG - O(n) dequeue
class BadQueue:
    def __init__(self):
        self.items = []
    
    def enqueue(self, x):
        self.items.append(x)
    
    def dequeue(self):
        return self.items.pop(0)  # Shifts all elements!

# ✅ CORRECT - O(1) dequeue
from collections import deque

class GoodQueue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, x):
        self.items.append(x)
    
    def dequeue(self):
        return self.items.popleft()
```

#### 2. Forgetting to Check Empty Queue
```python
# ❌ WRONG - Will crash on empty queue
def process_queue(queue):
    return queue.popleft()  # IndexError if empty

# ✅ CORRECT
def process_queue(queue):
    if not queue:
        raise ValueError("Cannot dequeue from empty queue")
    return queue.popleft()

# Or return None/sentinel value
def process_queue_safe(queue):
    return queue.popleft() if queue else None
```

#### 3. BFS Level Processing Error
```python
# ❌ WRONG - Mixes levels
def level_order_wrong(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        result.append(node.val)  # All nodes in one list!
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return result

# ✅ CORRECT - Separate levels
def level_order_correct(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)  # KEY: Capture level size
        current_level = []
        
        for _ in range(level_size):  # Process entire level
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result
```

#### 4. Circular Queue Index Management
```python
# ❌ WRONG - Doesn't handle wrap-around
class BadCircularQueue:
    def __init__(self, k):
        self.queue = [0] * k
        self.front = 0
        self.rear = 0
    
    def enqueue(self, val):
        self.rear += 1  # Will exceed array bounds!
        self.queue[self.rear] = val

# ✅ CORRECT - Modulo for wrap-around
class GoodCircularQueue:
    def __init__(self, k):
        self.queue = [0] * k
        self.size = k
        self.front = 0
        self.rear = -1
        self.count = 0
    
    def enqueue(self, val):
        if self.count == self.size:
            return False
        self.rear = (self.rear + 1) % self.size  # Wrap around
        self.queue[self.rear] = val
        self.count += 1
        return True
```

### Edge Cases to Test

```python
# 1. Empty queue operations
queue = deque()
# queue.popleft()  # Should handle gracefully

# 2. Single element
queue = deque([1])
queue.popleft()
# Queue should now be empty

# 3. Capacity limits (circular queue)
cq = CircularQueue(3)
cq.enqueue(1)
cq.enqueue(2)
cq.enqueue(3)
# cq.enqueue(4)  # Should return False/handle gracefully

# 4. Alternating enqueue/dequeue
q = deque()
q.append(1)
q.popleft()  # Queue empty again
q.append(2)
# Should work correctly

# 5. BFS on empty tree
result = level_order_traversal(None)
# Should return []

# 6. BFS on single node
root = TreeNode(1)
result = level_order_traversal(root)
# Should return [[1]]

# 7. Grid BFS with no valid path
grid = [[1, 1], [1, 1]]
# Should return -1 or appropriate value

# 8. Large inputs
# Test with 10,000+ elements to ensure O(1) operations
```

---

## Quick Reference

### When to Use Queue
✅ BFS traversal (graphs, trees)  
✅ Level-order processing  
✅ Shortest path (unweighted)  
✅ Task scheduling  
✅ Breadth-first search  
✅ Process data in FIFO order  
✅ Sliding window problems  
✅ Stream processing  

### Python Queue Options
```python
# Standard queue - RECOMMENDED
from collections import deque
q = deque()

# Priority queue
import heapq
pq = []
heapq.heappush(pq, item)

# Thread-safe queue
from queue import Queue
q = Queue()

# LIFO queue (actually a stack)
from queue import LifoQueue
```

### Essential Operations Cheat Sheet
```python
from collections import deque

q = deque()

# Basic operations
q.append(x)           # Enqueue - O(1)
q.popleft()          # Dequeue - O(1)
q[0]                 # Peek - O(1)
len(q)               # Size - O(1)
not q                # Is empty - O(1)

# Deque-specific
q.appendleft(x)      # Add to front
q.pop()              # Remove from rear
q.clear()            # Remove all
q.rotate(n)          # Rotate n steps
```

### BFS Template
```python
from collections import deque

def bfs(start):
    queue = deque([start])
    visited = {start}
    
    while queue:
        node = queue.popleft()
        
        # Process node
        process(node)
        
        # Add neighbors
        for neighbor in get_neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### Level-Order Template
```python
from collections import deque

def level_order(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result
```

### Interview Preparation Checklist

**Theory:**
- [ ] Understand FIFO principle
- [ ] Know all queue types (simple, circular, deque, priority)
- [ ] Understand time complexity of operations
- [ ] Know when to use queue vs stack vs heap

**Implementation:**
- [ ] Implement queue with linked list
- [ ] Implement queue with two stacks
- [ ] Implement circular queue
- [ ] Use collections.deque efficiently

**Common Problems:**
- [ ] BFS traversal (graphs and trees)
- [ ] Level-order traversal
- [ ] Shortest path problems
- [ ] Sliding window maximum
- [ ] Task scheduling with cooldown
- [ ] Rotting oranges / multi-source BFS

**Advanced Topics:**
- [ ] Monotonic queue
- [ ] Priority queue / heap
- [ ] BFS with state tracking
- [ ] Multi-source BFS

---

## Interview Tips

### What Interviewers Look For

1. **Understanding of FIFO**: Can you explain why queue is needed?
2. **Implementation choice**: Do you know when to use deque vs list?
3. **Edge case handling**: Empty queue, single element, capacity limits
4. **Time complexity**: Can you analyze and optimize?
5. **BFS mastery**: Most queue problems involve BFS
6. **Code clarity**: Clean, readable, well-structured code

### Common Interview Questions to Prepare

**Must Know:**
- Implement queue using stacks (LeetCode 232)
- Design circular queue (LeetCode 622)
- Binary tree level order traversal (LeetCode 102)
- Rotting oranges (LeetCode 994)
- Number of islands (LeetCode 200)

**Should Know:**
- Sliding window maximum (LeetCode 239)
- Perfect squares (LeetCode 279)
- Snakes and ladders (LeetCode 909)
- Word ladder (LeetCode 127)
- Course schedule (LeetCode 207)

**Advanced:**
- Shortest path with obstacles elimination (LeetCode 1293)
- Jump game (LeetCode 1345)
- Sliding window median (LeetCode 480)

### Key Phrases for Interviews

When queue is appropriate:
- "We need to process elements in FIFO order"
- "BFS will give us the shortest path"
- "We need level-by-level processing"
- "Let's use a queue to track states to explore"
- "A deque allows us to efficiently add/remove from both ends"

---

**Good luck with your interviews! Master these patterns and you'll handle any queue problem thrown at you! 🚀**
