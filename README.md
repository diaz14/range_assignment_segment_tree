# range_assignment_segment_tree

A segment tree that supports range assignment (set every element in an interval to a constant) and range queries for sum, min, and max in O(log n) per operation.

```python
from range_assignment_segment_tree import RangeAssignmentSegmentTree

tree = RangeAssignmentSegmentTree([3, 1, 4, 1, 5])
tree.assign(1, 4, 0)          # [3, 0, 0, 0, 5]
print(tree.sum(0, 5))         # 8
print(tree.min(0, 5))         # 0
print(tree.max(0, 5))         # 5
```

## Why this exists

Range assignment is the operation that ruins a naive segment tree: updating a contiguous block to a single value touches every leaf in the block, and doing that repeatedly is quadratic. The fix is lazy propagation, but the usual lazy-add tree cannot reuse its pending state for assignment because summing a pending add is different from overwriting with a constant. This library keeps a separate lazy slot that stores an optional assigned constant. When a node is fully covered by an assignment, its aggregate is recomputed in O(1) and the constant is parked on the node; children are only updated when a later query or partial update actually descends into them. The trade-off is an extra array of optional values and a small constant-factor cost on every push, in exchange for worst-case O(log n) per operation instead of O(k) for a k-element range.

## The edge you will hit

Ranges are half-open, `[left, right)`. An empty range such as `sum(2, 2)` raises `ValueError` rather than returning `0` or `None`, because the tree has no meaningful aggregate for an empty interval and silently returning a neutral element hides off-by-one errors in callers that compute endpoints from lengths. All indices must satisfy `0 <= left < right <= n`.

## API

`RangeAssignmentSegmentTree(values)`

- `values`: a non-empty list of integers. The tree is built immediately.

Methods:

- `assign(left, right, value)` — set every index in `[left, right)` to `value`.
- `sum(left, right)` — sum of the values in `[left, right)`.
- `min(left, right)` — minimum value in `[left, right)`.
- `max(left, right)` — maximum value in `[left, right)`.

All four methods raise `ValueError` if the range is empty or out of bounds. The constructor raises `ValueError` if `values` is empty.

## Implementation notes

The tree is stored in four parallel arrays sized `4 * n`, indexed from 1 at the root. `lazy[node]` is `None` when no assignment is pending and an integer otherwise. `None` is used instead of a sentinel because any integer, including `0`, is a valid assigned constant. Queries push pending assignments down before recursing, which keeps the stored aggregates consistent with the leaves at all times.
