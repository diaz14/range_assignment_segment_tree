"""Range-assignment segment tree with sum, min, and max aggregates.

Each node stores the aggregate over its interval. A pending assignment of a
constant ``c`` is stored in ``lazy`` and is applied to the whole interval
immediately, but is only pushed to the children when a later operation needs
to descend into them. This keeps a range assignment O(log n) amortized while
still answering range queries in O(log n).

Why lazy propagation instead of rebuilding: a range assignment to a constant
makes every leaf in the interval equal, so the aggregate can be computed
without touching the leaves. Deferring the push to children preserves that
saving across consecutive assignments that fully cover the same nodes.
"""

from __future__ import annotations

from typing import List, Optional


class RangeAssignmentSegmentTree:
    """Segment tree over a fixed-size array supporting range assignment and
    range queries for sum, min, and max.

    The tree is built once from an initial sequence and then mutated in place.
    Indices are 0-based and half-open: ``[left, right)``.

    A pending assignment is represented by a non-``None`` ``lazy`` value at a
    node; ``None`` means "children are up to date with respect to this node".
    Using ``None`` rather than a sentinel value keeps arbitrary integers valid
    as assigned constants.
    """

    def __init__(self, values: List[int]) -> None:
        if not values:
            raise ValueError("values must be non-empty")
        self._n = len(values)
        size = 4 * self._n
        self._sum: List[int] = [0] * size
        self._min: List[int] = [0] * size
        self._max: List[int] = [0] * size
        self._lazy: List[Optional[int]] = [None] * size
        self._build(1, 0, self._n, values)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self, node: int, lo: int, hi: int, values: List[int]) -> None:
        if hi - lo == 1:
            v = values[lo]
            self._sum[node] = v
            self._min[node] = v
            self._max[node] = v
            return
        mid = (lo + hi) // 2
        self._build(node * 2, lo, mid, values)
        self._build(node * 2 + 1, mid, hi, values)
        self._pull(node)

    def _pull(self, node: int) -> None:
        left = node * 2
        right = left + 1
        self._sum[node] = self._sum[left] + self._sum[right]
        self._min[node] = min(self._min[left], self._min[right])
        self._max[node] = max(self._max[left], self._max[right])

    # ------------------------------------------------------------------
    # Lazy helpers
    # ------------------------------------------------------------------

    def _apply(self, node: int, lo: int, hi: int, value: int) -> None:
        """Set the whole interval to ``value`` and record the pending push."""
        length = hi - lo
        self._sum[node] = value * length
        self._min[node] = value
        self._max[node] = value
        self._lazy[node] = value

    def _push(self, node: int, lo: int, hi: int) -> None:
        """Propagate a pending assignment to both children, if any."""
        pending = self._lazy[node]
        if pending is None:
            return
        mid = (lo + hi) // 2
        self._apply(node * 2, lo, mid, pending)
        self._apply(node * 2 + 1, mid, hi, pending)
        self._lazy[node] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign(self, left: int, right: int, value: int) -> None:
        """Assign ``value`` to every index in ``[left, right)``."""
        self._check_range(left, right)
        self._assign(1, 0, self._n, left, right, value)

    def _assign(
        self, node: int, lo: int, hi: int, left: int, right: int, value: int
    ) -> None:
        if right <= lo or hi <= left:
            return
        if left <= lo and hi <= right:
            self._apply(node, lo, hi, value)
            return
        self._push(node, lo, hi)
        mid = (lo + hi) // 2
        self._assign(node * 2, lo, mid, left, right, value)
        self._assign(node * 2 + 1, mid, hi, left, right, value)
        self._pull(node)

    def sum(self, left: int, right: int) -> int:
        """Sum of values in ``[left, right)``."""
        self._check_range(left, right)
        return self._query_sum(1, 0, self._n, left, right)

    def _query_sum(self, node: int, lo: int, hi: int, left: int, right: int) -> int:
        if right <= lo or hi <= left:
            return 0
        if left <= lo and hi <= right:
            return self._sum[node]
        self._push(node, lo, hi)
        mid = (lo + hi) // 2
        return self._query_sum(node * 2, lo, mid, left, right) + self._query_sum(
            node * 2 + 1, mid, hi, left, right
        )

    def min(self, left: int, right: int) -> int:
        """Minimum value in ``[left, right)``."""
        self._check_range(left, right)
        return self._query_min(1, 0, self._n, left, right)

    def _query_min(self, node: int, lo: int, hi: int, left: int, right: int) -> int:
        if right <= lo or hi <= left:
            return float("inf")
        if left <= lo and hi <= right:
            return self._min[node]
        self._push(node, lo, hi)
        mid = (lo + hi) // 2
        return min(
            self._query_min(node * 2, lo, mid, left, right),
            self._query_min(node * 2 + 1, mid, hi, left, right),
        )

    def max(self, left: int, right: int) -> int:
        """Maximum value in ``[left, right)``."""
        self._check_range(left, right)
        return self._query_max(1, 0, self._n, left, right)

    def _query_max(self, node: int, lo: int, hi: int, left: int, right: int) -> int:
        if right <= lo or hi <= left:
            return float("-inf")
        if left <= lo and hi <= right:
            return self._max[node]
        self._push(node, lo, hi)
        mid = (lo + hi) // 2
        return max(
            self._query_max(node * 2, lo, mid, left, right),
            self._query_max(node * 2 + 1, mid, hi, left, right),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _check_range(self, left: int, right: int) -> None:
        if left < 0 or right > self._n or left >= right:
            raise ValueError(
                f"invalid range [{left}, {right}) for size {self._n}"
            )
