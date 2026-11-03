import unittest

from range_assignment_segment_tree import RangeAssignmentSegmentTree


class TestRangeAssignmentSegmentTree(unittest.TestCase):
    def test_initial_aggregates(self):
        tree = RangeAssignmentSegmentTree([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertEqual(tree.sum(0, 8), 31)
        self.assertEqual(tree.min(0, 8), 1)
        self.assertEqual(tree.max(0, 8), 9)
        self.assertEqual(tree.sum(2, 5), 4 + 1 + 5)
        self.assertEqual(tree.min(2, 5), 1)
        self.assertEqual(tree.max(2, 5), 5)

    def test_single_element(self):
        tree = RangeAssignmentSegmentTree([42])
        self.assertEqual(tree.sum(0, 1), 42)
        self.assertEqual(tree.min(0, 1), 42)
        self.assertEqual(tree.max(0, 1), 42)
        tree.assign(0, 1, -7)
        self.assertEqual(tree.sum(0, 1), -7)
        self.assertEqual(tree.min(0, 1), -7)
        self.assertEqual(tree.max(0, 1), -7)

    def test_full_range_assignment(self):
        tree = RangeAssignmentSegmentTree([1, 2, 3, 4])
        tree.assign(0, 4, 10)
        self.assertEqual(tree.sum(0, 4), 40)
        self.assertEqual(tree.min(0, 4), 10)
        self.assertEqual(tree.max(0, 4), 10)
        self.assertEqual(tree.sum(1, 3), 20)
        self.assertEqual(tree.min(1, 3), 10)
        self.assertEqual(tree.max(1, 3), 10)

    def test_partial_assignment(self):
        tree = RangeAssignmentSegmentTree([1, 2, 3, 4, 5])
        tree.assign(1, 4, 0)
        self.assertEqual(tree.sum(0, 5), 1 + 0 + 0 + 0 + 5)
        self.assertEqual(tree.min(0, 5), 0)
        self.assertEqual(tree.max(0, 5), 5)
        self.assertEqual(tree.sum(2, 3), 0)
        self.assertEqual(tree.min(2, 3), 0)
        self.assertEqual(tree.max(2, 3), 0)

    def test_overlapping_assignments(self):
        tree = RangeAssignmentSegmentTree([5, 5, 5, 5, 5])
        tree.assign(0, 3, 1)
        tree.assign(2, 5, 2)
        self.assertEqual(tree.sum(0, 5), 1 + 1 + 2 + 2 + 2)
        self.assertEqual(tree.min(0, 5), 1)
        self.assertEqual(tree.max(0, 5), 2)
        self.assertEqual(tree.sum(0, 2), 2)
        self.assertEqual(tree.sum(2, 5), 6)

    def test_negative_values(self):
        tree = RangeAssignmentSegmentTree([-1, -2, -3, -4])
        self.assertEqual(tree.min(0, 4), -4)
        self.assertEqual(tree.max(0, 4), -1)
        self.assertEqual(tree.sum(0, 4), -10)
        tree.assign(1, 3, -10)
        self.assertEqual(tree.min(0, 4), -10)
        self.assertEqual(tree.max(0, 4), -1)
        self.assertEqual(tree.sum(0, 4), -1 - 10 - 10 - 4)

    def test_zero_assignment(self):
        tree = RangeAssignmentSegmentTree([7, 8, 9])
        tree.assign(0, 3, 0)
        self.assertEqual(tree.sum(0, 3), 0)
        self.assertEqual(tree.min(0, 3), 0)
        self.assertEqual(tree.max(0, 3), 0)
        tree.assign(1, 2, 100)
        self.assertEqual(tree.sum(0, 3), 100)
        self.assertEqual(tree.min(0, 3), 0)
        self.assertEqual(tree.max(0, 3), 100)

    def test_repeated_queries_after_assignment(self):
        tree = RangeAssignmentSegmentTree([1, 2, 3, 4, 5, 6])
        tree.assign(2, 5, 9)
        for _ in range(3):
            self.assertEqual(tree.sum(1, 5), 2 + 9 + 9 + 9)
            self.assertEqual(tree.min(1, 5), 2)
            self.assertEqual(tree.max(1, 5), 9)

    def test_range_validation(self):
        tree = RangeAssignmentSegmentTree([1, 2, 3])
        with self.assertRaises(ValueError):
            tree.sum(0, 4)
        with self.assertRaises(ValueError):
            tree.sum(-1, 2)
        with self.assertRaises(ValueError):
            tree.sum(2, 2)
        with self.assertRaises(ValueError):
            tree.assign(1, 0, 5)

    def test_empty_initial_values_rejected(self):
        with self.assertRaises(ValueError):
            RangeAssignmentSegmentTree([])

    def test_adjacent_ranges(self):
        tree = RangeAssignmentSegmentTree([1, 1, 1, 1])
        tree.assign(0, 2, 5)
        tree.assign(2, 4, 7)
        self.assertEqual(tree.sum(0, 1), 5)
        self.assertEqual(tree.sum(1, 2), 5)
        self.assertEqual(tree.sum(2, 3), 7)
        self.assertEqual(tree.sum(3, 4), 7)
        self.assertEqual(tree.sum(0, 4), 24)
        self.assertEqual(tree.min(0, 4), 5)
        self.assertEqual(tree.max(0, 4), 7)

    def test_interleaved_assign_and_query(self):
        tree = RangeAssignmentSegmentTree([0, 0, 0, 0, 0])
        tree.assign(0, 5, 1)
        self.assertEqual(tree.sum(0, 5), 5)
        tree.assign(1, 4, 2)
        self.assertEqual(tree.sum(0, 5), 1 + 2 + 2 + 2 + 1)
        tree.assign(2, 3, 3)
        self.assertEqual(tree.sum(0, 5), 1 + 2 + 3 + 2 + 1)
        self.assertEqual(tree.min(0, 5), 1)
        self.assertEqual(tree.max(0, 5), 3)


if __name__ == "__main__":
    unittest.main()
