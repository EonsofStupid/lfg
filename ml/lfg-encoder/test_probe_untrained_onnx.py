import unittest

import probe_untrained_onnx as probe


class ProbeShapesTests(unittest.TestCase):
    def test_requires_explicit_unique_bounded_shapes(self):
        self.assertEqual(probe.shapes(['1:1', '2:32'], 64), [(1, 1), (2, 32)])
        for value in ([], ['0:1'], ['1:0'], ['1:65'], ['1:1', '1:1'], ['1:-1'], ['1:1:1']):
            with self.subTest(value=value), self.assertRaises(ValueError):
                probe.shapes(value, 64)


if __name__ == '__main__':
    unittest.main()
