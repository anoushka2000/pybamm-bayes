#
# Basic test to make CI works
# 
import unittest

class TestCI(unittest.TestCase):
    def test_ci(self):
        self.assertEqual(1, 1)

if __name__ == "__main__":
    unittest.main()