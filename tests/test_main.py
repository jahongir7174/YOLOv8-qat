import unittest

from main import main
from utils.trainer import Trainer
from utils.tester import Tester


class TestMain(unittest.TestCase):
    def test_main(self):
        # Test the main function
        # Create test cases to cover different scenarios and edge cases
        # Use assert statements to check if the actual output matches the expected output
        self.assertIsNone(main())
        

if __name__ == '__main__':
    unittest.main()
