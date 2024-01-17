import unittest
from unittest.mock import patch, MagicMock
import os
from typing import Any, Dict

# Import the main function
from main import main

class TestMain(unittest.TestCase):

    @patch('main.Trainer')
    @patch('main.Tester')
    @patch('main.util.load_config')
    @patch('os.makedirs')
    @patch('os.getenv')
    @patch('sys.argv', ['main.py', '--train'])
    def test_main_train(self, mock_getenv: MagicMock, mock_makedirs: MagicMock, 
                        mock_load_config: MagicMock, mock_tester: MagicMock, 
                        mock_trainer: MagicMock) -> None:
        """
        Test the main function for training scenario.
        """
        # Set up mock environment
        mock_getenv.side_effect = lambda x, default: {'LOCAL_RANK': '0', 'WORLD_SIZE': '1'}.get(x, default)
        mock_load_config.return_value = MagicMock()

        # Call the main function
        main()

        # Assertions to verify the correct calls were made
        mock_makedirs.assert_called_once_with('weights')
        mock_load_config.assert_called_once_with('utils/args.yaml')
        mock_trainer.assert_called_once()
        mock_tester.assert_not_called()  # Ensure Tester is not called during training

    @patch('main.Trainer')
    @patch('main.Tester')
    @patch('main.util.load_config')
    @patch('os.makedirs')
    @patch('os.getenv')
    @patch('sys.argv', ['main.py', '--test'])
    def test_main_test(self, mock_getenv: MagicMock, mock_makedirs: MagicMock, 
                    mock_load_config: MagicMock, mock_tester: MagicMock, 
                    mock_trainer: MagicMock) -> None:
        """
        Test the main function for testing scenario.
        """
        # Set up mock environment
        mock_getenv.side_effect = lambda x, default: {'LOCAL_RANK':'0', 'WORLD_SIZE': '1'}.get(x, default)
        mock_load_config.return_value = MagicMock()

        # Call the main function
        main()

        # Assertions to verify the correct calls were made
        mock_makedirs.assert_not_called()
        mock_load_config.assert_called_once_with('utils/args.yaml')
        mock_tester.assert_called_once()
        mock_trainer.assert_not_called()  # Ensure Trainer is not called during testing

if __name__ == '__main__':
    unittest.main()