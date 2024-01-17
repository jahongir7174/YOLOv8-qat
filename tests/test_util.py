import unittest
from unittest.mock import patch, MagicMock
import os
import platform
import numpy as np
import torch
from typing import List

from utils.util import compute_metric, make_anchors, non_max_suppression, setup_multi_processes, setup_seed, wh2xy

class TestUtil(unittest.TestCase):
    def test_setup_seed(self) -> None:
        """
        Test setup_seed function to ensure it sets the random seeds correctly.
        """
        with patch('random.seed') as mock_random_seed, \
            patch('np.random.seed') as mock_np_seed, \
            patch('torch.manual_seed') as mock_torch_seed:
            setup_seed(42)
            mock_random_seed.assert_called_with(42)
            mock_np_seed.assert_called_with(42)
            mock_torch_seed.assert_called_with(42)

    def test_setup_multi_processes(self) -> None:
        """
        Test setup_multi_processes function for proper environment setup.
        """
        with patch('torch.multiprocessing.set_start_method') as mock_set_start_method, \
            patch('cv2.setNumThreads') as mock_set_cv_threads, \
            patch.dict('os.environ', {}, clear=True):
            setup_multi_processes()
            mock_set_cv_threads.assert_called_with(0)
            self.assertEqual(os.environ.get('OMP_NUM_THREADS'), '1')
            self.assertEqual(os.environ.get('MKL_NUM_THREADS'), '1')
            if platform.system() != 'Windows':
                mock_set_start_method.assert_called_with('fork', force=True)

    def test_wh2xy(self) -> None:
        """
        Test wh2xy function to convert width-height format to x-y format.
        """
        input_tensor: torch.Tensor = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]])
        output: torch.Tensor = wh2xy(input_tensor)
        expected_output: torch.Tensor = torch.tensor([[0, 0, 20, 20], [10, 10, 50, 50]])
        self.assertTrue(torch.equal(output, expected_output))

    def test_make_anchors(self) -> None:
        """
        Test make_anchors function for correct anchor generation.
        """
        input_tensor: List[torch.Tensor] = [torch.randn(1, 3, 10, 10) for _ in range(3)]
        strides: List[int] = [8, 16, 32]
        anchors, stride_tensor = make_anchors(input_tensor, strides)
        self.assertEqual(anchors.shape[0], 300)
        self.assertEqual(stride_tensor.shape[0], 300)

    def test_compute_metric(self) -> None:
        """
        Test compute_metric function for metric computation.
        """
        output: torch.Tensor = torch.tensor([[10, 10, 20, 20, 0.9, 0]])
        target: torch.Tensor = torch.tensor([[0, 10, 10, 20, 20]])
    
        iou_v: torch.Tensor = torch.linspace(0.5, 0.95, 10)
        correct: torch.Tensor = compute_metric(output, target, iou_v)
        self.assertIsInstance(correct, torch.Tensor)

    def test_non_max_suppression(self) -> None:
        """
        Test non_max_suppression function for filtering detections.
        """
        outputs: List[torch.Tensor] = [torch.randn(1, 3, 10, 10) for _ in range(3)]
        conf_threshold: float = 0.5
        iou_threshold: float = 0.5
        nc: int = 80  # Number of classes
        nms_outputs: List[torch.Tensor] = non_max_suppression(outputs, conf_threshold, iou_threshold, nc)
        self.assertIsInstance(nms_outputs, list)
        for output in nms_outputs:
            self.assertIsInstance(output, torch.Tensor)
            if output.nelement() != 0:
                self.assertTrue((output[:, 4] >= conf_threshold).all())  # Check confidence threshold
                self.assertTrue((output[:, 5] < nc).all())  # Check class labels


if __name__ == '__main__':
    unittest.main()