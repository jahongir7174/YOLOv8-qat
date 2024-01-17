import torch
import unittest
from unittest.mock import patch, MagicMock
from utils.tester import Tester
from typing import Any

class TestTester(unittest.TestCase):
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='image1.jpg\nimage2.jpg')
    def test_get_validation_filenames(self, mock_open: unittest.mock.MagicMock) -> None:
        """
        Test the get_validation_filenames method of the Tester class.
        """
        tester = Tester(args=MagicMock(), params=MagicMock())
        filenames = tester.get_validation_filenames()
        expected_filenames = ['../Dataset/COCO/images/val2017/image1.jpg', '../Dataset/COCO/images/val2017/image2.jpg']
        self.assertEqual(filenames, expected_filenames)

        # Verify that open was called with the correct path
        mock_open.assert_called_once_with('../Dataset/COCO/val2017.txt')

    @patch('utils.tester.DataLoader')
    @patch('utils.tester.Dataset')
    def test_prepare_data_loader(self, mock_dataset: MagicMock, mock_dataloader: MagicMock) -> None:
        """
        Test the prepare_data_loader method of the Tester class.
        """
        tester = Tester(args=MagicMock(input_size=640, batch_size=32), params=MagicMock())
        filenames = ['image1.jpg', 'image2.jpg']
        tester.prepare_data_loader(filenames)
        mock_dataset.assert_called_once()
        mock_dataloader.assert_called_once()

    @patch('torch.jit.load', return_value=MagicMock())
    def test_load_model(self, mock_load: MagicMock) -> None:
        """
        Test the load_model method of the Tester class.
        """
        tester = Tester(args=MagicMock(), params=MagicMock())
        model = tester.load_model()
        self.assertIsNotNone(model)
        mock_load.assert_called_once_with(f='./weights/best.ts')

    @patch('utils.tester.tqdm')
    @patch('utils.tester.non_max_suppression', return_value=[MagicMock()])
    @patch('utils.tester.compute_metric', return_value=MagicMock())
    @patch('utils.tester.compute_ap', return_value=(0, 0, 0, 0, 0, 0))
    def test_evaluate(self, mock_compute_ap: MagicMock, mock_compute_metric: MagicMock, 
                      mock_nms: MagicMock, mock_tqdm: MagicMock) -> None:
        """
        Test the evaluate method of the Tester class.
        """
        tester = Tester(args=MagicMock(input_size=640, batch_size=32), params=MagicMock())
        loader = MagicMock()
        model = MagicMock()
        device = torch.device('cpu')
        iou_v = torch.linspace(0.5, 0.95, 10, device=device)
        n_iou = iou_v.numel()
        results = tester.evaluate(loader, model, device, iou_v, n_iou)
        self.assertIsNotNone(results)

        # Verify that the mocked functions were called during evaluation
        mock_compute_ap.assert_called()  # Check if compute_ap is called
        mock_compute_metric.assert_called()  # Check if compute_metric is called
        mock_nms.assert_called()  # Check if non_max_suppression is called
        mock_tqdm.assert_called()  # Check if tqdm is used for the progress bar

    def test_print_results(self) -> None:
        """
        Test the print_results method of the Tester class.
        """
        tester = Tester(args=MagicMock(), params=MagicMock())
        with patch('builtins.print') as mock_print:
            tester.print_results(0.8, 0.7, 0.6, 0.5)
            mock_print.assert_called_with("Precision: 0.8000, Recall: 0.7000, mAP@0.5: 0.6000, mAP: 0.5000")

if __name__ == '__main__':
    unittest.main()