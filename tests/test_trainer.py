import unittest
from unittest.mock import patch, MagicMock
from utils.trainer import Trainer
from typing import Any, Tuple, Callable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

class TestTrainer(unittest.TestCase):

    def setUp(self) -> None:
        self.args: MagicMock = MagicMock()
        self.params: MagicMock = MagicMock()
        self.trainer: Trainer = Trainer(self.args, self.params)

    @patch('utils.trainer.util.setup_seed')
    @patch('utils.trainer.util.setup_multi_processes')
    def test_setup_training_environment(self, mock_setup_seed: MagicMock, mock_setup_multi_processes: MagicMock) -> None:
        """
        Test the setup_training_environment method of the Trainer class.
        """
        self.trainer.setup_training_environment()
        mock_setup_seed.assert_called_once()
        mock_setup_multi_processes.assert_called_once()

    @patch('torch.load', return_value={'model': MagicMock()})
    def test_load_and_prepare_model(self, mock_torch_load: MagicMock) -> None:
        """
        Test the load_and_prepare_model method of the Trainer class.
        """
        model = self.trainer.load_and_prepare_model()
        self.assertIsNotNone(model)
        mock_torch_load.assert_called_once_with('./weights/v8_n.pth')

    def test_configure_optimizer_and_scheduler(self) -> None:
        """
        Test the configure_optimizer_and_scheduler method of the Trainer class.
        """
        optimizer, scheduler = self.trainer.configure_optimizer_and_scheduler()
        self.assertIsInstance(optimizer, Optimizer)
        self.assertIsInstance(scheduler, LambdaLR)

    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='image1.jpg\nimage2.jpg')
    @patch('utils.trainer.Dataset')
    @patch('utils.trainer.data.DataLoader')
    def test_prepare_data_loader(self, mock_dataloader: MagicMock, mock_dataset: MagicMock, mock_open: MagicMock) -> None:
        """
        Test the prepare_data_loader method of the Trainer class.
        """
        loader = self.trainer.prepare_data_loader()
        mock_dataset.assert_called_once()
        mock_dataloader.assert_called_once()
        self.assertIsInstance(loader, DataLoader)

        # Check if the open function was called with the correct file path
        mock_open.assert_called_once_with('../Dataset/COCO/train2017.txt')

    def test_learning_rate(self) -> None:
        """
        Test the learning_rate method of the Trainer class.
        """
        lr_fn: Callable[[Any], float] = self.trainer.learning_rate()
        self.assertTrue(callable(lr_fn))

    @patch('utils.trainer.csv.DictWriter')
    def test_train(self, mock_csv_writer: MagicMock) -> None:
        """
        Test the train method of the Trainer class.
        """
        # Setup
        mock_writer = MagicMock()
        mock_csv_writer.return_value = mock_writer

        # Call train method
        best_mean_ap: float = self.trainer.train()

        # Assertions
        self.assertIsNotNone(best_mean_ap)
        mock_csv_writer.assert_called()  # Check if DictWriter was called
        mock_writer.writerow.assert_called()  # Check if write operation was performed

        # Add additional checks if necessary, e.g., check the content of written rows
        # Here we assume that the DictWriter is used to write training metrics in each epoch
        # We can assert that the writer wrote rows corresponding to each epoch of training
        num_epochs = self.args.epochs
        self.assertEqual(mock_writer.writerow.call_count, num_epochs)

        # Optionally, inspect the specific contents of the written rows
        # This part depends on the actual data structure you expect
        # Example:
        for call_args in mock_writer.writerow.call_args_list:
            written_row = call_args[0][0]  # Extract the dictionary passed to write operation
            self.assertIn('epoch', written_row)
            self.assertIn('box', written_row)
            self.assertIn('cls', written_row)
            self.assertIn('mAP', written_row)
            # Add more field checks as per your requirement


    def test_warmup_lr_and_momentum(self) -> None:
        """
        Test the _warmup_lr_and_momentum private method of the Trainer class.
        """
        x: int = 0
        num_warmup: int = 100
        self.trainer._warmup_lr_and_momentum(x, num_warmup)
        # Check if the learning rate and momentum have been set
        for group in self.trainer.optimizer.param_groups:
            self.assertIn('lr', group)
            self.assertIn('momentum', group)

    @patch('utils.trainer.util.ComputeLoss')
    def test_forward_and_backward(self, mock_compute_loss: MagicMock) -> None:
        """
        Test the _forward_and_backward private method of the Trainer class.
        """
        samples: MagicMock = MagicMock()
        targets: MagicMock = MagicMock()
        criterion: MagicMock = mock_compute_loss.return_value
        loss_box, loss_cls = self.trainer._forward_and_backward(samples, targets, criterion)
        self.assertIsNotNone(loss_box)
        self.assertIsNotNone(loss_cls)

    def test_optimize(self) -> None:
        """
        Test the _optimize private method of the Trainer class.
        """
        accumulate: int = 1
        self.trainer._optimize(accumulate)
        # Check if the optimizer has stepped
        self.trainer.optimizer.step.assert_called_once()

    def test_log_progress(self) -> None:
        """
        Test the _log_progress private method of the Trainer class.
        """
        # Mocking the progress bar and average meter
        p_bar: MagicMock = MagicMock()
        avg_box_loss: MagicMock = MagicMock()
        avg_cls_loss: MagicMock = MagicMock()
        epoch: int = 0
        self.trainer._log_progress(epoch, avg_box_loss, avg_cls_loss, p_bar)
        p_bar.set_description.assert_called()

    def test_convert_and_test(self) -> None:
        """
        Test the _convert_and_test private method of the Trainer class.
        """
        last: Tuple[float, float, float, float] = self.trainer._convert_and_test()
        self.assertIsNotNone(last)

    def test_write_to_csv(self) -> None:
        """
        Test the _write_to_csv private method of the Trainer class.
        """
        epoch:int = 0
        avg_box_loss: MagicMock = MagicMock()
        avg_cls_loss: MagicMock = MagicMock()
        last: Tuple[float, float, float, float] = (0, 0, 0, 0)
        writer: MagicMock = MagicMock()
        f: MagicMock = MagicMock()
        self.trainer._write_to_csv(epoch, avg_box_loss, avg_cls_loss, last, writer, f)
        writer.writerow.assert_called()
    def test_update_best_map(self) -> None:
        """
        Test the _update_best_map private method of the Trainer class.
        """
        best_mean_ap: float = 0
        last: Tuple[float, float, float, float] = (0.5, 0, 0, 0)
        updated_best_map: float = self.trainer._update_best_map(best_mean_ap, last)
        self.assertEqual(updated_best_map, 0.5)

if __name__ == '__main__':
    unittest.main()