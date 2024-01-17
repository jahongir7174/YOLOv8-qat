import unittest
import cv2
import numpy as np
import torch
from typing import Tuple, List
from utils.dataset import (Albumentations, Dataset, augment_hsv, mix_up,
                           random_perspective, resample, resize, wh2xy, xy2wh)


class TestDataset(unittest.TestCase):
    def test_load_image(self):
        """
        Test the load_image function from the Dataset class.

        This test verifies that the load_image function returns an image and its shape correctly.
        """
        dataset = Dataset(filenames=["path/to/image.jpg"], input_size=640, params={}, augment=True)
        image, shape = dataset.load_image(0)
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(len(shape), 2)

    def test_load_mosaic(self):
        """
        Test the load_mosaic function from the Dataset class.

        This test checks if the load_mosaic function returns a numpy array for both the image and the label.
        """
        dataset = Dataset(filenames=["path/to/image1.jpg", "path/to/image2.jpg"], input_size=640, params={}, augment=True)
        image, label = dataset.load_mosaic(0, dataset.params)
        self.assertIsInstance(image, np.ndarray)
        self.assertIsInstance(label, np.ndarray)

    def test_resize(self):
        """
        Test the resize function.

        This test ensures that the resize function properly returns a numpy array of the resized image.
        """
        image = cv2.imread("path/to/image.jpg")
        resized_image, _, _, _, _ = resize(image, 640)
        self.assertIsInstance(resized_image, np.ndarray)

    def test_wh2xy(self):
        """
        Test the wh2xy function.

        This test checks if the wh2xy function converts width-height format to x-y format correctly.
        """
        box = np.array([[0.5, 0.5, 0.1, 0.1]])
        converted_box = wh2xy(box)
        self.assertIsInstance(converted_box, np.ndarray)

    def test_xy2wh(self):
        """
        Test the xy2wh function.

        This test verifies that the xy2wh function accurately converts x-y format to width-height format.
        """
        box = np.array([[320, 320, 384, 384]])
        converted_box = xy2wh(box, 640, 640)
        self.assertIsInstance(converted_box, np.ndarray)

    def test_resample(self):
        """
        Test the resample function.

        This test checks if the resample function returns one of the expected interpolation methods.
        """
        self.assertIn(resample(), [cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4])

    def test_augment_hsv(self):
        """
        Test the augment_hsv function.

        This test confirms that the augment_hsv function applies HSV augmentation to an image and returns a numpy array.
        """
        image = cv2.imread("path/to/image.jpg")
        params = {'hsv_h': 0.5, 'hsv_s': 0.5, 'hsv_v': 0.5}
        augment_hsv(image, params)
        self.assertIsInstance(image, np.ndarray)

    def test_random_perspective(self):
        """
        Test the random_perspective function.

        This test ensures that the random_perspective function applies a perspective transformation and returns a numpy array.
        """
        image = cv2.imread("path/to/image.jpg")
        transformed_image, _ = random_perspective(image)
        self.assertIsInstance(transformed_image, np.ndarray)

    def test_mix_up(self):
        """
        Test the mix_up function.

        This test checks whether the mix_up function correctly blends two images and their labels into a single output.
        """
        image1 = cv2.imread("path/to/image1.jpg")
        image2 = cv2.imread("path/to/image2.jpg")
        mixed_image, _ = mix_up(image1, np.array([[0, 0.5, 0.5, 0.1, 0.1]]), image2, np.array([[1, 0.5, 0.5, 0.1, 0.1]]))
        self.assertIsInstance(mixed_image, np.ndarray)

    def test_albumentations(self):
        """
        Test the Albumentations class.

        This test verifies that the Albumentations class correctly applies a series of augmentations to an image, bounding boxes, and labels.
        """
        albumentations_transform = Albumentations()
        image = cv2.imread("path/to/image.jpg")
        boxes = np.array([[0, 0.5, 0.5, 0.1, 0.1]])
        labels = np.array([0])
        transformed_image, transformed_boxes, transformed_labels = albumentations_transform(image, boxes, labels)
        self.assertIsInstance(transformed_image, torch.Tensor)
        self.assertIsInstance(transformed_boxes, np.ndarray)
        self.assertIsInstance(transformed_labels, np.ndarray)

    def test_dataset_getitem(self):
        """
        Test the __getitem__ method of the Dataset class.

        This test checks if the Dataset's __getitem__ method returns the correct tuple format for a given dataset item.
        """
        dataset = Dataset(filenames=["path/to/image.jpg"], input_size=640, params={}, augment=True)
        item = dataset[0]
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 4)

    def test_dataset_len(self):
        """
        Test the __len__ method of the Dataset class.

        This test ensures that the __len__ method accurately counts the number of items in the dataset.
        """
        dataset = Dataset(filenames=["path/to/image.jpg", "path/to/image2.jpg"], input_size=640, params={}, augment=True)
        self.assertEqual(len(dataset), 2)

if __name__ == '__main__':
    unittest.main()