import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision.transforms import functional as F
import math
import albumentations as A
from albumentations.pytorch import ToTensorV2

FORMATS = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp')


class Dataset(data.Dataset):
    def __init__(self, filenames, input_size, params, augment):
        self.params = params
        self.mosaic = augment
        self.augment = augment
        self.input_size = input_size
        self.albumentations = Albumentations()  # Initialize Albumentations here

        # Read labels
        self.labels = self.load_label(filenames)
        self.filenames = list(self.labels.keys())  # update
        self.n = len(self.filenames)  # number of samples
        self.indices = list(range(self.n))

    def __getitem__(self, index):
        index = self.indices[index]

        params = self.params
        mosaic = self.mosaic and random.random() < params['mosaic']

        if mosaic:
            # Load MOSAIC
            image, label = self.load_mosaic(index, params)
            # MixUp augmentation
            if random.random() < params['mix_up']:
                index = random.choice(self.indices)
                mix_image1, mix_label1 = image, label
                mix_image2, mix_label2 = self.load_mosaic(index, params)

                image, label = mix_up(mix_image1, mix_label1, mix_image2, mix_label2)
        else:
            # Load image
            image, shape = self.load_image(index)
            h, w = image.shape[:2]

            # Resize
            image, ratio, pad = resize(image, self.input_size, self.augment)

            label = self.labels[index].copy()
            if label.size:
                label[:, 1:] = wh2xy(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])
            if self.augment:
                image, label = random_perspective(image, label, params)

        nl = len(label)  # number of labels
        h, w = image.shape[:2]
        cls = label[:, 0:1]
        box = label[:, 1:5]
        box = xy2wh(box, w, h)

        if self.augment:
            # Albumentations
            image, box, cls = self.albumentations(image, box, cls)
            nl = len(box)  # update after albumentations
            # HSV color-space
            augment_hsv(image, params)
            # Flip up-down
            if random.random() < params['flip_ud']:
                image = np.flipud(image)
                if nl:
                    box[:, 1] = 1 - box[:, 1]
            # Flip left-right
            if random.random() < params['flip_lr']:
                image = np.fliplr(image)
                if nl:
                    box[:, 0] = 1 - box[:, 0]

        target_cls = torch.zeros((nl, 1))
        target_box = torch.zeros((nl, 4))
        if nl:
            target_cls = torch.from_numpy(cls)
            target_box = torch.from_numpy(box)

        # Convert HWC to CHW, BGR to RGB
        sample = image.transpose((2, 0, 1))
        sample = np.ascontiguousarray(sample[::-1])

        return torch.from_numpy(sample), target_cls, target_box, torch.zeros(nl)

    def __len__(self):
        return len(self.filenames)

    def load_image(self, i):
        image = cv2.imread(self.filenames[i])
        h, w = image.shape[:2]
        r = self.input_size / max(h, w)
        if r != 1:
            image = cv2.resize(image, dsize=(int(w * r), int(h * r)), interpolation=resample() if self.augment else cv2.INTER_LINEAR)
        return image, (h, w)

    def load_mosaic(self, index, params):
        label4 = []
        border = [-self.input_size // 2, -self.input_size // 2]
        image4 = np.full((self.input_size * 2, self.input_size * 2, 3), 0, dtype=np.uint8)
        y1a, y2a, x1a, x2a, y1b, y2b, x1b, x2b = (None, None, None, None, None, None, None, None)

        xc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))
        yc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))

        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)

        for i, index in enumerate(indices):
            # Load image
            image, _ = self.load_image(index)
            shape = image.shape
            if i == 0:  # top left
                x1a = max(xc - shape[1], 0)
                y1a = max(yc - shape[0], 0)
                x2a = xc
                y2a = yc
                x1b = shape[1] - (x2a - x1a)
                y1b = shape[0] - (y2a - y1a)
                x2b = shape[1]
                y2b = shape[0]
            if i == 1:  # top right
                x1a = xc
                y1a = max(yc - shape[0], 0)
                x2a = min(xc + shape[1], self.input_size * 2)
                y2a = yc
                x1b = 0
                y1b = shape[0] - (y2a - y1a)
                x2b = min(shape[1], x2a - x1a)
                y2b = shape[0]
            if i == 2:  # bottom left
                x1a = max(xc - shape[1], 0)
                y1a = yc
                x2a = xc
                y2a = min(self.input_size * 2, yc + shape[0])
                x1b = shape[1] - (x2a - x1a)
                y1b = 0
                x2b = shape[1]
                y2b = min(y2a - y1a, shape[0])
            if i == 3:  # bottom right
                x1a = xc
                y1a = yc
                x2a = min(xc + shape[1], self.input_size * 2)
                y2a = min(self.input_size * 2, yc + shape[0])
                x1b = 0
                y1b = 0
                x2b = min(shape[1], x2a - x1a)
                y2b = min(y2a - y1a, shape[0])

            image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            pad_w = x1a - x1b
            pad_h = y1a - y1b

            # Labels
            label = self.labels[index].copy()
            if len(label):
                label[:, 1:] = wh2xy(label[:, 1:], shape[1], shape[0], pad_w, pad_h)
            label4.append(label)

        # Concat/clip labels
        label4 = np.concatenate(label4, 0)
        for x in label4[:, 1:]:
            np.clip(x, 0, 2 * self.input_size, out=x)

        # Augment
        image4, label4 = random_perspective(image4, label4, params, border)

        return image4, label4

    @staticmethod
    def collate_fn(batch):
        samples, cls, box, indices = zip(*batch)

        cls = torch.cat(cls, 0)
        box = torch.cat(box, 0)

        new_indices = list(indices)
        for i in range(len(indices)):
            new_indices[i] += i
        indices = torch.cat(new_indices, 0)

        targets = {'cls': cls,
                   'box': box,
                   'idx': indices}
        return torch.stack(samples, 0), targets

    @staticmethod
    def load_label(filenames):
        path = os.path.join(os.path.dirname(filenames[0]), '.cache')
        if os.path.exists(path):
            return torch.load(path)
        x = {}
        for filename in filenames:
            try:
                # verify images
                with open(filename, 'rb') as f:
                    image = Image.open(f)
                    image.verify()  # PIL verify
                shape = image.size  # image size
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert image.format.lower() in FORMATS, f'invalid image format {image.format}'

                # verify labels
                label_file = filename.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
                if os.path.isfile(label_file):
                    with open(label_file) as f:
                        label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        label = np.array(label, dtype=np.float32)
                    nl = len(label)
                    if nl:
                        assert (label >= 0).all()
                        assert label.shape[1] == 5
                        assert (label[:, 1:] <= 1).all()
                        _, i = np.unique(label, axis=0, return_index=True)
                        if len(i) < nl:  # duplicate row check
                            label = label[i]  # remove duplicates
                    else:
                        label = np.zeros((0, 5), dtype=np.float32)
                else:
                    label = np.zeros((0, 5), dtype=np.float32)
                if filename:
                    x[filename] = label
            except FileNotFoundError:
                pass
            except AssertionError:
                pass
        torch.save(x, path)
        return x


def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def xy2wh(x, w, h):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def resample():
    choices = (cv2.INTER_AREA,
               cv2.INTER_CUBIC,
               cv2.INTER_LINEAR,
               cv2.INTER_NEAREST,
               cv2.INTER_LANCZOS4)
    return random.choice(seq=choices)


def augment_hsv(image, params):
    # HSV color-space augmentation
    h = params['hsv_h']
    s = params['hsv_s']
    v = params['hsv_v']

    r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype('uint8')
    lut_s = np.clip(x * r[1], 0, 255).astype('uint8')
    lut_v = np.clip(x * r[2], 0, 255).astype('uint8')

hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=image)  # Convert the processed HSV image back to BGR format

# Random Perspective Transformation
def random_perspective(image, targets=(), params=None, border=(0, 0)):
    if targets is None:  # targets = [cls, xyxy]
        targets = []

    height, width, _ = image.shape
    if not isinstance(border, (tuple, list)):
        border = [border] * 4

    # Coordinates of four points before perspective transformation
    tl = np.array([0, 0])
    tr = np.array([width, 0])
    bl = np.array([0, height])
    br = np.array([width, height])

    # Calculate the range of perspective transformation
    border = np.array(border)
    if (border == 0).all():
        border = np.random.uniform(0.25, 0.75)  # Random scaling factor in the range of 0.25-0.75

    # Coordinates of four target points after perspective transformation
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = src + np.random.uniform(-border, border, size=src.shape)  # Add random perturbation

    matrix = cv2.getPerspectiveTransform(src, dst)  # Calculate perspective transformation matrix
    image = cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(128, 128, 128))  # Execute perspective transformation

    # Perspective transformation of labels
    if len(targets):
        n = len(targets)
        points = targets[:, 1:5].reshape(n, -1, 2)
        points = cv2.perspectiveTransform(points, matrix)  # Perspective transformation of label coordinates
        points = points.reshape(n, -1, 4)  # Reshape to match the input
        targets[:, 1:5] = points.reshape(n, -1)

    return image, targets

# MixUp Data Augmentation
def mix_up(image1, label1, image2, label2):
    image = (image1 + image2) / 2  # Calculate the average of two images
    label = np.concatenate((label1, label2), 0)  # Concatenate labels
    return image, label

# Resize Image and Perform Padding
def resize(image, size, augment=False):
    h, w, _ = image.shape
    r = size / max(h, w)  # Calculate the scaling factor
    if augment and (random.randint(0, 1) or r < 1):  # Random scaling or proportional scaling
        new_ar = max(h, w) / min(h, w) * random.uniform(1 - r, 1 + r)  # Random aspect ratio
        scale = r * random.uniform(0.5, 2)  # Random scaling factor
        if h < w:
            nh = int(size * random.uniform(0.5, 2))
            nw = int(nh * new_ar)
        else:
            nw = int(size * random.uniform(0.5, 2))
            nh = int(nw / new_ar)
        image = cv2.resize(image, (nw, nh), interpolation=resample())
    else:
        nw, nh = w, h

    # Create a canvas and fill it with gray color
    shape = (size, size, image.shape[2])
    image_new = np.full(shape, 128, dtype=np.uint8)

    # Paste the original image to the center of the canvas
    dx, dy = (size - nw) // 2, (size - nh) // 2
    image_new[dy:dy + nh, dx:dx + nw] = image
    return image_new, nw / w, nh / h, dx / size, dy / size  # Return the adjusted image, scaling factor, and padding parameters

# Albumentations Class for Data Augmentation
class Albumentations:
    def __init__(self):
        self.transform = A.Compose([
            A.RandomResizedCrop(height=640, width=640, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['cls']))

    def __call__(self, image, boxes, labels):
        transformed = self.transform(image=image, bboxes=boxes, cls=labels)
        transformed_image = transformed['image']
        transformed_boxes = np.array(transformed['bboxes'])
        transformed_labels = np.array(transformed['cls'])
        return transformed_image, transformed_boxes, transformed_labels
