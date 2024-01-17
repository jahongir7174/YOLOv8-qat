import copy
import math
import random
from time import time

import numpy as np
import torch
import torchvision
import cv2
from os import environ
from platform import system

def setup_seed(seed=0):
    """
    Setup random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    # Set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # Disable OpenCV multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # Setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # Setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'

def wh2xy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def make_anchors(x, strides, offset=0.5):
    anchors, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=x[i].device, dtype=x[i].dtype) + offset  # shift x
        sy = torch.arange(end=h, device=x[i].device, dtype=x[i].dtype) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchors.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=x[i].dtype, device=x[i].device))
    return torch.cat(anchors), torch.cat(stride_tensor)

def compute_metric(output, target, iou_v):
    # Intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    a1, a2 = target[:, 1:].unsqueeze(1).chunk(2, 2)
    b1, b2 = output[:, :4].unsqueeze(0).chunk(2, 2)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    
    # IoU = intersection / (area1 + area2 - intersection)
    iou = intersection / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - intersection + 1e-7)

    correct = np.zeros((output.shape[0], iou_v.shape[0]), dtype=bool)
    
    for i in range(len(iou_v)):
        # IoU > threshold and classes match
        x = torch.where((iou >= iou_v[i]) & (target[:, 0:1] == output[:, 5]))
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=output.device)

def non_max_suppression(outputs, conf_threshold, iou_threshold, nc):
    max_wh = 7680
    max_det = 300
    max_nms = 30000

    shape = outputs[0].shape[0]
    stride = torch.tensor([8.0, 16.0, 32.0])

    anchors, strides = (x.transpose(0, 1) for x in make_anchors(outputs, stride, 0.5))

    box, cls = torch.cat([i.view(shape, nc + 4, -1) for i in outputs], dim=2).split((4, nc), 1)
    a, b = box.chunk(2, 1)
    a = anchors.unsqueeze(0) - a
    b = anchors.unsqueeze(0) + b
    box = torch.cat(((a + b) / 2, b - a), dim=1)
    outputs = torch.cat((box * strides, cls.sigmoid()), dim=1)

    bs = outputs.shape[0]  # batch size
    nc = outputs.shape[1] - 4  # number of classes
    xc = outputs[:, 4:4 + nc].amax(1) > conf_threshold  # candidates

    # Settings
    start_time = time()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    nms_outputs = [torch.zeros((0, 6), device=outputs.device)] * bs
    for index, output in enumerate(outputs):  # image index, image inference
        output = output.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not output.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls = output.split((4, nc), 1)
        box = wh2xy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            output = torch.cat((box[i], output[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            output = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]

        # Check shape
        n = output.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_det:  # excess boxes
            output = output[output[:, 4].argsort(descending=True)[:max_det]]  # sort by confidence

        # Batched NMS
        c = output[:, 5:6] * max_wh  # classes
        output = torch.cat((box * max_wh, output[:, :4], c), 1)  # to float
        output = non_max_suppression_cpu(output, conf_threshold, iou_threshold, offset=0)

        # Save
        nms_outputs[index] = output[:max_nms] if output.shape[0] > max_nms else output

        # Print time and images
        print(f'{index}/{bs}, {output.shape[0]} detections: {output[:, 5].tolist()}')
        if (time() - start_time) > time_limit:
            break  # time limit exceeded
    return nms_outputs

def compute_ap(predictions, targets):
    """
    Compute Average Precision (AP) for a given set of predictions and targets.

    Args:
        predictions (list): List of predicted bounding boxes and scores.
            Each element is a tuple (confidence, x1, y1, x2, y2).
        targets (list): List of ground truth bounding boxes.
            Each element is a tuple (class, x1, y1, x2, y2).

    Returns:
        float: Average Precision (AP) value.
    """
    # Sort predictions by confidence score in descending order
    predictions.sort(key=lambda x: x[0], reverse=True)

    true_positives = []
    false_positives = []
    num_targets = len(targets)

    for prediction in predictions:
        confidence, pred_x1, pred_y1, pred_x2, pred_y2 = prediction
        best_iou = 0
        best_target = -1

        for i, target in enumerate(targets):
            target_class, target_x1, target_y1, target_x2, target_y2 = target
            if target_class == 0:  # Consider only objects of class 0 (change as needed)
                iou = calculate_iou((pred_x1, pred_y1, pred_x2, pred_y2), (target_x1, target_y1, target_x2, target_y2))
                if iou > best_iou:
                    best_iou = iou
                    best_target = i

        if best_iou >= 0.5:
            if targets[best_target][0] != -1:
                true_positives.append(1)
                targets[best_target] = (-1, 0, 0, 0, 0)  # Mark target as used
            else:
                false_positives.append(1)
        else:
            false_positives.append(1)

    num_predictions = len(predictions)
    precision = sum(true_positives) / (sum(true_positives) + sum(false_positives))
    recall = sum(true_positives) / num_targets if num_targets > 0 else 0

    return compute_ap_with_precision_recall(precision, recall)

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # Calculate the area of intersection
    x_intersection = max(0, min(x2, x4) - max(x1, x3))
    y_intersection = max(0, min(y2, y4) - max(y1, y3))
    intersection_area = x_intersection * y_intersection

    # Calculate the area of each bounding box
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x4 - x3) * (y4 - y3)

    # Calculate the IoU (Intersection over Union)
    iou = intersection_area / (area_box1 + area_box2 - intersection_area + 1e-6)

    return iou

def compute_ap_with_precision_recall(precision, recall):
    """
    Compute Average Precision (AP) from precision and recall values using the
    VOC 2010 method. This method calculates the AP as the area under the precision-recall curve.

    Args:
        precision (list): List of precision values.
        recall (list): List of recall values.

    Returns:
        float: Average Precision (AP) value.
    """
    m_rec = [0] + recall + [1]
    m_pre = [0] + precision + [0]

    # Compute the precision envelope
    for i in range(len(m_pre) - 2, -1, -1):
        m_pre[i] = max(m_pre[i], m_pre[i + 1])

    # Integrate area under the curve
    ap = 0
    for i in range(1, len(m_rec)):
        ap += (m_rec[i] - m_rec[i - 1]) * m_pre[i]

    return ap
