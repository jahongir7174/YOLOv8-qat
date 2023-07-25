import copy
import math
import random
from time import time

import numpy
import torch
import torchvision


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def wh2xy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
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
    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2) = target[:, 1:].unsqueeze(1).chunk(2, 2)
    (b1, b2) = output[:, :4].unsqueeze(0).chunk(2, 2)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    # IoU = intersection / (area1 + area2 - intersection)
    iou = intersection / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - intersection + 1e-7)

    correct = numpy.zeros((output.shape[0], iou_v.shape[0]))
    correct = correct.astype(bool)
    for i in range(len(iou_v)):
        # IoU > threshold and classes match
        x = torch.where((iou >= iou_v[i]) & (target[:, 0:1] == output[:, 5]))
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1),
                                 iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
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
        # sort by confidence and remove excess boxes
        output = output[output[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = output[:, 5:6] * max_wh  # classes
        boxes, scores = output[:, :4] + c, output[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        i = i[:max_det]  # limit detections

        nms_outputs[index] = output[i]
        if (time() - start_time) > time_limit:
            break  # time limit exceeded

    return nms_outputs


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = numpy.ones(nf // 2)  # ones padding
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode='valid')  # y-smoothed


def compute_ap(tp, conf, pred_cls, target_cls, eps=1E-16):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Object-ness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
    # Returns
        The average precision
    """
    # Sort by object-ness
    i = numpy.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = numpy.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    p = numpy.zeros((nc, 1000))
    r = numpy.zeros((nc, 1000))
    ap = numpy.zeros((nc, tp.shape[1]))
    px, py = numpy.linspace(0, 1, 1000), []  # for plotting
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of outputs
        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = numpy.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = numpy.concatenate(([1.0], precision[:, j], [0.0]))

            # Compute the precision envelope
            m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))

            # Integrate area under curve
            x = numpy.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap[ci, j] = numpy.trapz(numpy.interp(x, m_rec, m_pre), x)  # integrate

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap = ap50.mean(), ap.mean()
    return tp, fp, m_pre, m_rec, map50, mean_ap


def compute_iou(box1, box2, eps=1E-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
    # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)  # CIoU


def strip_optimizer(filename):
    x = torch.load(filename, map_location=torch.device('cpu'))
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, filename)


def clip_gradients(model, max_norm=10.0):
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


def load_weight(ckpt, model):
    dst = model.state_dict()
    src = torch.load(ckpt, 'cpu')['model'].float().state_dict()
    ckpt = {}
    for k, v in src.items():
        if k in dst and v.shape == dst[k].shape:
            ckpt[k] = v
    model.load_state_dict(state_dict=ckpt, strict=False)
    return model


def weight_decay(model, decay):
    p1 = []
    p2 = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            p1.append(param)
        else:
            p2.append(param)
    return [{'params': p1, 'weight_decay': 0.00},
            {'params': p2, 'weight_decay': decay}]


def export_onnx(model, args, filename):
    model.eval()
    import onnx  # noqa

    inputs = ['images']
    outputs = ['outputs']
    dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'},
               'outputs': {0: 'batch', 2: 'anchors'}}

    x = torch.zeros((1, 3, args.input_size, args.input_size))

    torch.onnx.export(model.cpu(), x.cpu(), filename,
                      verbose=False,
                      opset_version=13,
                      dynamic_axes=dynamic,
                      # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
                      do_constant_folding=True,
                      input_names=inputs,
                      output_names=outputs)


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num


class YOLODetector:
    def __init__(self, onnx_path=None, session=None):
        self.session = session
        from onnxruntime import InferenceSession

        if self.session is None:
            assert onnx_path is not None
            self.session = InferenceSession(onnx_path,
                                            providers=['CPUExecutionProvider'])

        self.output_names = []
        for output in self.session.get_outputs():
            self.output_names.append(output.name)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x):

        return self.session.run(self.output_names, {self.input_name: x})


class Assigner:
    """
    A task-aligned assigner for object detection
    """

    def __init__(self, top_k=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.top_k = top_k
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def __call__(self, pred_scores, pred_boxes, anchors, true_labels, true_boxes, mask_gt):
        size = pred_scores.size(0)
        num_max_boxes = true_boxes.size(1)
        if num_max_boxes == 0:
            device = true_boxes.device
            return (torch.full_like(pred_scores[..., 0], self.bg_idx).to(device),
                    torch.zeros_like(pred_boxes).to(device),
                    torch.zeros_like(pred_scores).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device))

        num_anchors = anchors.shape[0]
        lt, rb = true_boxes.view(-1, 1, 4).chunk(2, 2)
        bbox_deltas = torch.cat((anchors[None] - lt, rb - anchors[None]), dim=2)
        bbox_deltas = bbox_deltas.view(true_boxes.shape[0], true_boxes.shape[1], num_anchors, -1)
        mask_in_gts = bbox_deltas.amin(3).gt_(1E-9)
        na = pred_boxes.shape[-2]
        mask_true = (mask_in_gts * mask_gt).bool()
        overlaps = torch.zeros([size, num_max_boxes, na], dtype=pred_boxes.dtype, device=pred_boxes.device)
        bbox_scores = torch.zeros([size, num_max_boxes, na], dtype=pred_scores.dtype, device=pred_scores.device)
        ind = torch.zeros([2, size, num_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=size).view(-1, 1).expand(-1, num_max_boxes)
        ind[1] = true_labels.squeeze(-1)
        bbox_scores[mask_true] = pred_scores[ind[0], :, ind[1]][mask_true]

        pd_boxes = pred_boxes.unsqueeze(1).expand(-1, num_max_boxes, -1, -1)[mask_true]
        gt_boxes = true_boxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_true]
        overlaps[mask_true] = compute_iou(gt_boxes, pd_boxes).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        top_k_metrics, top_k_indices = torch.topk(align_metric, self.top_k, dim=-1, largest=True)

        top_k_mask = mask_gt.expand(-1, -1, self.top_k).bool()
        top_k_indices.masked_fill_(~top_k_mask, 0)

        mask = torch.zeros(align_metric.shape, dtype=torch.int8, device=top_k_indices.device)
        ones = torch.ones_like(top_k_indices[:, :, :1], dtype=torch.int8, device=top_k_indices.device)
        for k in range(self.top_k):
            mask.scatter_add_(-1, top_k_indices[:, :, k:k + 1], ones)
        mask.masked_fill_(mask > 1, 0)

        mask_top_k = mask.to(align_metric.dtype)
        mask_pos = mask_top_k * mask_in_gts * mask_gt

        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, num_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)

        indices = torch.arange(end=size, dtype=torch.int64, device=true_labels.device)[..., None]
        target_index = target_gt_idx + indices * num_max_boxes
        target_labels = true_labels.long().flatten()[target_index]

        target_bboxes = true_boxes.view(-1, 4)[target_index]

        target_labels.clamp_(0)

        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.num_classes),
                                    dtype=torch.int64,
                                    device=target_labels.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_bboxes, target_scores, fg_mask.bool(), target_gt_idx


class BoxLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred_bboxes, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = compute_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])

        return ((1.0 - iou) * weight).sum() / target_scores_sum


class ComputeLoss:
    def __init__(self, model, params):
        super().__init__()
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device

        self.no = model.no
        self.nc = model.nc
        self.params = params
        self.device = device
        self.stride = model.stride

        self.box_loss = BoxLoss().to(device)
        self.cls_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.assigner = Assigner(top_k=10, num_classes=self.nc, alpha=0.5, beta=6.0)

    def __call__(self, outputs, targets):
        shape = outputs[0].shape
        x_cat = torch.cat([i.view(shape[0], self.no, -1) for i in outputs], 2)
        pred_distri, pred_scores = torch.split(x_cat, split_size_or_sections=(4, self.nc), dim=1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        size = torch.tensor(shape[2:], device=self.device, dtype=pred_scores.dtype)
        size = size * self.stride[0]
        anchors, strides = make_anchors(outputs, self.stride, 0.5)

        # targets
        indices = targets['idx'].view(-1, 1)
        batch_size = pred_scores.shape[0]
        box_targets = torch.cat((indices, targets['cls'].view(-1, 1), targets['box']), 1)
        box_targets = box_targets.to(self.device)
        if box_targets.shape[0] == 0:
            true = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = box_targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            true = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    true[j, :n] = box_targets[matches, 1:]
            x = true[..., 1:5].mul_(size[[1, 0, 1, 0]])
            y = x.clone()
            y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
            y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
            y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
            y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
            true[..., 1:5] = y
        gt_labels, gt_bboxes = true.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.box_decode(anchors, pred_distri)

        scores = pred_scores.detach().sigmoid()
        bboxes = (pred_bboxes.detach() * strides).type(gt_bboxes.dtype)
        target_bboxes, target_scores, fg_mask, _ = self.assigner(scores, bboxes,
                                                                 anchors * strides,
                                                                 gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        loss_cls = self.cls_loss(pred_scores, target_scores.to(pred_scores.dtype)).sum()
        loss_cls = loss_cls / target_scores_sum

        # box loss
        loss_box = torch.zeros(1, device=self.device)
        if fg_mask.sum():
            target_bboxes /= strides
            loss_box = self.box_loss(pred_bboxes,
                                     target_bboxes,
                                     target_scores,
                                     target_scores_sum, fg_mask)

        loss_box *= self.params['box']  # box gain
        loss_cls *= self.params['cls']  # cls gain

        return loss_box, loss_cls

    @staticmethod
    def box_decode(anchor_points, pred_dist):
        a, b = pred_dist.chunk(2, -1)
        a = anchor_points - a
        b = anchor_points + b
        return torch.cat((a, b), -1)
