from itertools import accumulate
import torch
import numpy as np
import csv
import copy
import os
import tqdm
import warnings
import yaml
from argparse import ArgumentParser
from torch.utils import data
from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, args, params):
        self.args = args
        self.params = params
        self.setup_training_environment()

    def setup_training_environment(self):
        util.setup_seed()
        util.setup_multi_processes()

        self.model = self.load_and_prepare_model()
        self.optimizer, self.scheduler = self.configure_optimizer_and_scheduler()
        self.loader = self.prepare_data_loader()

    def load_and_prepare_model(self):
        model = nn.yolo_v8_n(len(self.params['names']))
        state = torch.load('./weights/v8_n.pth')['model']
        model.load_state_dict(state.float().state_dict())
        model.eval()

        for m in model.modules():
            if type(m) is nn.Conv and hasattr(m, 'norm'):
                torch.ao.quantization.fuse_modules(m, [["conv", "norm"]], True)
        model.train()

        model = nn.QAT(model)
        model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
        torch.quantization.prepare_qat(model, inplace=True)
        model.cuda()

        return model

    def configure_optimizer_and_scheduler(self):
        accumulate = max(round(64 / (self.args.batch_size * self.args.world_size)), 1)
        self.params['weight_decay'] *= self.args.batch_size * self.args.world_size * accumulate / 64

        optimizer = torch.optim.SGD(util.weight_decay(self.model, self.params['weight_decay']),
                                    self.params['lr0'], self.params['momentum'], nesterov=True)

        lr = self.learning_rate()
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr, last_epoch=-1)

        return optimizer, scheduler

    def prepare_data_loader(self):
        filenames = []
        with open('../Dataset/COCO/train2017.txt') as reader:
            for filename in reader.readlines():
                filename = filename.rstrip().split('/')[-1]
                filenames.append('../Dataset/COCO/images/train2017/' + filename)

        sampler = None
        dataset = Dataset(filenames, self.args.input_size, self.params, True)

        if self.args.distributed:
            sampler = data.distributed.DistributedSampler(dataset)

        loader = data.DataLoader(dataset, self.args.batch_size, sampler is None, sampler,
                                 num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)

        return loader

    def learning_rate(self):
        def fn(x):
            return (1 - x / self.args.epochs) * (1.0 - self.params['lrf']) + self.params['lrf']

        return fn

    def train(self):
        best_mean_ap = 0
        num_steps = len(self.loader)
        criterion = util.ComputeLoss(self.model, self.params)
        num_warmup = max(round(self.params['warmup_epochs'] * num_steps), 100)

        with open('weights/step.csv', 'w') as f:
            if self.args.local_rank == 0:
                writer = csv.DictWriter(f, fieldnames=['epoch', 'box', 'cls', 'Recall', 'Precision', 'mAP@50', 'mAP'])
                writer.writeheader()

            for epoch in range(self.args.epochs):
                self.model.train()

                if self.args.distributed:
                    self.sampler.set_epoch(epoch)

                if self.args.epochs - epoch == 10:
                    self.loader.dataset.mosaic = False

                p_bar = enumerate(self.loader)

                if self.args.local_rank == 0:
                    print(('\n' + '%10s' * 4) % ('epoch', 'memory', 'box', 'cls'))

                if self.args.local_rank == 0:
                    p_bar = tqdm.tqdm(p_bar, total=num_steps)  # progress bar

                self.optimizer.zero_grad()
                avg_box_loss = util.AverageMeter()
                avg_cls_loss = util.AverageMeter()

                for i, (samples, targets) in p_bar:
                    samples = samples.cuda()
                    samples = samples.float()
                    samples = samples / 255.0
                    x = i + num_steps * epoch

                    # Warmup
                    if x <= num_warmup:
                        self._warmup_lr_and_momentum(x, num_warmup)

                    # Forward and Backward
                    loss_box, loss_cls = self._forward_and_backward(samples, targets, criterion)
                    avg_box_loss.update(loss_box.item(), samples.size(0))
                    avg_cls_loss.update(loss_cls.item(), samples.size(0))

                    # Optimize
                    self._optimize(accumulate)

                    # Log
                    self._log_progress(epoch, avg_box_loss, avg_cls_loss, p_bar)

                # Scheduler
                self.scheduler.step()

                if self.args.local_rank == 0:
                    # Convert model
                    last = self._convert_and_test()

                    # Write to CSV
                    self._write_to_csv(epoch, avg_box_loss, avg_cls_loss, last, writer, f)

                    # Update best mAP
                    best_mean_ap = self._update_best_map(best_mean_ap, last)

                    # Save last, best and delete
                    self._save_models(best_mean_ap)

        torch.cuda.empty_cache()
        return best_mean_ap

    def _warmup_lr_and_momentum(self, x, num_warmup, epoch):
        # Warmup logic
        xp = [0, num_warmup]
        fp = [1, 64 / (self.args.batch_size * self.args.world_size)]
        accumulate = max(1, np.interp(x, xp, fp).round())
        for j, y in enumerate(self.optimizer.param_groups):
            if j == 0:
                fp = [self.params['warmup_bias_lr'], y['initial_lr'] * self.learning_rate()(epoch)]
            else:
                fp = [0.0, y['initial_lr'] * self.learning_rate()(epoch)]
            y['lr'] = np.interp(x, xp, fp)
            if 'momentum' in y:
                fp = [self.params['warmup_momentum'], self.params['momentum']]
                y['momentum'] = np.interp(x, xp, fp)

    def _forward_and_backward(self, samples, targets, criterion):
        outputs = self.model(samples)
        loss_box, loss_cls = criterion(outputs, targets)
        avg_box_loss = util.AverageMeter()
        avg_cls_loss = util.AverageMeter()
        loss_box *= self.args.batch_size  # loss scaled by batch_size
        loss_cls *= self.args.batch_size  # loss scaled by batch_size
        loss_box *= self.args.world_size  # gradient averaged between devices in DDP mode
        loss_cls *= self.args.world_size  # gradient averaged between devices in DDP mode
        (loss_box + loss_cls).backward()
        return loss_box, loss_cls

    def _optimize(self, accumulate, x):  # 传递 x 参数
        if x % accumulate == 0:
            util.clip_gradients(self.model)  # clip gradients
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _log_progress(self, epoch, avg_box_loss, avg_cls_loss, p_bar):
        if self.args.local_rank == 0:
            memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}'  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 2) % (f'{epoch + 1}/{self.args.epochs}', memory,
                                               avg_box_loss.avg, avg_cls_loss.avg)
            p_bar.set_description(s)

    def _convert_and_test(self):
        save = copy.deepcopy(self.model.module if self.args.distributed else self.model)
        save.eval()
        save.to(torch.device('cpu'))
        torch.ao.quantization.convert(save, inplace=True)
        last = self.test()
        return last

    def _write_to_csv(self, epoch, avg_box_loss, avg_cls_loss, last, writer, f):
        writer.writerow({'epoch': str(epoch + 1).zfill(3),
                         'box': str(f'{avg_box_loss.avg:.3f}'),
                         'cls': str(f'{avg_cls_loss.avg:.3f}'),
                         'mAP': str(f'{last[0]:.3f}'),
                         'mAP@50': str(f'{last[1]:.3f}'),
                         'Recall': str(f'{last[2]:.3f}'),
                         'Precision': str(f'{last[2]:.3f}')})
        f.flush()

    def _update_best_map(self, best_mean_ap, last):
        if last[0] > best_mean_ap:
            best_mean_ap = last[0]
        return best_mean_ap

if __name__ == "__main__":
    parser = ArgumentParser(description="YOLOv8 Training Script")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    trainer = Trainer(args, config)
    best_mean_ap = trainer.train()
    print(f"Best mAP: {best_mean_ap}")
