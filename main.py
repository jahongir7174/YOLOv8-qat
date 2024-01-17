import os
import warnings
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from nets import nn
from utils import util
from utils.dataset import CustomDataset
from utils.trainer import Trainer
from utils.tester import Tester

warnings.filterwarnings("ignore")


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    config_path = 'utils/args.yaml'
    params = util.load_config(config_path)

    if args.train:
        trainer = Trainer(args, params)
        best_mean_ap = trainer.train()
        print(f'Best mAP: {best_mean_ap:.3f}')

    if args.test:
        tester = Tester(args, params)
        tester.test()


if __name__ == "__main__":
    main()
