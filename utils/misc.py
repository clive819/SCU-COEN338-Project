import os
from argparse import ArgumentParser
from collections import defaultdict
from time import time
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


def baseParser() -> ArgumentParser:
    parser = ArgumentParser('COEN 338', add_help=False)

    # MARK: - model config
    parser.add_argument('--numGroups', default=8, type=int)
    parser.add_argument('--numChannels', default=64, type=int)

    # MARK: - dataset
    parser.add_argument('--dataDir', default='./dataset/train', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--imageType', default='png', type=str)

    # MARK: - miscellaneous
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--encoderWeight', default='', type=str)
    parser.add_argument('--decoderWeight', default='', type=str)
    parser.add_argument('--discriminatorWeight', default='', type=str)
    parser.add_argument('--seed', default=1234, type=int)

    return parser


class MetricsLogger(object):
    def __init__(self, folder: str = './logs'):
        self.writer = SummaryWriter(folder)
        self.cache = defaultdict(list)
        self.lastStep = time()

    def addScalar(self, tag: str, value: Any, step: int = None, wallTime: float = None):
        self.writer.add_scalar(tag, value, step, wallTime)

    def close(self):
        self.writer.close()

    def flush(self):
        self.writer.flush()

    def step(self, metrics: dict):
        elapse = time() - self.lastStep
        print(f'Elapse: {elapse: .4f}s, {1 / elapse: .2f} steps/sec')
        self.lastStep = time()

        for key in metrics:
            self.cache[key].append(metrics[key].cpu().item())

    def epochEnd(self, epoch: int):
        losses = []
        for key in self.cache:
            avg = np.mean(self.cache[key])
            if 'loss' in key:
                losses.append(avg)
            self.writer.add_scalar(f'Average/{key}', avg, epoch)

        avg = np.mean(losses)
        self.writer.add_scalar('Average/loss', avg, epoch)
        self.cache.clear()


def logMetrics(metrics: Dict[str, Tensor]):
    log = '[ '
    log += ' ] [ '.join([f'{k} = {v.cpu().item():.4f}' for k, v in metrics.items()])
    log += ' ]'
    print(log)


def calcFileSize(path: str) -> Tuple[int, float]:
    byte = os.stat(path).st_size

    return byte // 1000, byte / 1e6
