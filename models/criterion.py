import torch
from torch import nn, Tensor


class TVLoss(nn.Module):
    def __init__(self, tvLossWeight: int = 1):
        super(TVLoss, self).__init__()
        self.tvLossWeight = tvLossWeight

    def forward(self, x: Tensor) -> Tensor:
        batchSize = x.size()[0]
        h = x.size()[2]
        w = x.size()[3]
        countH = self.tensorSize(x[:, :, 1:, :])
        countW = self.tensorSize(x[:, :, :, 1:])
        tvH = torch.pow((x[:, :, 1:, :] - x[:, :, :h - 1, :]), 2).sum()
        tvW = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w - 1]), 2).sum()
        return self.tvLossWeight * 2 * (tvH / countH + tvW / countW) / batchSize

    @staticmethod
    def tensorSize(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
