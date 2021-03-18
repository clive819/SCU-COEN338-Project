import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    @torch.no_grad()
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        mse = torch.mean((y - x) ** 2)

        return 10 * torch.log10(1 / mse)


class SSIM(nn.Module):
    def __init__(self, windowSize: int = 11, sigma: float = 1.5):
        super(SSIM, self).__init__()

        self.window = self.createWindow(windowSize, sigma)
        self.windowSize = windowSize

    @staticmethod
    def gaussian(windowSize: int, sigma: float) -> Tensor:
        ans = Tensor([math.exp(-(x - windowSize // 2) ** 2 / float(2 * sigma ** 2)) for x in range(windowSize)])

        return ans / ans.sum()

    def createWindow(self, windowSize: int, sigma: float) -> Tensor:
        oneDimensionWindow = self.gaussian(windowSize, sigma).unsqueeze(1)
        twoDimensionWindow = oneDimensionWindow.mm(oneDimensionWindow.t()).unsqueeze(0).unsqueeze(0)

        return twoDimensionWindow.expand(3, 1, windowSize, windowSize).contiguous()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        assert x.size()[1] == y.size()[1] == 3, 'expect inputs to be 3 channel image tensor wrapped in batch'

        channel = 3
        window = self.window.type_as(x)

        mu1 = F.conv2d(x, window, padding=self.windowSize // 2, groups=channel)
        mu2 = F.conv2d(y, window, padding=self.windowSize // 2, groups=channel)

        mu1Sq = mu1 ** 2
        mu2Sq = mu2 ** 2
        mu1Mu2 = mu1 * mu2

        sigma1Sq = F.conv2d(x * x, window, padding=self.windowSize // 2, groups=channel) - mu1Sq
        sigma2Sq = F.conv2d(y * y, window, padding=self.windowSize // 2, groups=channel) - mu2Sq
        sigma12 = F.conv2d(x * y, window, padding=self.windowSize // 2, groups=channel) - mu1Mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim = ((2 * mu1Mu2 + c1) * (2 * sigma12 + c2)) / ((mu1Sq + mu2Sq + c1) * (sigma1Sq + sigma2Sq + c2))
        return torch.mean(ssim)


class MSSSIM(nn.Module):
    def __init__(self, windowSize: int = 11, sigma: float = 1.5):
        super(MSSSIM, self).__init__()

        self.window = self.createWindow(windowSize, sigma)
        self.windowSize = windowSize
        self.weights = Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    @staticmethod
    def gaussian(windowSize: int, sigma: float) -> Tensor:
        ans = Tensor([math.exp(-(x - windowSize // 2) ** 2 / float(2 * sigma ** 2)) for x in range(windowSize)])

        return ans / ans.sum()

    def createWindow(self, windowSize: int, sigma: float) -> Tensor:
        oneDimensionWindow = self.gaussian(windowSize, sigma).unsqueeze(1)
        twoDimensionWindow = oneDimensionWindow.mm(oneDimensionWindow.t()).unsqueeze(0).unsqueeze(0)

        return twoDimensionWindow.expand(3, 1, windowSize, windowSize).contiguous()

    def ssim(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        channel = 3
        window = self.window.type_as(x)

        mu1 = F.conv2d(x, window, padding=self.windowSize // 2, groups=channel)
        mu2 = F.conv2d(y, window, padding=self.windowSize // 2, groups=channel)

        mu1Sq = mu1 ** 2
        mu2Sq = mu2 ** 2
        mu1Mu2 = mu1 * mu2

        sigma1Sq = F.conv2d(x * x, window, padding=self.windowSize // 2, groups=channel) - mu1Sq
        sigma2Sq = F.conv2d(y * y, window, padding=self.windowSize // 2, groups=channel) - mu2Sq
        sigma12 = F.conv2d(x * y, window, padding=self.windowSize // 2, groups=channel) - mu1Mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        v1 = 2 * sigma12 + c2
        v2 = sigma1Sq + sigma2Sq + c2
        cs = torch.mean(v1 / v2)

        ssim = ((2 * mu1Mu2 + c1) * v1) / ((mu1Sq + mu2Sq + c1) * v2)
        return torch.mean(ssim), cs

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        assert x.size()[1] == y.size()[1] == 3, 'expect inputs to be 3 channel image tensor wrapped in batch'

        ssims = []
        mcs = []

        for _ in range(5):
            sim, cs = self.ssim(x, y)

            ssims.append(sim)
            mcs.append(cs)

            x = F.avg_pool2d(x, (2, 2))
            y = F.avg_pool2d(y, (2, 2))

        ssims = torch.stack(ssims)
        mcs = torch.stack(mcs)

        weights = self.weights.type_as(x)

        pow1 = mcs ** weights
        pow2 = ssims ** weights

        return torch.prod(pow1[:-1] * pow2[-1])
