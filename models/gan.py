from torch import nn, Tensor

from .backbone import DownSample, ResidualBlock


class Discriminator(nn.Module):
    def __init__(self, numChannels: int, numGroups: int):
        super(Discriminator, self).__init__()

        self.module = nn.Sequential(
            nn.Conv2d(3, numChannels, 9, padding=4),
            nn.GroupNorm(numGroups, numChannels),
            DownSample(numChannels, numGroups),
            ResidualBlock(numChannels, numGroups),
            DownSample(numChannels, numGroups),
            ResidualBlock(numChannels, numGroups),
            DownSample(numChannels, numGroups),
            ResidualBlock(numChannels, numGroups),
            DownSample(numChannels, numGroups),
            ResidualBlock(numChannels, numGroups),
            DownSample(numChannels, numGroups),
            ResidualBlock(numChannels, numGroups),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(numChannels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)
