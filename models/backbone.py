from torch import nn, Tensor


class ResidualBlock(nn.Module):
    def __init__(self, inChannels: int, numGroups: int):
        super(ResidualBlock, self).__init__()

        self.module = nn.Sequential(
            nn.Conv2d(inChannels, inChannels, 1),
            nn.GroupNorm(numGroups, inChannels),
            nn.PReLU(),
            nn.Conv2d(inChannels, inChannels, 3, padding=1),
            nn.GroupNorm(numGroups, inChannels)
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.module(x)
        return x + out


class DownSample(nn.Module):
    def __init__(self, numChannel: int, numGroups: int):
        super(DownSample, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(numChannel, numChannel, 1, stride=2),
            nn.GroupNorm(numGroups, numChannel)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(numChannel, numChannel, 3, stride=2, padding=1),
            nn.GroupNorm(numGroups, numChannel)
        )

        self.activation = nn.PReLU()

    def forward(self, x: Tensor) -> Tensor:
        out1 = self.conv1(x)
        out2 = self.conv3(x)

        return self.activation(out1 + out2)
