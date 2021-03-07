from torch import nn, Tensor

from .backbone import ResidualBlock, DownSample


class Encoder338(nn.Module):
    def __init__(self, numChannels: int, numGroups: int):
        super(Encoder338, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, numChannels, 9, padding=4),
            nn.GroupNorm(numGroups, numChannels)
        )

        self.stage2 = nn.Sequential(
            DownSample(numChannels, numGroups),
            ResidualBlock(numChannels, numGroups),
            ResidualBlock(numChannels, numGroups)
        )

        self.stage3 = nn.Sequential(
            DownSample(numChannels, numGroups),
            ResidualBlock(numChannels, numGroups),
            ResidualBlock(numChannels, numGroups)
        )

        self.stage4 = nn.Sequential(
            DownSample(numChannels, numGroups),
            ResidualBlock(numChannels, numGroups),
            ResidualBlock(numChannels, numGroups)
        )

        self.stage5 = nn.Sequential(
            DownSample(numChannels, numGroups),
            ResidualBlock(numChannels, numGroups),
            ResidualBlock(numChannels, numGroups)
        )

        self.stage6 = nn.Sequential(
            DownSample(numChannels, numGroups),
            ResidualBlock(numChannels, numGroups),
            ResidualBlock(numChannels, numGroups)
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.stage1(x)
        y = self.stage2(y)
        y = self.stage3(y)
        y = self.stage4(y)
        y = self.stage5(y)
        y = self.stage6(y)

        return y


class Decoder338(nn.Module):
    def __init__(self, numChannels: int, numGroups: int):
        super(Decoder338, self).__init__()

        self.input = nn.Sequential(
            ResidualBlock(64, numGroups),
        )

        self.residualBlocks = nn.Sequential(*[ResidualBlock(numChannels, numGroups) for _ in range(16)])

        self.upSampling = nn.Sequential(
            nn.Conv2d(numChannels, numChannels * 16, 1),
            nn.PixelShuffle(4),
            nn.PReLU(),
            nn.Conv2d(numChannels, numChannels * 16, 1),
            nn.PixelShuffle(4),
            nn.PReLU(),
            nn.Conv2d(numChannels, numChannels * 4, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(numChannels, 3, 9, padding=4),
            nn.Tanh()
        )

    def forward(self, x: Tensor) -> Tensor:
        out1 = self.input(x)
        out2 = self.residualBlocks(out1)
        out = self.upSampling(out1 + out2)

        return (out + 1.) / 2.


class AutoEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(AutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(self.encoder(x))
