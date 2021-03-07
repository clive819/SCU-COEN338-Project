import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torchvision
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader

from models import Encoder338, Decoder338, Discriminator, AutoEncoder, TVLoss
from utils import COEN338Dataset, baseParser, MetricsLogger, logMetrics, MSSSIM


def main(args):
    print(args)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if not os.path.exists(args.outputDir):
        os.mkdir(args.outputDir)

    # load data
    dataset = COEN338Dataset(args.dataDir, args.imageType, args.imageWidth, args.imageHeight, args.mode)
    dataLoader = DataLoader(dataset, batch_size=args.batchSize, shuffle=True, pin_memory=True,
                            num_workers=args.numWorkers)

    # load model
    encoder = Encoder338(args.numChannels, args.numGroups)
    decoder = Decoder338(args.numChannels, args.numGroups)
    autoEncoder = AutoEncoder(encoder, decoder).to(device)
    discriminator = Discriminator(args.numChannels, args.numGroups).to(device)
    featureExtractor = torchvision.models.vgg19(True).features.eval().to(device)

    # resume training
    if args.encoderWeight and os.path.exists(args.encoderWeight):
        print(f'loading pre-trained encoder weights from {args.encoderWeight}')
        encoder.load_state_dict(torch.load(args.encoderWeight, map_location=device))

    if args.decoderWeight and os.path.exists(args.decoderWeight):
        print(f'loading pre-trained decoder weights from {args.decoderWeight}')
        decoder.load_state_dict(torch.load(args.decoderWeight, map_location=device))

    if args.discriminatorWeight and os.path.exists(args.discriminatorWeight):
        print(f'loading pre-trained discriminator weights from {args.discriminatorWeight}')
        discriminator.load_state_dict(torch.load(args.discriminatorWeight, map_location=device))

    optimizerG = AdamW(autoEncoder.parameters(), lr=args.lr, weight_decay=args.weightDecay)
    optimizerD = AdamW(discriminator.parameters(), lr=args.lr, weight_decay=args.weightDecay)

    tv = TVLoss().to(device)
    msssim = MSSSIM().to(device)

    assert args.imageWidth == args.imageHeight, 'image height and width should be the same'
    # Based on Eq.14 of https://arxiv.org/pdf/1511.08861.pdf
    gaussianWindow = MSSSIM(args.imageWidth, 0.5).window.view(1, 3, args.imageWidth, args.imageWidth).to(device)
    alpha = 0.84

    batches = len(dataLoader)
    logger = MetricsLogger()

    for epoch in range(args.epochs):
        losses = []
        for batch, img in enumerate(dataLoader):
            original = img.to(device)

            # MARK: - update discriminator
            optimizerD.zero_grad()

            reconstructed = autoEncoder(original)

            realD = discriminator(original)
            fakeD = discriminator(reconstructed)

            lossD = 1. - realD.mean() + fakeD.mean()
            lossD.backward()

            optimizerD.step()

            # MARK: - update generator
            optimizerG.zero_grad()

            reconstructed = autoEncoder(original)

            fakeD = discriminator(reconstructed)

            oriFeatures = featureExtractor(original)
            genFeatures = featureExtractor(reconstructed)

            adversarialLoss = torch.mean(1. - fakeD)
            contentLoss = F.l1_loss(genFeatures, oriFeatures)
            imageLoss = alpha * (1. - msssim(reconstructed, original)) + \
                        (1 - alpha) * torch.mean(gaussianWindow * torch.abs(original - reconstructed))
            tvLoss = tv(reconstructed)

            lossG = imageLoss + 1e-3 * adversarialLoss + 6e-3 * contentLoss + 2e-8 * tvLoss
            lossG.backward()

            optimizerG.step()

            # MARK: - print & save training details
            losses.append(lossG + lossD)
            metrics = {
                'lossD': lossD,
                'lossG': lossG
            }

            print(f'Epoch {epoch} | {batch + 1} / {batches}')
            logMetrics(metrics)
            logger.step(metrics)

        logger.epochEnd(epoch)
        avgLoss = torch.mean(torch.stack(losses)).cpu()
        print(f'Epoch {epoch}, loss: {avgLoss:.8f}')

        if not os.path.exists(args.outputDir):
            os.mkdir(args.outputDir)

        torch.save(encoder.state_dict(), os.path.join(args.outputDir, 'encoder.pt'))
        torch.save(decoder.state_dict(), os.path.join(args.outputDir, 'decoder.pt'))
        torch.save(discriminator.state_dict(), os.path.join(args.outputDir, 'discriminator.pt'))

        logger.addScalar('Model', avgLoss, epoch)
        logger.flush()
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser('python3 train.py', parents=[baseParser()])

    # MARK: - training config
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batchSize', default=4, type=int)
    parser.add_argument('--weightDecay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=30000, type=int)
    parser.add_argument('--imageHeight', default=416, type=int)
    parser.add_argument('--imageWidth', default=416, type=int)

    # MARK: - miscellaneous
    parser.add_argument('--outputDir', default='./checkpoint', type=str)
    parser.add_argument('--numWorkers', default=4, type=int)

    main(parser.parse_args())
