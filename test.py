import os
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

from models import Encoder338, Decoder338
from utils import COEN338Dataset, baseParser, PSNR, SSIM, calcFileSize


def main(args):
    print(args)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if not os.path.exists(args.outputDir):
        os.mkdir(args.outputDir)

    # load data
    dataset = COEN338Dataset(args.dataDir, args.imageType, mode='test')
    dataLoader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=args.numWorkers)

    # load model
    encoder = Encoder338(args.numChannels, args.numGroups).to(device)
    decoder = Decoder338(args.numChannels, args.numGroups).to(device)

    # resume training
    if args.encoderWeight and os.path.exists(args.encoderWeight):
        print(f'loading pre-trained encoder weights from {args.encoderWeight}')
        encoder.load_state_dict(torch.load(args.encoderWeight, map_location=device))

    if args.decoderWeight and os.path.exists(args.decoderWeight):
        print(f'loading pre-trained decoder weights from {args.decoderWeight}')
        decoder.load_state_dict(torch.load(args.decoderWeight, map_location=device))

    # evaluation
    encoder.eval()
    decoder.eval()

    toImage = T.ToPILImage()
    psnr = PSNR().to(device)
    ssim = SSIM().to(device)

    ignoreEdge = 32
    psnrs = []
    ssims = []
    ratios = []

    with torch.no_grad():
        for idx, x in enumerate(dataLoader):
            idx += 1
            print(f'\nprocessing {idx} / {len(dataLoader)}')

            x = x.to(device)

            height, width = x.size()[-2:]

            compressed = encoder(x)
            reconstructed = decoder(compressed)[:, :, :height, :width]

            compressedPath = os.path.join(args.outputDir, f'{idx}.npy')

            np.save(compressedPath, compressed.cpu().numpy())

            original = toImage(x.squeeze()[:, :-ignoreEdge, :-ignoreEdge])
            prediction = toImage(reconstructed.squeeze()[:, :-ignoreEdge, :-ignoreEdge])

            originalPath = os.path.join(args.outputDir, f'{idx}.png')
            predictionPath = os.path.join(args.outputDir, f'{idx}_re.png')

            original.save(originalPath)
            prediction.save(predictionPath)

            originalSize = calcFileSize(originalPath)
            compressedSize = calcFileSize(compressedPath)

            psnrs.append(psnr(reconstructed, x).cpu().item())
            ssims.append(ssim(reconstructed, x).cpu().item())
            ratios.append(compressedSize[1] / originalSize[1])

            print(f'original size: {originalSize[1]:.2f} MB ({originalSize[0]} KB)')
            print(f'compressed size: {compressedSize[1]:.2f} MB ({compressedSize[0]} KB)')
            print(f'PSNR: {psnrs[-1]:.2f}')
            print(f'SSIM: {ssims[-1]:.2f}')

    print(f'\nAverage PSNR: {np.mean(psnrs):.2f}')
    print(f'Average SSIM: {np.mean(ssims):.2f}')
    print(f'Average Compression Ratio: {np.mean(ratios):.2f}')


if __name__ == '__main__':
    parser = ArgumentParser('python3 train.py', parents=[baseParser()])

    # MARK: - miscellaneous
    parser.add_argument('--outputDir', default='./output', type=str)
    parser.add_argument('--numWorkers', default=0, type=int)

    main(parser.parse_args())
