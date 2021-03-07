import os
from glob import glob

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T


class COEN338Dataset(Dataset):
    def __init__(self, baseDir: str, imageType: str, imageWidth: int = 800, imageHeight: int = 800,
                 mode: str = 'train'):
        self.imagePaths = glob(os.path.join(baseDir, f'*.{imageType}'))

        if mode == 'train':
            self.transforms = T.Compose([
                T.RandomResizedCrop((imageHeight, imageWidth)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.RandomErasing()
            ])
        else:
            self.transforms = T.ToTensor()

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        path = self.imagePaths[idx]
        img = Image.open(path)

        return self.transforms(img)
