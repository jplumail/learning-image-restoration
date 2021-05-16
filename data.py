import os
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset


class BSDS300_images(VisionDataset):
    basedir = "BSDS300"
    train_file = "iids_train.txt"
    test_file = "iids_test.txt"
    url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
    archive_filename = "BSDS300-images.tgz"

    def __init__(self, root, sigma, train=True, transform=None, download=False):
        super(BSDS300_images, self).__init__(root, transform=transform)

        self.train = train
        self.sigma = sigma / 255
        
        self.root = root
        images_basefolder = os.path.join(root, self.basedir, "images")
        subfolder = "train" if self.train else "test"
        self.image_folder = os.path.join(images_basefolder, subfolder)
        id_file = self.train_file if self.train else self.test_file
        self.id_path = os.path.join(root, self.basedir, id_file)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        
        self.ids = np.loadtxt(self.id_path).astype('int')
        self.transform = transform
    
    def _check_exists(self):
        return os.path.exists(self.id_path) and os.path.exists(self.image_folder)
    
    def download(self):
        if self._check_exists():
            print("Files already downloaded")
            return
        download_and_extract_archive(self.url, download_root=self.root, filename=self.archive_filename)

    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, str(self.ids[idx])) + ".jpg"
        im = Image.open(img_name)
        if self.transform:
            im = self.transform(im)
        im_noisy = im + torch.randn(im.size()) * self.sigma
        return im_noisy, im


class BerkeleyLoader(data.DataLoader):
    def __init__(self, sigma, train=True, **kwargs):
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(90, pad_if_needed=True, padding_mode="reflect"),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.ToTensor()
        dataset = BSDS300_images("data", sigma, train=train, transform=transform, download=True)
        super(BerkeleyLoader, self).__init__(dataset, **kwargs)


if __name__ == "__main__":
    train_data = BerkeleyLoader(25, train=True, batch_size=10)
    for im_noisy, im in train_data:
        print(im_noisy.size(), im.size())