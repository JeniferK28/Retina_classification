import os
import natsort
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):

        with open(annotations_file) as f:
            lines = f.readlines()
        self.img_labels = lines
        self.img_dir = img_dir
        all_imgs = os.listdir(img_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.total_imgs[idx])
        image = Image.open(img_path).convert('RGB')
        label = int(self.img_labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label


class CustomGradDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, gradcam= 'True'):
        if gradcam == 'True':
            with open(annotations_file) as f:
                lines = f.readlines()
            self.img_labels = lines
        self.img_dir = img_dir
        all_imgs = os.listdir(img_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.gradcam = gradcam

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.total_imgs[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.gradcam == 'True':
            label = int(self.img_labels[idx])
            return image, label
        else: return image