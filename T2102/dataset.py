from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

#Torchvision-Style Transforms
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize, GaussianBlur, RandomRotation, ColorJitter
import numpy as np
import torch


class MaskDataset(Dataset):
    def __init__(self, img_paths, labels = None, transform = None):
        self.img_paths = img_paths
        self.transform = transform
        self.labels = labels

    def __getitem__(self, index):
        pth = self.img_paths[index]
        pth = self.validate_ext(pth)
        image = Image.open(pth)

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, self.labels[index]

    def show_pic(self, index):
        pth = self.img_paths[index]
        pth = self.validate_ext(pth)
        image = Image.open(pth)
        return image

    def get_label(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.img_paths)
    
    def __repr__(self):
        return "number of dataset : " + str(self.__len__())

    def validate_ext(self, pth):
        if os.path.exists(pth + '.jpg'):
            return pth + '.jpg'
        elif os.path.exists(pth + '.png'):
            return pth + '.png'
        elif os.path.exists(pth + '.jpeg'):
            return pth + '.jpeg'

