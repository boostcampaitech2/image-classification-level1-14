from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

    

class MaskDataset(Dataset):
    def __init__(self, img_paths, labels, transform = None):
        self.img_paths = img_paths
        self.transform = transform
        self.labels = labels

    def __getitem__(self, index):
        pth = self.img_paths[index]
        
        if os.path.exists(pth + '.jpg'):
            self.img_paths[index] = pth + '.jpg'
        elif os.path.exists(pth + '.png'):
            self.img_paths[index] = pth + '.png'
        elif os.path.exists(pth + '.jpeg'):
            self.img_paths[index] = pth + '.jpeg'
            
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, self.labels[index]

    def show_pic(self, index):
        image = Image.open(self.img_paths[index])
        return image
        
    def get_label(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.img_paths)
    
    def __repr__(self):
        return "number of dataset : " + str(self.__len__())

