from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

#Torchvision-Style Transforms
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize, GaussianBlur, RandomRotation, ColorJitter
import numpy as np
import torch

# class AddGaussianNoise(object):
#     def __init__(self, mean=0., std=1.):
#         self.std = std
#         self.mean = mean

#     def __call__(self, tensor):
#         return tensor + torch.randn(tensor.size()) * self.std + self.mean

#     def __repr__(self):
#         return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# def get_transforms(need=('train', 'val'), img_size=(512, 384)):
#     transformations = {}
#     if 'train' in need:
#         transformations['train'] = transforms.Compose([
#             Resize((img_size[0], img_size[1])),
#             RandomRotation([-8, +8]),
#             GaussianBlur(51, (0.1, 2.0)),
#             ColorJitter(brightness=0.5, saturation=0.5, hue=0.5),  # todo : param
#             ToTensor(),
#             Normalize(mean=mean, std=std),
#             AddGaussianNoise(0., 1.)
#         ])
#     if 'val' in need:
#         transformations['val'] = transforms.Compose([
#             Resize((img_size[0], img_size[1])),
#             ToTensor(),
#             Normalize(mean=mean, std=std),
#         ])
#     return transformations

### 마스크 여부, 성별, 나이를 mapping할 클래스를 생성합니다.

class MaskLabels:
    mask = 0
    incorrect = 1
    normal = 2

class GenderLabels:
    male = 0
    female = 1

class AgeGroup:
    map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 60 else 2

class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1.jpg": MaskLabels.mask,
        "mask2.jpg": MaskLabels.mask,
        "mask3.jpg": MaskLabels.mask,
        "mask4.jpg": MaskLabels.mask,
        "mask5.jpg": MaskLabels.mask,
        "incorrect_mask.jpg": MaskLabels.incorrect,
        "normal.jpg": MaskLabels.normal
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, img_dir, transform=None):
        """
        MaskBaseDataset을 initialize 합니다.

        Args:
            img_dir: 학습 이미지 폴더의 root directory 입니다.
            transform: Augmentation을 하는 함수입니다.
        """
        self.img_dir = img_dir
        # self.mean = mean
        # self.std = std
        self.transform = transforms.Compose([
            # Resize((img_size[0], img_size[1])),
            # RandomRotation([-8, +8]),
            # GaussianBlur(51, (0.1, 2.0)),
            # ColorJitter(brightness=0.5, saturation=0.5, hue=0.5),  # todo : param
            ToTensor()#,
            # Normalize(mean=mean, std=std)#,
          #  AddGaussianNoise(0., 1.)
        ])

        self.setup()

    def set_transform(self, transform):
        """
        transform 함수를 설정하는 함수입니다.
        """
        if transforms == None:
            return transforms.Compose([
            Resize((img_size[0], img_size[1])),
            RandomRotation([-8, +8]),
            GaussianBlur(51, (0.1, 2.0)),
            ColorJitter(brightness=0.5, saturation=0.5, hue=0.5),  # todo : param
            ToTensor(),
            Normalize(mean=mean, std=std)#,
          #  AddGaussianNoise(0., 1.)
        ])
        else:
            return transform
        
    def setup(self):
        """
        image의 경로와 각 이미지들의 label을 계산하여 저장해두는 함수입니다.
        """
        profiles = os.listdir(self.img_dir)
        for profile in profiles:
            for file_name, label in self._file_names.items():
                img_path = os.path.join(self.img_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.mask_labels.append(label)

                    id, gender, race, age = profile.split("_")
                    gender_label = getattr(GenderLabels, gender)
                    age_label = AgeGroup.map_label(age)

                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

    def __getitem__(self, index):
        """
        데이터를 불러오는 함수입니다. 
        데이터셋 class에 데이터 정보가 저장되어 있고, index를 통해 해당 위치에 있는 데이터 정보를 불러옵니다.
        
        Args:
            index: 불러올 데이터의 인덱스값입니다.
        """
        # 이미지를 불러옵니다.
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        
        # 레이블을 불러옵니다.
        mask_label = self.mask_labels[index]
        gender_label = self.gender_labels[index]
        age_label = self.age_labels[index]
        multi_class_label = mask_label * 6 + gender_label * 3 + age_label
        
        # 이미지를 Augmentation 시킵니다.
        image_transform = image#self.transform(image)#self.transform(image=np.array(image))['image']
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

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

