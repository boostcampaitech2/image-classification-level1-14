!pip install pandas
!apt-get update
!apt-get -y install libgl1-mesa-glx
!pip install opencv-python
!pip install albumentations

import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import cv2
import csv
import torch
import torch.utils.data as data

from albumentations import *
from albumentations.pytorch import ToTensorV2


train_data_path = './../input/data/train'
train_image_path = f'{train_data_path}/images'

def get_fixed_labeled_csv(): 
    df = pd.read_csv(f"{train_data_path}/train.csv")

    id_overlap_error = ["003397"]
    gender_labeling_error = ['006359', '006360', '006361', '006362', '006363', '006364']
    mask_labeling_error = ['000020', '004418', '005227']

    id_max = int(max(df['id']))
    id_new = id_max+1

    new_data_list=[]

    for idx in tqdm(range(len(df))):  # tqdm 을 이용하면 현재 데이터가 얼마나 처리되고 있는지 파악되어 좋습니다.
        _path = df['path'].iloc[idx]  # 순서대로 가져와야 하기 때문에 iloc을 사용해 가져옵니다.
        _gender = df['gender'].iloc[idx]
        _age = df['age'].iloc[idx]
        _id = df['id'].iloc[idx]

        if _id in id_overlap_error:
            _id='%06d'%(id_new)
            id_new += 1
        
        if _id in gender_labeling_error:
            if _gender == "male":
                _gender = 'female'
            else:
                _gender = 'male'
        
        for img_name in Path(f"{train_image_path}/{_path}").iterdir():  # 각 dir의 이미지들을 iterative 하게 가져옵니다.
            img_stem = img_name.stem  # 해당 파일의 파일명만을 가져옵니다. 확장자 제외.
            if not img_stem.startswith('._'):  # avoid hidden files
                if _id in mask_labeling_error:
                    if img_stem == "incorrect_mask":
                        img_stem = 'normal'
                    elif img_stem == 'normal':
                        img_stem = 'incorrect_mask'
                new_data_list.append([_id, _age, _gender, img_stem, img_name.__str__()]) 
        
    df = pd.DataFrame(new_data_list)
    df.columns = ['id', 'age', 'gender', 'stem', 'img_path']
    
    df['label'] = 0  # SET SCORE
    # AGE
    df['label'] += ((df['age'] >= 30) & (df['age'] < 60))*1
    df['label'] += (df['age'] >= 60)*2

    # GENDER
    df['label'] += (df['gender'] == 'female')*3

    # MASK wearing condition
    df['label'] += (df['stem'].isin(['incorrect_mask']))*6
    df['label'] += (df['stem'].isin(['normal']))*12

    df.to_csv('./labeled_data.csv', sep=',' ,na_rep='NaN')

def get_transforms(need=('train', 'val'), img_size=(512, 384), mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
    """
    train 혹은 validation의 augmentation 함수를 정의합니다. train은 데이터에 많은 변형을 주어야하지만, validation에는 최소한의 전처리만 주어져야합니다.
    
    Args:
        need: 'train', 혹은 'val' 혹은 둘 다에 대한 augmentation 함수를 얻을 건지에 대한 옵션입니다.
        img_size: Augmentation 이후 얻을 이미지 사이즈입니다.
        mean: 이미지를 Normalize할 때 사용될 RGB 평균값입니다.
        std: 이미지를 Normalize할 때 사용될 RGB 표준편차입니다.

    Returns:
        transformations: Augmentation 함수들이 저장된 dictionary 입니다. transformations['train']은 train 데이터에 대한 augmentation 함수가 있습니다.
    """
    transformations = {}
    if 'train' in need:
        transformations['train'] = Compose([
            Resize(img_size[0], img_size[1], p=1.0),
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            GaussNoise(p=0.5),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    if 'val' in need:
        transformations['val'] = Compose([
            Resize(img_size[0], img_size[1]),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    return transformations

class MaskBaseDataset(data.Dataset):
    num_classes = 3 * 2 * 3

    def __init__(self,image_path,label,transform=None):
        self.image_path = image_path
        self.label = label
        self.transform = transform
        
    def set_transform(self, transform):
        """
        transform 함수를 설정하는 함수입니다.
        """
        self.transform = transform
        

    def __getitem__(self, index):
        """
        데이터를 불러오는 함수입니다. 
        데이터셋 class에 데이터 정보가 저장되어 있고, index를 통해 해당 위치에 있는 데이터 정보를 불러옵니다.
        
        Args:
            index: 불러올 데이터의 인덱스값입니다.
        """
        # 이미지를 불러옵니다.
        image = Image.open(image_path[index])
        
        # 이미지를 Augmentation 시킵니다.
        image_transform = self.transform(image=np.array(image))['image']
        return image_transform, self.label[index]

    def __len__(self):
        return len(self.image_path)