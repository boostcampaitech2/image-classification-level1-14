!pip install timm

import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm.notebook import tqdm
from time import time
from sklearn.model_selection import train_test_split
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data

from torchvision import transforms, models
from torchvision.transforms import Resize, ToTensor, Normalize
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import re
import csv
import timm
from dataset import get_transforms, MaskBaseDataset, get_fixed_labeled_csv
from model import train_model

def train(config_file):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dir = '/opt/ml/input/data/train'

    # Random seed 설정
    SEED = 2021
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

    # 정의한 Augmentation 함수와 Dataset 클래스 객체를 생성합니다.
    get_fixed_labeled_csv()
    transform = get_transforms
    image_path = pd.read_csv("labeled_data.csv").img_path
    label = pd.read_csv("labeled_data.csv").label

    dataset = MaskBaseDataset(image_path,label)

    # train dataset과 validation dataset을 8:2 비율로 나눕니다.
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = data.random_split(dataset, [n_train, n_val])


    # 각 dataset에 augmentation 함수를 설정합니다.
    train_dataset.dataset.set_transform(transform(need = 'train'))
    val_dataset.dataset.set_transform(transform(need = "val"))
    
    image_datasets={'train':train_dataset, 'validation':val_dataset}
    dataloaders = {
        'train':
        data.DataLoader(image_datasets['train'],
                        batch_size=12,
                        shuffle=True,
                        num_workers=4),  # for Kaggle
        'validation':
        data.DataLoader(image_datasets['validation'],
                        batch_size=12,
                        shuffle=False,
                        num_workers=4)  # for Kaggle
    }
    #모델을 설정합니다. 
    num_classes = 18
    model = timm.create_model('tf_efficientnetv2_s_in21ft1k', pretrained=True)
    
    
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Linear(1280,num_classes)

    # 옵티마이저 정의
    optimizer = optim.Adam(model.classifier.parameters())

    # 손실함수 정의
    loss_fn = nn.CrossEntropyLoss()
        
    
    model_trained = train_model(model, loss_fn, optimizer, num_epochs=30)
    
    !mkdir models
    !mkdir models/pytorch
    torch.save(model_trained, 'models/pytorch/weights.h5')

    # # meta 데이터와 이미지 경로를 불러옵니다.
    test_dir = '/opt/ml/input/data/eval'
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    test_image_dir = os.path.join(test_dir, 'images')

    # # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    test_image_paths = [os.path.join(test_image_dir, img_id) for img_id in submission.ImageID]

    def get_test_transforms(img_size=(512, 384), mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        transformations = Compose([
                    Resize(img_size[0], img_size[1]),
                    Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                    ToTensorV2(p=1.0),
                ], p=1.0)
        return transformations
   
    test_transform = get_test_transforms(mean=mean, std=std)

    test_dataset = TestDataset(test_image_paths, test_transform)

    test_loader = data.DataLoader(
        test_dataset,
        shuffle=False
    )

    # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
    model = torch.load('models/pytorch/weights.h5')
    model.eval()

    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
    all_predictions = []
    for images in tqdm(test_loader):
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
    print('test inference is done!')

if __name__ == "__main__" :
    train()
    # evaluate(test_data)
