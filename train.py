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
from dataset_fixed import get_transforms, MaskBaseDataset
from model import model_trained

def main(config_file):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dir = '/opt/ml/input/data/train'

    SEED = 2021
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

    # 정의한 Augmentation 함수와 Dataset 클래스 객체를 생성합니다.
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
        
    
    # 학습을 위한 데이터셋 생성
    dataset_example = ExampleDataset()
    
    # 학습을 위한 데이터로더 생성
    dataloader_example = DataLoader(dataset_example)
    
    ##########################################################
    # 세번째 과제 Transfer Learning & Hyper Parameter Tuning # 
    ##########################################################
    for e in range(epochs):
        for X,y in dataloader_example:
            output = model(X)
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
        optimizer.step()

if __name__ == "__main__" :
    train()
    # evaluate(test_data)
