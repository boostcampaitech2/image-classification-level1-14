import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
import pretrainedmodels
import timm
from pytorch_pretrained_vit import ViT
from efficientnet_pytorch import EfficientNet

class resnet50(nn.Module):
    def __init__(self, num_classes_mask, num_classes_gender, num_classes_age):
        super(resnet50, self).__init__()

        self.net = pretrainedmodels.__dict__['resnet50'](pretrained="imagenet")

        self.linear_mask = nn.Sequential(nn.Linear(2048, num_classes_mask))
        self.linear_gender = nn.Sequential(nn.Linear(2048, num_classes_gender))
        self.linear_age = nn.Sequential(nn.Linear(2048, num_classes_age))

        # for param in self.net.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.net.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        mask = self.linear_mask(x)
        gender = torch.sigmoid(self.linear_gender(x))
        age = self.linear_age(x)
        return {'mask': mask, 'gender': gender, 'age': age}

class efficient(nn.Module):
    def __init__(self, num_classes_mask, num_classes_gender, num_classes_age):
        super(efficient,self).__init__()

        self.net = EfficientNet.from_pretrained('efficientnet-b4')

        self.linear_mask = nn.Sequential(nn.Linear(1792, num_classes_mask))
        self.linear_gender = nn.Sequential(nn.Linear(1792, num_classes_gender))
        self.linear_age = nn.Sequential(nn.Linear(1792, num_classes_age))

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.net.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        mask = self.linear_mask(x)
        gender = torch.sigmoid(self.linear_gender(x))
        age = self.linear_age(x)
        return {'mask': mask, 'gender': gender, 'age': age}

class vit16(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(vit16, self).__init__()
        vit_model_sample = timm.create_model(model_name = "vit_base_patch16_224", # 불러올 모델 architecture,
                                     num_classes=num_classes, # 예측 해야하는 class 수
                                     pretrained = True # 사전학습된 weight 불러오기
                                     )
        self.model = vit_model_sample

    def forward(self, x):
        x = self.model(x)
        return x