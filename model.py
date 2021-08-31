import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from efficientnet_pytorch import EfficientNet
import pretrainedmodels

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)

class resnet50(nn.Module):
    def __init__(self, num_classes_mask, num_classes_gender, num_classes_age):
        super(resnet50,self).__init__()

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