import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
import pretrainedmodels
import timm
from pytorch_pretrained_vit import ViT
from efficientnet_pytorch import EfficientNet


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


# Custom Model Template
class MyModel(nn.Module):
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
    def __init__(self, model_name, pretrained=True, num_classes_mask=3, num_classes_gender=2, num_classes_age=3):
        super().__init__()
        self.num_classes_mask = num_classes_mask
        self.num_classes_gender = num_classes_gender
        self.num_claases_age = num_classes_age
        
        self.model_name = model_name
        self.pretrained = pretrained
        
        self.net = timm.create_model(model_name=model_name, pretrained=pretrained)
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
