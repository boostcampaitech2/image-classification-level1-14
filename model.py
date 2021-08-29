import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import timm

# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

class convit(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = timm.create_model('convit_base',pretrained=True)
        num_feat = self.net._fc.in_features
        self._fc = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        return self.net(x)


class efficient(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b3')
        num_feat = self.net._fc.in_features
        self._fc = nn.Linear(num_feat, num_classes)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return self.net(x)


# Custom Model Template
class resnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = models.resnet50(pretrained=True)
        self.net.classifier = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, num_classes))

    def forward(self, x):
        return self.net(x)


# Custom Model Template
class VGG19(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.net = models.vgg19_bn(pretrained=True)
        self.net.classifier = nn.Sequential(
            nn.Linear(512*7*7,4896),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4896, 4896),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4896,num_classes)
        )

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """

        return self.net(x)