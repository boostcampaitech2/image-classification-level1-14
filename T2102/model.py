import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F

class MaskModel(nn.Module):
    def __init__(self, num_classes: int = 18):
        super(MaskModel, self).__init__()
        self.resnet_18 = torchvision.models.resnet18(pretrained = True)
        self.resnet_18.fc = torch.nn.Linear(512, 18, bias = True)

        for p in self.resnet_18.parameters():
            p.requires_grad = False
        
        for p in self.resnet_18.fc.parameters():
            p.requires_grad = True
         
    def init_params(self):
        torch.nn.init.xavier_uniform_(self.resnet_18.fc.weight)
        self.resnet_18.fc.bias.data.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet_18(x)
        return x
