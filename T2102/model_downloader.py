import torchvision
import torch
model = torchvision.models.resnet18(pretrained = True)
torch.save(model, './base_models/'+'resnet18.pt')
