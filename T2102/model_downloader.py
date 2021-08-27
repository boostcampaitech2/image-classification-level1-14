import torchvision
import torch
model = torchvision.models.resnet50(pretrained = True)
# torch.save(model, './base_models/'+'resnet50.pt')
