import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F

from utils import num_unfreeze_ratio, base_model_dir, model_list, model_index

class MaskModel(nn.Module):
    def __init__(self, num_classes: int = 18, unfreeze = None):
        super(MaskModel, self).__init__()
        #self.base_model = None#torchvision.models.resnet18(pretrained = True)
        cur_model = model_list[model_index]
        self.base_model = torch.load(base_model_dir + cur_model + ".pt")
        print(self.base_model)

        # self.resnet_18 = torchvision.models.resnet18(pretrained = True)
        self.linear = torch.nn.Linear(1000, 18, bias = True)
        # self.resnet_18.fc = torch.nn.Linear(512, 18, bias = True)

        num_seq = 0
        for c in self.base_model.named_children():
            if isinstance(c[1], nn.Sequential):
                num_seq += 1

        num_unfreeze = num_seq // num_unfreeze_ratio
        print("%d of sequential layers is unfrozen." %(num_unfreeze))
        idx = 0
        for c in self.base_model.named_children():
            if isinstance(c[1], nn.Sequential):
                if idx >= num_unfreeze:
                    break
                for p in c[1].parameters():
                    p.requires_grad = False

                idx += 1
        
        for p in self.linear.parameters():
            p.requires_grad = True
         
        # for p in self.base_model.fc.parameters():
        #     p.requres_grad = True

        

    def init_params(self):
        # torch.nn.init.xavier_uniform_(self.base_model.fc.weight)
        # self.base_model.fc.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        x = self.linear(x)
        return x
