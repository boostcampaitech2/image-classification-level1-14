!pip install timm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 
from network import CustomNet
from dataset import ExampleDataset
from loss import ExampleLoss
import timm


def main(config_file):
    SEED = 2021
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore






num_classes = 18
model = timm.create_model('tf_efficientnetv2_s_in21ft1k', pretrained=True)
model.classifier = nn.Linear(1280,num_classes)

model.train()
 
# 옵티마이저 정의
params = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.001)

# 손실함수 정의
loss_fn = nn.CrossEntropyLoss()
    
###########################################
# 두번째 과제 Custom Dataset & DataLoader # -> 이번에는 공부하실 부분입니다!!
###########################################
 
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
