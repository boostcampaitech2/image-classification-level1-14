from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms

#Torchvision-Style Transforms
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize, GaussianBlur, RandomRotation, ColorJitter
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

def submit():
    model_name = "ex322_model-efficientnet-b0_unfreezeRatio-1of2_baseDataset_weightedLoss-true_epochs-20resnet_18"
    PATH = './models/'+model_name + '.pt'

    # 테스트 데이터셋 폴더 경로를 지정해주세요.
    test_dir = '/opt/ml/input/data/eval'
    # testset = MaskBaseDataset(test_dir)
    # testloader = DataLoader(testset, shuffle = True)
    # meta 데이터와 이미지 경로를 불러옵니다.
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    transform = transforms.Compose([
        # Resize((512, 384), Image.BILINEAR),
        ToTensor(),
        # Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ])
    dataset = TestDataset(image_paths, transform)

    loader = DataLoader(
        dataset,
        shuffle=False
    )

    # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
    device = torch.device('cuda')
    # model = MyModel(num_classes=18).to(device)
    model = torch.load(PATH)
    model.eval()

    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
    all_predictions = []
    for images in tqdm(loader):
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join(test_dir, 'submission'+f"{model_name}" + '.csv'), index=False)
    print('test inference is done!')

if __name__ == "__main__":
    submit()