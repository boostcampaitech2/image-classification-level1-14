import argparse
import os
from importlib import import_module
import cropface_test

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset

import multiprocessing


def load_model(saved_model, num_classes_mask, num_classes_gender, num_classes_age, device):
    # model.py import 하고 해당 모듈의 args.model class 불러옴
    model_module = getattr(import_module(
        "model"), args.model)  # default: BaseModel
    model = model_module(num_classes_mask=num_classes_mask,
                         num_classes_gender=num_classes_gender, num_classes_age=num_classes_age)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'bestAcc.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes_mask = 3
    num_classes_gender = 2
    num_classes_age = 3

    model = load_model(model_dir, num_classes_mask,
                       num_classes_gender, num_classes_age, device).to(device)
    model = torch.nn.DataParallel(model)
    print(f"model dir :  {model_dir}")
    model.eval()

    img_root = os.path.join(data_dir, 'new_imgs')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)

            outs = model(inputs)
            mask_preds = torch.argmax(outs['mask'], dim=-1)
            gender_preds = torch.argmax(outs['gender'], dim=-1)
            age_preds = torch.argmax(outs['age'], dim=-1)
            pred = MaskBaseDataset.encode_multi_class(
                mask_preds, gender_preds, age_preds)

            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(os.path.join(output_dir, f'output.csv'))
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for validing (default: 64)')
    parser.add_argument('--resize', type=tuple, default=(260, 200),
                        help='resize size for image when you trained (default: (128,96))')
    parser.add_argument('--model', type=str, default='efficient',
                        help='model type (default: resnet50)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get(
        'SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get(
        'SM_CHANNEL_MODEL', '/opt/ml/input/model/efficient'))
    parser.add_argument('--output_dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))

    if not os.path.isdir("/opt/ml/input/data/eval/new_imgs"):
        cropface_test.get_cropped_and_fixed_images()
    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
