import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from dataset import get_fixed_labeled_csv, get_cropped_and_fixed_images
from dataset import encode_multi_class, rand_bbox
from loss import create_criterion

from sklearn.metrics import f1_score
import wandb


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(
        range(batch_size), k=n) if shuffle else list(range(n))
    # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    figure = plt.figure(figsize=(12, 18 + 2))
    # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    wandb.init(project=args.wandb_ProjectName, entity=args.wandb_ID)
    seed_everything(args.seed)
    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    # default: BaseAugmentation
    dataset_module = getattr(import_module("dataset"), args.dataset)
    dataset = dataset_module(
        data_dir=data_dir,
    )

    # -- augmentation
    transform_module = getattr(import_module(
        "dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    num_classes_mask = 3
    num_classes_gender = 2
    num_classes_age = 3
    model_module = getattr(import_module(
        "model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes_mask=num_classes_mask, num_classes_gender=num_classes_gender, num_classes_age=num_classes_age
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"),
                         args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )

    print("*"*100)
    print(f"optimizer , {optimizer}")
    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    early_stop_count = 0
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0

        columns = ["guess", "truth"]
        test_table = wandb.Table(columns=columns)
        figure = None

        for idx, train_batch in enumerate(train_loader):
            inputs, mask_label, gender_label, age_label = train_batch
            inputs = inputs.to(device)
            mask_label = mask_label.to(device)
            gender_label = gender_label.to(device)
            age_label = age_label.to(device)

            optimizer.zero_grad()

            if args.cutMix and np.random.random() < args.cutMixProb:
                lam = np.random.beta(1, 1)
                rand_index = torch.randperm(inputs.size()[0]).to(device)

                mask_label_a = mask_label  # 원본 이미지 label
                mask_label_b = mask_label[rand_index]  # 패치 이미지 label
                gender_label_a = gender_label
                gender_label_b = gender_label[rand_index]
                age_label_a = age_label
                age_label_b = age_label[rand_index]

                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index,
                                                            :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                           (inputs.size()[-1] * inputs.size()[-2]))
                if lam > 0.5:
                    labels = encode_multi_class(
                        mask_label_a, gender_label_a, age_label_a)
                else:
                    labels = encode_multi_class(
                        mask_label_b, gender_label_b, age_label_b)

                outs = model(inputs)

                loss_mask = criterion(
                    outs['mask'], mask_label_a)*lam + criterion(outs['mask'], mask_label_b) * (1 - lam)
                loss_gender = criterion(
                    outs['gender'], gender_label_a)*lam + criterion(outs['gender'], gender_label_b) * (1 - lam)
                loss_age = criterion(
                    outs['age'], age_label_a)*lam + criterion(outs['age'], age_label_b) * (1 - lam)
                loss = loss_mask + loss_gender + loss_age

            else:
                labels = encode_multi_class(
                    mask_label, gender_label, age_label)
                outs = model(inputs)

                loss_mask = criterion(outs['mask'], mask_label)
                loss_gender = criterion(outs['gender'], gender_label)
                loss_age = criterion(outs['age'], age_label)
                loss = loss_mask + loss_gender + loss_age

            mask_preds = torch.argmax(outs['mask'], dim=-1)
            gender_preds = torch.argmax(outs['gender'], dim=-1)
            age_preds = torch.argmax(outs['age'], dim=-1)
            preds = encode_multi_class(mask_preds, gender_preds, age_preds)

            if figure is None:
                inputs = inputs[:16, :, :, :]
                # runtime 오류 계속 나면 여기좀 체크해주세요... 뭐가 문젠지...
                inputs_np = torch.clone(inputs).detach(
                ).cpu().permute(0, 2, 3, 1).numpy()
                inputs_np = dataset_module.denormalize_image(
                    inputs_np, dataset.mean, dataset.std)
                figure = grid_image(
                    inputs_np, labels, preds, n=16, shuffle=False
                )

            loss.backward(retain_graph=True)
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss,
                                  epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc,
                                  epoch * len(train_loader) + idx)
                wandb.log(
                    {'Train/loss': train_loss, 'Train/accuracy': train_acc})

                scheduler.step(loss_value)
                loss_value = 0
                matches = 0

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            epoch_f1 = 0
            n_iter = 0

            for val_batch in val_loader:
                inputs, mask_label, gender_label, age_label = val_batch
                inputs = inputs.to(device)
                mask_label = mask_label.to(device)
                gender_label = gender_label.to(device)
                age_label = age_label.to(device)
                labels = encode_multi_class(
                    mask_label, gender_label, age_label)

                outs = model(inputs)

                mask_preds = torch.argmax(outs['mask'], dim=-1)
                gender_preds = torch.argmax(outs['gender'], dim=-1)
                age_preds = torch.argmax(outs['age'], dim=-1)
                preds = encode_multi_class(mask_preds, gender_preds, age_preds)

                loss_mask = criterion(outs['mask'], mask_label)
                loss_gender = criterion(outs['gender'], gender_label)
                loss_age = criterion(outs['age'], age_label)
                loss_item = (loss_mask + loss_gender + loss_age).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                epoch_f1 += f1_score(labels.cpu().numpy(),
                                     preds.cpu().numpy(), average='macro')
                n_iter += 1

                for i in range(args.valid_batch_size):
                    test_table.add_data(labels[i], preds[i])

            epoch_f1 = epoch_f1/n_iter
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(
                    f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                print('model path', f"{save_dir}")
                best_val_acc = val_acc
            else:
                early_stop_count += 1

            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2} || "
                f"f1_score : {epoch_f1:0.4}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            wandb.log({"table_key": test_table})
            wandb.log(
                {'Val/loss': val_loss, 'Val/accuracy': val_acc, "f1_score": epoch_f1})
            print()

            if early_stop_count == 20:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # from dotenv import load_dotenv
    import os
    # load_dotenv(verbose=True)

    # 다양한 하이퍼파라미터 세팅으로 학습 진행
    # 텐서보드를 통해 학습 양상을 실시간으로 확인, 점검하고 실험 간 결과 비교
    # 커스텀 모듈을 추가해 성능을 더 끌어올려보기
    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=2141,
                        help='random seed (default: 2141)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset',
                        help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation',
                        help='data augmentation type (default: CustomAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list,
                        default=[128, 128], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=16,
                        help='input batch size for validing (default: 16)')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='model type (default: resnet50)')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='label_smoothing',
                        help='criterion type (default: focal)')
    parser.add_argument('--lr_decay_step', type=int, default=100,
                        help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp',
                        help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--cutMix', default=True, help='Decide using cutmix')
    parser.add_argument('--cutMixProb', default=0.3,
                        help='Ratio using cutmixed input')
    parser.add_argument('--wandb_ID', default='tkdlqh2',
                        help='Weight&Biases ID')
    parser.add_argument('--wandb_ProjectName',
                        default='my-test-project', help='Weight&Biases Project Name')

    if os.path.isfile("/opt/ml/code/labeled_data.csv"):
        if not os.path.isdir("/opt/ml/input/data/train/new_imgs"):
            print("Saving Cropped images")
            get_cropped_and_fixed_images()
    else:
        print("You have to make error-fixed csv!!")
        get_fixed_labeled_csv()
        print("Saving Cropped images")
        get_cropped_and_fixed_images()

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get(
        'SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/new_imgs'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get(
        'SM_MODEL_DIR', '/opt/ml/image-classification-level1-14'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
