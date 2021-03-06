import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
import wandb
import cropface
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset, MaskSplitByProfileDataset
from dataset import get_fixed_labeled_csv, rand_bbox

from loss import create_criterion

from sklearn.metrics import f1_score


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=32, shuffle=False):

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
    # -- use wandb
    if args.use_wandb:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_ID)
    else:
        pass

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
    num_classes_mask = dataset.num_classes_mask
    num_classes_gender = dataset.num_classes_gender
    num_classes_age = dataset.num_classes_age

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
        drop_last=True
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
        model_name=model_name,
        num_classes_mask=num_classes_mask,
        num_classes_gender=num_classes_gender,
        num_classes_age=num_classes_age,
        pretrained=pretrained
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"),
                         args.optimizer)  # default: SGD
    optimizer = opt_module(
        model.parameters(),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    # earlystop
    patience = 7
    counter = 0

    best_val_acc = 0
    best_val_f1 = 0
    best_val_loss = np.inf

    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0

        figure = None

        for idx, train_batch in enumerate(train_loader):
            inputs, mask_label, gender_label, age_label = train_batch
            inputs = inputs.to(device)
            mask_label = mask_label.to(device)
            gender_label = gender_label.to(device)
            age_label = age_label.to(device)

            optimizer.zero_grad()

            if args.use_cut_mix and np.random.random() < args.cutMixProb:
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
                    labels = MaskBaseDataset.encode_multi_class(
                        mask_label_a, gender_label_a, age_label_a)
                else:
                    labels = MaskBaseDataset.encode_multi_class(
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
                labels = MaskBaseDataset.encode_multi_class(
                    mask_label, gender_label, age_label)
                outs = model(inputs)

                loss_mask = criterion(outs['mask'], mask_label)
                loss_gender = criterion(outs['gender'], gender_label)
                loss_age = criterion(outs['age'], age_label)
                loss = loss_mask + loss_gender + loss_age

            mask_preds = torch.argmax(outs['mask'], dim=-1)
            gender_preds = torch.argmax(outs['gender'], dim=-1)
            age_preds = torch.argmax(outs['age'], dim=-1)
            preds = MaskBaseDataset.encode_multi_class(
                mask_preds, gender_preds, age_preds)

            if figure is None:
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
                train_f1 = f1_score(labels.cpu().detach().numpy().tolist(
                ), preds.cpu().detach().numpy().tolist(), average='macro')
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || training f1 {train_f1:4.2} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss,
                                  epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc,
                                  epoch * len(train_loader) + idx)
                logger.add_scalar("Train/f1", train_f1,
                                  epoch * len(train_loader) + idx)
                wandb.log({'Train/loss': train_loss,
                          'Train/accuracy': train_acc})

                loss_value = 0
                matches = 0
        scheduler.step(float(loss))

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            val_f1_items = []

            for val_batch in val_loader:
                inputs, mask_label, gender_label, age_label = val_batch
                inputs = inputs.to(device)
                mask_label = mask_label.to(device)
                gender_label = gender_label.to(device)
                age_label = age_label.to(device)
                labels = MaskBaseDataset.encode_multi_class(
                    mask_label, gender_label, age_label)

                outs = model(inputs)
                mask_preds = torch.argmax(outs['mask'], dim=-1)
                gender_preds = torch.argmax(outs['gender'], dim=-1)
                age_preds = torch.argmax(outs['age'], dim=-1)
                preds = MaskBaseDataset.encode_multi_class(
                    mask_preds, gender_preds, age_preds)

                loss_mask = criterion(outs['mask'], mask_label)
                loss_gender = criterion(outs['gender'], gender_label)
                loss_age = criterion(outs['age'], age_label)
                loss_item = (loss_mask + loss_gender + loss_age).item()
                acc_item = (labels == preds).sum().item()
                f1_item = f1_score(labels.cpu().detach().numpy().tolist(
                ), preds.cpu().detach().numpy().tolist(), average='macro')

                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                val_f1_items.append(f1_item)

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            val_f1 = np.sum(val_f1_items) / len(val_loader)
            best_val_loss = min(best_val_loss, val_loss)
            best_val_f1 = max(best_val_f1, val_f1)

            if val_acc > best_val_acc:
                print(
                    f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(),
                           f"{save_dir}/bestAcc.pth")
                print('model path', f"{save_dir}")
                best_val_acc = val_acc

            if val_f1 > best_val_f1:
                print(
                    f"New best model for val f1 : {val_f1:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/bestF1.pth")
                print('model path', f"{save_dir}")
                best_val_f1 = val_f1
                counter = 0
            else:
                counter += 1

            if counter > patience:
                print("Early Stopping...")
                break

            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2}, f1: {val_f1:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}, best f1: {best_val_f1:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("Val/f1", val_f1, epoch)
            logger.add_figure("results", figure, epoch)
            wandb.log(
                {'Val/loss': val_loss, 'Val/accuracy': val_acc, "f1_score": val_f1})
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # from dotenv import load_dotenv
    import os
    # load_dotenv(verbose=True)

    from ConfigParser import seed, epochs, dataset, augmentation, resize, batch_size, valid_batch_size, model, optimizer, lr, val_ratio, criterion, lr_decay_step, log_interval, name, model_name, pretrained
    from ConfigParser import use_cropped_data, use_cut_mix, cut_mix_prob, use_wandb, wandb_ID, wandb_project_name
    parser.add_argument('--seed', type=int, default=seed,
                        help='random seed (config: ' + str(seed) + ')')
    parser.add_argument('--epochs', type=int, default=epochs,
                        help='number of epochs to train (config: ' + str(epochs) + ')')
    parser.add_argument('--dataset', type=str, default=dataset,
                        help='dataset augmentation type (config: ' + dataset + ')')
    parser.add_argument('--augmentation', type=str, default=augmentation,
                        help='data augmentation type (config: ' + augmentation + ')')
    parser.add_argument("--resize", nargs="+", type=list, default=resize,
                        help='resize size for image when training (config : ' + str(resize) + ')')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='input batch size for training (config: ' + str(batch_size) + ')')
    parser.add_argument('--valid_batch_size', type=int, default=valid_batch_size,
                        help='input batch size for validing (config: ' + str(valid_batch_size) + ')')
    parser.add_argument('--model', type=str, default=model,
                        help='model type (config: ' + model + ')')
    parser.add_argument('--optimizer', type=str, default=optimizer,
                        help='optimizer type (config: ' + optimizer + ')')
    parser.add_argument('--lr', type=float, default=lr,
                        help='learning rate (config: ' + str(lr) + ')')
    parser.add_argument('--val_ratio', type=float, default=val_ratio,
                        help='ratio for validaton (config: ' + str(lr) + ')')
    parser.add_argument('--criterion', type=str, default=criterion,
                        help='criterion type (config: ' + str(val_ratio) + ')')
    parser.add_argument('--lr_decay_step', type=int, default=lr_decay_step,
                        help='learning rate scheduler deacy step (config: ' + str(lr_decay_step) + ')')
    parser.add_argument('--log_interval', type=int, default=log_interval,
                        help='how many batches to wait before logging training status (config: '+str(log_interval)+')')
    parser.add_argument('--name', default=name,
                        help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--use_cut_mix', type=bool, default=use_cut_mix,
                        help='Choose to use cut mix')
    parser.add_argument('--cutMixProb', type=float, default=cut_mix_prob,
                        help='When you do cut mix, it do cut mix in this probability')
    parser.add_argument('--use_wandb', type=bool, default=use_wandb,
                        help='Choose which to use wandb')
    parser.add_argument('--wandb_ID', type=str, default=wandb_ID,
                        help='Your wandb ID')
    parser.add_argument('--wandb_project_name', type=str, default=wandb_project_name,
                        help='Your wandb project name to log')

    if not os.path.isfile("/opt/ml/code/labeled_data.csv"):
        print("You have to make error-fixed csv!!")
        get_fixed_labeled_csv()

    # Container environment
    if use_cropped_data:
        if os.path.isfile("/opt/ml/code/labeled_data.csv"):
            if not os.path.isdir("/opt/ml/input/data/train/new_imgs"):
                print("Saving Cropped images")
                cropface.get_cropped_and_fixed_images()
        else:
            print("You have to make error-fixed csv!!")
            get_fixed_labeled_csv()
            print("Saving Cropped images")
            cropface.get_cropped_and_fixed_images()

        parser.add_argument('--data_dir', type=str,
                            default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/new_imgs'))
    else:
        parser.add_argument('--data_dir', type=str,
                            default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))

    parser.add_argument('--model_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/input/model'))

    args = parser.parse_args()
    print(args)
    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)

    # tensorboard --logdir "/opt/ml/code/p1_baseline/model/exp13"
