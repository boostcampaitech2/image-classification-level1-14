
import utils
from utils import num_classes, config, IS_LOG, BALANCE_TESTSET, num_unfreeze_ratio, K_FOLD, K_FOLD_NUM, K_FOLD_EPOCH

from MaskDataset import MaskDataset
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random

seed = 2141
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

DATASET_SIZE = config['dataset']['data_size']
BATCH_SIZE = int(config['dataset']['batch_size'])
EPOCHS = int( config['trainer']['epoch'] )
EX_NUM = config['trainer']['ex']

ex_num_name = "ex" + EX_NUM +"_num_unfreeze_ratio_"+str(num_unfreeze_ratio)+"_epochs_"+ str(EPOCHS) +"_dataset_size_" + str(DATASET_SIZE) + "_banace_testset_" + str(BALANCE_TESTSET)

config['trainer']['ex'] = str(int(config['trainer']['ex']) + 1)
with open('config.cfg', 'w') as configfile:
    config.write(configfile)





from utils import BASE_DATASET
import torch.utils.data as data

train_data= test_data = None
if BASE_DATASET:
####################################################################################################
    from utils import img_dir

    from basedataset import get_transforms, MaskBaseDataset,mean, std
    ex_num_name += "_base_dataset"

    # 정의한 Augmentation 함수와 Dataset 클래스 객체를 생성합니다.
    # transform = get_transforms(mean=mean, std=std)

    dataset = MaskBaseDataset(
        img_dir=img_dir
    )

    # train dataset과 validation dataset을 8:2 비율로 나눕니다.


    # 각 dataset에 augmentation 함수를 설정합니다.
    # train_data.dataset.set_transform(transform['train'])
    # test_data.dataset.set_transform(transform['val'])
    

####################################################################################################
else:
    if not BALANCE_TESTSET:
        labels, img_paths = utils.get_labels_and_img_paths(DATASET_SIZE)

        dataset = MaskDataset(img_paths, labels)

        # train_data, test_data = utils.split_eval(data)
        # print("train data : %d test_data : %d" %(len(train_data), len(test_data)))



    # else:
        # if K_FOLD:
        #     raise Exception('config conflicts: k fold and custom dataset cannot occur simultaneously.')
    #     (true_labels, true_img_paths), (false_labels, false_img_paths) = utils.get_bal_labels_and_bal_img_paths(DATASET_SIZE)

    #     false_dataset = MaskDataset(false_img_paths, false_labels)
    #     true_dataset = MaskDataset(true_img_paths, true_labels)

    #     each_test_size = 500
    #     false_train_data, false_test_data = utils.split_eval(false_dataset,BALANCE_TESTSET, size = each_test_size)
    #     true_train_data, true_test_data = utils.split_eval(true_dataset, BALANCE_TESTSET, size = each_test_size)

    #     train_data = torch.utils.data.ConcatDataset([false_train_data, true_train_data])
    #     test_data = torch.utils.data.ConcatDataset([false_test_data, true_test_data])
    #     ####################################################################################################
    #     # ex_num_name += "_only negative trainset"

    #     # test_data = false_test_data
    #     ####################################################################################################
    #     print("train data : %d test_data : %d" %(len(train_data), len(test_data)))

    #     false = [i for i in range(6,18)]
    #     count_f = 0
    #     count_t = 0

    #     for t in test_data:
    #         if t[1] in false:
    #             count_f +=1
    #         else:
    #             count_t +=1 
    #     print("testdata balance stat\nmask data: %dn normal and incorrect data %d" %(count_t, count_f))
    #     if count_t != count_f:
    #         print("The test dataset is not balance. This may be woring. Proceed?")
    #         while True:
    #             ans = input("type yes: ")
    #             if ans == "yes":
    #                 break



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from model import MaskModel
model = MaskModel(num_classes).to(device)
model.init_params()



from LossOptim import get_loss_and_opt
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from Evaluete import evaluate

if IS_LOG:
    writer = SummaryWriter("app/" + ex_num_name)
    writer_epoch_index = 0
else:
    print("#"*10)
    print("NOT RECORDING IN TENSORBOARD. ")
    while True:
        confirm = input("type yes: ")
        if confirm == "yes":
            break


def train(EPOCHS, fold = None):
    model.train()
    num_of_batches = len(dataloader)

    writer = None

    e_count = 0

    if K_FOLD:
        e_count = fold * K_FOLD_EPOCH 

    for e in range(EPOCHS):
        print("\nepoch number :", e_count)
        running_acc_sum = 0
        loss_sum = 0
        train_f1_score_sum = 0

        for idx, (batch_in, batch_out) in enumerate(tqdm(dataloader)):
            model.train()
            X = batch_in.to(device)
            Y = batch_out.to(device)
            
            logits = model(X)
            mini_batch_loss = loss(logits, Y)
            _, preds = torch.max(logits, 1)
            
            opt.zero_grad()
            mini_batch_loss.backward()
            opt.step()
            

            mini_batch_acc = torch.sum(preds == Y.data).item() / BATCH_SIZE
            mini_batch_f1_score = f1_score(Y.cpu().detach().numpy().tolist(), preds.cpu().detach().numpy().tolist(), average = 'macro')

            loss_sum += mini_batch_loss.item() # CrossEntLoss returns ave loss of mini batch
            running_acc_sum += mini_batch_acc
            train_f1_score_sum += mini_batch_f1_score

        e_count += 1

        epoch_loss = loss_sum / num_of_batches
        train_f1_score = train_f1_score_sum / num_of_batches
        train_acc = running_acc_sum/ num_of_batches

        test_loss, test_acc, test_f1_score = evaluate(model, loss, device, testloader,num_testset, e_count, writer, record = IS_LOG)

        print("train loss : %-10.5f test loss = %-10.5f" %(epoch_loss, test_loss))
        print("train acc: %-10.5f test accuracy = %-10.5f" %(train_acc, test_acc))
        print("train  f1 marco average: %-10.5f test  f1 marco average: %-10.5f" %(train_f1_score, test_f1_score))


        if IS_LOG:
            writer.add_scalar('loss/training loss', 
                epoch_loss, e_count)
            writer.add_scalar('f1/training f1', 
                train_f1_score, e_count)
            writer.add_scalar('acc/training acc', 
                train_acc, e_count)

    
    torch.save(model, './models/'+ex_num_name+'resnet_18.pt')
    print("epoch = %d done!" %(EPOCHS))

    if IS_LOG:
        writer.flush()
        writer.close()





loss, opt, ex_num_name = get_loss_and_opt(ex_num_name, device, model)
print("#"*10)
print(ex_num_name)
# K-fold branch
if K_FOLD:
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits = K_FOLD_NUM, shuffle = True)
    ex_num_name += "_kfold_true"
    print(len(dataset))
    EPOCHS = K_FOLD_EPOCH
    for fold, (train_ids, test_ids) in enumerate(k_fold.split(dataset)):
        print("fold: %d, number of dataset :%d" %(fold, len(train_ids)))

        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_sampler = torch.utils.data.SubsetRandomSampler(test_ids)

        dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, sampler = train_sampler)
        num_dataset = len(train_ids)
        testloader = DataLoader(dataset, batch_size = BATCH_SIZE, sampler = val_sampler)
        num_testset = len(test_ids)
        train(EPOCHS, fold)

else:
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_data, test_data = data.random_split(dataset, [n_train, n_val])

    dataloader = DataLoader(train_data, shuffle = True, batch_size = BATCH_SIZE)
    num_dataset = len(train_data)
    testloader = DataLoader(test_data, shuffle = True, batch_size = BATCH_SIZE)
    num_testset = len(test_data)

    train(EPOCHS)
# if __name__ == "__main__" :
    # train()
    # evaluate(test_data)
