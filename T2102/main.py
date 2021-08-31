
import utils
from utils import EX_NUM, NUM_CLASSES, config, IS_LOG, BALANCE_TESTSET, \
                    K_FOLD, K_FOLD_NUM, K_FOLD_EPOCH,\
                    DATASET_SIZE, BATCH_SIZE, EPOCHS, EX_NUM, MODEL_NAME #UNFREEZE_RATIO

from MaskDataset import MaskDataset
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

# if UNFREEZE_RATIO == float('inf'):
#     unfz_msg = 'allFrozen'
# elif UNFREEZE_RATIO == 1:
#     unfz_msg = 'allUnfrozen'
# else:
#     unfz_msg = '1of'+ str(UNFREEZE_RATIO)


ex_num_name = "ex" + EX_NUM + "_model-" + MODEL_NAME# + "_unfreezeRatio-" + unfz_msg

config['trainer']['ex'] = str(int(config['trainer']['ex']) + 1)
with open('config.cfg', 'w') as configfile:
    config.write(configfile)





from utils import BASE_DATASET
import torch.utils.data as data

train_data= test_data = None
if BASE_DATASET:
####################################################################################################
    from utils import img_dir

    # from basedataset import get_transforms, MaskBaseDataset,mean, std
    # ex_num_name += "_baseDataset"

    # dataset = MaskBaseDataset(
    #     img_dir=img_dir
    # )

    from importlib import import_module
    ex_num_name += "_baselineBaseDataset"
    from baseline.dataset import  MaskBaseDataset, CustomAugmentation

    # -- dataset
    # dataset_module = getattr(import_module("baseline.dataset"), MaskBaseDataset)  # default: BaseAugmentation
    # dataset = dataset_module(
    #     data_dir=img_dir,
    # )
    dataset = MaskBaseDataset(img_dir)

    # -- augmentation
    transform = CustomAugmentation(        resize=[128, 96], mean=dataset.mean, std=dataset.std)
#     transform_module = getattr(import_module("baseline.dataset"), CustomAugmentation)  # default: BaseAugmentation
#     transform = transform_module(
# ,
#     )
    dataset.set_transform(transform)
    

####################################################################################################
else:
    if not BALANCE_TESTSET:
        labels, img_paths = utils.get_labels_and_img_paths(DATASET_SIZE)
        ex_num_name +="_datasetSize-" + str(DATASET_SIZE)
        dataset = MaskDataset(img_paths, labels)

        # train_data, test_data = utils.split_eval(data)
        # print("train data : %d test_data : %d" %(len(train_data), len(test_data)))



    # else:
        # if K_FOLD:
        #     raise Exception('config conflicts: k fold and custom dataset cannot occur simultaneously.')
        # ex_num_name +=  + "_banace_testset_" + str(BALANCE_TESTSET)
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
model = MaskModel(NUM_CLASSES).to(device)
model.init_params()


from LossOptim import get_loss_and_opt
from torch.utils.tensorboard import SummaryWriter

import logging


from train import train
loss, opt, ex_num_name = get_loss_and_opt(ex_num_name, device, model)
print("#"*10)

ex_num_name += "_kfold-"+str(K_FOLD_NUM) + "_kfoldEpochs-"+ str(K_FOLD_EPOCH) if K_FOLD else "_epochs-"+ str(EPOCHS)
print(ex_num_name)
logging.basicConfig(filename= './logs/' + ex_num_name + '.log',level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %I:%M:%S %p')

logging.info('Start training...' + ex_num_name)

writer = None
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

# # K-fold branch
if K_FOLD:
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits = K_FOLD_NUM, shuffle = True)

    EPOCHS = K_FOLD_EPOCH
    for fold, (train_ids, test_ids) in enumerate(k_fold.split(dataset)):
        print("fold: %d, number of dataset :%d" %(fold, len(train_ids)))

        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_sampler = torch.utils.data.SubsetRandomSampler(test_ids)

        dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, sampler = train_sampler)
        num_dataset = len(train_ids)
        testloader = DataLoader(dataset, batch_size = BATCH_SIZE, sampler = val_sampler)
        num_testset = len(test_ids)
        early_stop = train(model ,dataloader, device, loss, opt, EPOCHS, testloader, num_testset, ex_num_name, writer, fold)
        if early_stop:
            break

else:

    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_data, test_data = data.random_split(dataset, [n_train, n_val])

    dataloader = DataLoader(train_data, shuffle = True, batch_size = BATCH_SIZE)
    num_dataset = len(train_data)
    testloader = DataLoader(test_data, shuffle = True, batch_size = BATCH_SIZE)
    num_testset = len(test_data)

    train(model ,dataloader, device, loss, opt, EPOCHS, testloader, num_testset, ex_num_name, writer)

logging.info('Finished')

# if __name__ == "__main__" :
    # train()
    # evaluate(test_data)
