import pandas as pd
import os
import torch.utils.data

import configparser

## configparser load
config = configparser.ConfigParser()
config.read('config.cfg')
if not config.sections():
    raise Exception('config file is missing')

## num class
num_classes = 18

## unfreeze of pretrained
num_unfreeze_ratio = int( config['trainer']['unfreeze'].split('/')[-1] )
if num_unfreeze_ratio == 0:
    num_unfreeze_ratio = float('inf')

## directory
img_dir = '/opt/ml/input/data/train/images'
info_dir = '/opt/ml/input/data/train/train.csv'
base_model_dir = '/opt/ml/code/base_models/'


## weight loss
weigted_loss = None
if config['loss']['weight_loss'] == "True":
    weigted_loss = True

elif config['loss']['weight_loss'] == "False":
    weigted_loss = False
else:
    raise Exception('weight loss config must be True or False')

## base_dataset
if config['dataset']['base_dataset'] == "True":
    BASE_DATASET = True

elif config['dataset']['base_dataset'] == "False":
    BASE_DATASET = False


IS_LOG = config['log']['tensorboard']
if IS_LOG == "False":
    IS_LOG = False
elif IS_LOG == "True":
    IS_LOG = True
else:
    raise Exception("is log optin only True of False")



BALANCE_TESTSET = config['dataset']['balance_testset']
if BALANCE_TESTSET == "False":
    BALANCE_TESTSET = False
elif BALANCE_TESTSET == "True":
    BALANCE_TESTSET = True
else:
    raise Exception("balance testset optin only True of False")

## kfold
if config['dataset']['k-fold'] == "True":
    K_FOLD = True

elif config['dataset']['k-fold'] == "False":
    K_FOLD = False

if config['dataset']['k-fold-num'].isdigit():
    K_FOLD_NUM = int(config['dataset']['k-fold-num'])
else:
    raise Exception("k-fold-num conig must be integer")

if config['dataset']['k-fold-epoch'].isdigit():
    K_FOLD_EPOCH = int(config['dataset']['k-fold-epoch'])
else:
    raise Exception("k-fold-epoch conig must be integer")
## data augmentation

# model select
model_list = config['model']['model'].split(',')
model_index = int(config['model']['index'])

def get_person_dir():
    info = pd.read_csv(info_dir)
    return info['path']



def create_classes(dict):
    if dict['mask'] == 0:#wear
        if dict['gen'] == 'male':
            if dict['age'] < 30:
                return 0
            elif 30 <= dict['age'] < 60:
                return 1
            else:
                return 2
        else:
            if dict['age'] < 30:
                return 3
            elif 30 <= dict['age'] < 60:
                return 4
            else:
                return 5
    elif dict['mask'] == 1:#incorrect
        if dict['gen'] == 'male':
            if dict['age'] < 30:
                return 6
            elif 30 <= dict['age'] < 60:
                return 7
            else:
                return 8
        else:
            if dict['age'] < 30:
                return 9
            elif 30 <= dict['age'] < 60:
                return 10
            else:
                return 11
    elif dict['mask'] == 2:#normal
        if dict['gen'] == 'male':
            if dict['age'] < 30:
                return 12
            elif 30 <= dict['age'] < 60:
                return 13
            else:
                return 14
        else:
            if dict['age'] < 30:
                return 15
            elif 30 <= dict['age'] < 60:
                return 16
            else:
                return 17




def get_labels_and_img_paths(size):


    mask_list = create_mask_list()


    size = config['dataset']['data_size']
    if size == 'all':
        return get_all_labels_and_img_paths(mask_list)
    elif size.isdigit():
        return get_mini_labels_mini_img_paths(mask_list, int(size) )
    else:
        raise Exception('size is not integer number')



def create_mask_list(balance_testset = False):

    all_masks = []
    for i in range(5):
        all_masks.append('mask' + str(i+1))
    
    not_masks = ['incorrect_mask', 'normal']

    mask_list = []
    if balance_testset == False:
        mask_list = all_masks + not_masks
        return mask_list
    else:
        return all_masks, not_masks

def get_bal_labels_and_bal_img_paths(size):
    all_masks, not_masks = create_mask_list(balance_testset = True)
    
    true_l_and_p = false_l_and_p = None
    if size == 'all':
        true_l_and_p = get_all_labels_and_img_paths(all_masks)
        false_l_and_p = get_all_labels_and_img_paths(not_masks)
    elif size.isdigit():
        size = int(size)
        true_l_and_p = get_mini_labels_mini_img_paths(all_masks, size )
        false_l_and_p = get_mini_labels_mini_img_paths(not_masks, size )
    else:
        raise Exception('size is not integer number')
    return true_l_and_p , false_l_and_p


def get_mini_labels_mini_img_paths(mask_list, size):
    labels, img_paths = get_all_labels_and_img_paths(mask_list)
    assert len(labels) > size, "Size is bigger than the data size."

    return labels[:size], img_paths[:size]

def get_all_labels_and_img_paths(mask_list):
    img_paths = []
    labels = []


    person_dir = get_person_dir()
    for p_dir in person_dir:

        img_paths += create_img_paths(mask_list, img_dir, p_dir)
        labels += create_labels(mask_list, p_dir)

    return labels, img_paths



def create_labels(mask_list, p_dir):
        split_str = p_dir.split('_')
        gender = split_str[1]
        age = split_str[3]

        classes = []

        for ms_idx, ms in enumerate(mask_list):
            label_stat = {}
            if ms == "incorrect_mask":# incorrect
                label_stat['mask'] = 1
            elif ms == "normal":#normal
                label_stat['mask'] = 2
            else:# wear
                label_stat['mask'] = 0
            
            label_stat['age'] = int(age)
            label_stat['gen'] = gender
            classes.append(create_classes(label_stat))
        return classes




def create_img_paths(mask_list, img_dir, p_dir):
    img_path = []
    for ms in mask_list:
        img_path.append(os.path.join(img_dir, p_dir, ms))
    return img_path



def split_eval(dataset, balance_testset = False, size = None):
    """
        return dataset, testset
    """

    validate_num = len(dataset) // 10
    if balance_testset:
        if size == None:
            raise Exception("utils.split_eval size must be integer.")
        validate_num = size

    return torch.utils.data.random_split(dataset, [ len(dataset) - validate_num, validate_num ], generator=torch.Generator().manual_seed(42))



def get_class_weight(size = 'all'):
    # data = get_labels_and_img_paths(size)
    # labels, paths = data
    # df = pd.DataFrame({"labels" : labels, "paths":paths})
    # label_counts = df.labels.value_counts()
    # target = label_counts.index
    # total = label_counts.sum()

    # wj=n_samples / (n_classes * n_samplesj)

    # w = []
    # for i in label_counts:
    #     w.append(total / (18 * i))
    # weighted = pd.DataFrame(w)
    # weights = weighted.set_index(target).sort_index().values.tolist()

    data = get_labels_and_img_paths(size)
    labels, paths = data
    df = pd.DataFrame({"labels" : labels, "paths":paths})
    label_counts = df.labels.value_counts()
    target = label_counts.index
    total = label_counts.sum()

    # wj=n_samples / (n_classes * n_samplesj)

    w = []
    for i in label_counts:
        w.append(total / (18 * i))

    weighted = pd.DataFrame({'label':target, 'w':w})
    weighted = weighted.set_index('label')
    weighted = weighted.sort_index()
    weights = weighted.w.tolist()    
        

    return weights
