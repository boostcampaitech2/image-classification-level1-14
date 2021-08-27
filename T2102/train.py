
import utils
from dataset import MaskDataset
from utils import num_classes
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import config
from utils import num_unfreeze_ratio
# torch.manual_seed(42)
# torch.cuda.empty_cache()


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


DATASET_SIZE = config['dataset']['data_size']
BATCH_SIZE = int(config['dataset']['batch_size'])



EPOCHS = int( config['trainer']['epoch'] )

EX_NUM = config['trainer']['ex']
ex_num_name = "ex" + EX_NUM +"_num_unfreeze_ratio_"+str(num_unfreeze_ratio)+"_epochs_"+ str(EPOCHS) +"_dataset_size_" + str(DATASET_SIZE) + "_banace_testset_" + str(BALANCE_TESTSET)

config['trainer']['ex'] = str(int(config['trainer']['ex']) + 1)
with open('config.cfg', 'w') as configfile:
    config.write(configfile)





from utils import BASE_DATASET

train_data= test_data = None
if BASE_DATASET:
####################################################################################################
    from utils import img_dir

    from basedataset import get_transforms, MaskBaseDataset,mean, std
    import torch.utils.data as data


    # 정의한 Augmentation 함수와 Dataset 클래스 객체를 생성합니다.
    # transform = get_transforms(mean=mean, std=std)

    dataset = MaskBaseDataset(
        img_dir=img_dir
    )

    # train dataset과 validation dataset을 8:2 비율로 나눕니다.
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_data, test_data = data.random_split(dataset, [n_train, n_val])

    # 각 dataset에 augmentation 함수를 설정합니다.
    # train_data.dataset.set_transform(transform['train'])
    # test_data.dataset.set_transform(transform['val'])
    ex_num_name += "_base_dataset"

####################################################################################################
else:
    if not BALANCE_TESTSET:
        labels, img_paths = utils.get_labels_and_img_paths(DATASET_SIZE)

        data = MaskDataset(img_paths, labels)

        train_data, test_data = utils.split_eval(data)
        print("train data : %d test_data : %d" %(len(train_data), len(test_data)))
    else:
        (true_labels, true_img_paths), (false_labels, false_img_paths) = utils.get_bal_labels_and_bal_img_paths(DATASET_SIZE)

        false_dataset = MaskDataset(false_img_paths, false_labels)
        true_dataset = MaskDataset(true_img_paths, true_labels)

        each_test_size = 500
        false_train_data, false_test_data = utils.split_eval(false_dataset,BALANCE_TESTSET, size = each_test_size)
        true_train_data, true_test_data = utils.split_eval(true_dataset, BALANCE_TESTSET, size = each_test_size)

        train_data = torch.utils.data.ConcatDataset([false_train_data, true_train_data])
        test_data = torch.utils.data.ConcatDataset([false_test_data, true_test_data])
        ####################################################################################################
        # ex_num_name += "_only negative trainset"

        # test_data = false_test_data
        ####################################################################################################
        print("train data : %d test_data : %d" %(len(train_data), len(test_data)))

        false = [i for i in range(6,18)]
        count_f = 0
        count_t = 0

        for t in test_data:
            if t[1] in false:
                count_f +=1
            else:
                count_t +=1 
        print("testdata balance stat\nmask data: %dn normal and incorrect data %d" %(count_t, count_f))
        if count_t != count_f:
            print("The test dataset is not balance. This may be woring. Proceed?")
            while True:
                ans = input("type yes: ")
                if ans == "yes":
                    break


dataloader = DataLoader(train_data, shuffle = True, batch_size = BATCH_SIZE)
num_dataset = len(train_data)
testloader = DataLoader(test_data, shuffle = True, batch_size = BATCH_SIZE)
num_testset = len(test_data)



from model import MaskModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MaskModel(num_classes).to(device)

import torch.optim as optim

from utils import weigted_loss, get_class_weight
if weigted_loss:
    weights = get_class_weight()
    weights = list(map(lambda x:x*5, weights))
    print(weights)
    ex_num_name += "_weighted_loss_true_mult_5"
    # print("weighted cross entropy: weights = ", ' '.join(str(w) for w in weights))
    loss = nn.CrossEntropyLoss(torch.tensor(weights).to(device))
else:
    loss = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr = 1e-3,)


from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score


def train():
    print("#"*10)
    if not IS_LOG:
        print("NOT RECORDING IN TENSORBOARD. ")
        while True:
            confirm = input("type yes: ")
            if confirm == "yes":
                break
        
    print("#"*10)


    print(ex_num_name)
    model.train()
    model.init_params()

    num_of_batches = len(dataloader)


    writer = SummaryWriter("app/" + ex_num_name)

    for e in range(EPOCHS):
        print("\nepoch number :", e)
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

            
        epoch_loss = loss_sum / num_of_batches
        train_f1_score = train_f1_score_sum / num_of_batches
        train_acc = running_acc_sum/ num_of_batches

        test_loss, test_acc, test_f1_score = evaluate(testloader,e+1, writer, record = IS_LOG)

        print("train loss : %-10.5f test loss = %-10.5f" %(epoch_loss, test_loss))
        print("train acc: %-10.5f test accuracy = %-10.5f" %(train_acc, test_acc))
        print("train  f1 marco average: %-10.5f test  f1 marco average: %-10.5f" %(train_f1_score, test_f1_score))


        if IS_LOG:
            writer.add_scalar('loss/training loss', 
                epoch_loss, e+1)
            writer.add_scalar('f1/training f1', 
                train_f1_score, e+1)
            writer.add_scalar('acc/training acc', 
                train_acc, e+1)


    torch.save(model, './models/'+ex_num_name+'resnet_18.pt')
    print("epoch = %d done!" %(EPOCHS))
    writer.flush()
    writer.close()

def evaluate(testloader, cur_train_epoch = None, writer = None, record = False, final = False):
    model.eval()
    num_acc = 0

    y_true = []
    y_pred = []

    test_loss_sum = 0
    num_of_batches = len(testloader)

    for batch_in, batch_out in testloader:
        with torch.no_grad():
            X = batch_in.to(device)
            Y = batch_out.to(device)
            
            logit = model(X)
            _, preds = torch.max(logit, 1)
            
            test_mini_match_loss = loss(logit, Y)
            test_loss_sum += test_mini_match_loss.item()

            num_acc += torch.sum(preds == Y.data).item()
            
            y_pred += preds.cpu().detach().numpy().tolist()
            y_true += Y.cpu().detach().numpy().tolist()
    
    test_loss = test_loss_sum / num_of_batches
    test_acc = num_acc / num_testset
    test_f1_score = f1_score(y_true, y_pred, average = 'macro')

    if record == True:
        if writer == None or cur_train_epoch == None:
            raise Exception
        writer.add_scalar("loss/test loss", test_loss, cur_train_epoch)
        writer.add_scalar("acc/test acc", test_acc,  cur_train_epoch)
        writer.add_scalar("f1/test f1 score", test_f1_score,  cur_train_epoch)



    return test_loss, test_acc, test_f1_score





if __name__ == "__main__" :
    train()
    # evaluate(test_data)
