import configparser
config = configparser.ConfigParser()
config.read('config.cfg')

if not config.sections():
    print(config.sections())
    raise Exception


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

EX_NUM = config['trainer']['ex']
ex_num_name = "ex" + EX_NUM + "_dataset_size_" + str(DATASET_SIZE) + "banace_testset_" + str(BALANCE_TESTSET)


EPOCHS = int( config['trainer']['epoch'] )

config['trainer']['ex'] = str(int(config['trainer']['ex']) + 1)
with open('config.cfg', 'w') as configfile:
    config.write(configfile)





import utils
from dataset import MaskDataset
from utils import num_classes
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


train_data= test_data = None

if not BALANCE_TESTSET:
    labels, img_paths = utils.get_labels_and_img_paths(DATASET_SIZE)

    dataset = MaskDataset(img_paths, labels)
    train_data, test_data = utils.split_eval(dataset)
else:
    (false_labels, false_img_paths), (true_labels, true_img_paths) = utils.get_bal_labels_and_bal_img_paths(DATASET_SIZE)

    false_dataset = MaskDataset(false_img_paths, false_labels)
    true_dataset = MaskDataset(true_img_paths, true_labels)

    false_train_data, false_test_data = utils.split_eval(false_dataset)
    true_train_data, true_test_data = utils.split_eval(true_dataset)

    train_data = torch.utils.data.ConcatDataset([false_train_data, true_train_data])
    test_data = torch.utils.data.ConcatDataset([false_test_data, true_test_data])

dataloader = DataLoader(train_data, shuffle = True, batch_size = BATCH_SIZE)
num_dataset = len(train_data)
testloader = DataLoader(test_data, shuffle = True, batch_size = BATCH_SIZE)
num_testset = len(test_data)


from model import MaskModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MaskModel(num_classes).to(device)

import torch.optim as optim
loss = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr = 1e-3)


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
        running_acc = 0
        loss_sum = 0
        train_f1_score_sum = 0

        for idx, (batch_in, batch_out) in enumerate(tqdm(dataloader)):
            X = batch_in.to(device)
            Y = batch_out.to(device)
            
            logits = model(X)
            mini_batch_loss = loss(logits, Y)
            _, preds = torch.max(logits, 1)
            
            opt.zero_grad()
            mini_batch_loss.backward()
            opt.step()
            loss_sum += mini_batch_loss.item() # CrossEntLoss returns ave loss of mini batch
            
            running_acc += torch.sum(preds == Y.data).item()

            mini_batch_f1_score = f1_score(Y.cpu().detach().numpy().tolist(), preds.cpu().detach().numpy().tolist(), average = 'macro')

            train_f1_score_sum += mini_batch_f1_score

            if IS_LOG and idx % 1000 == 9:
                
                cur_epoch = e * num_dataset + idx

                train_f1_score = mini_batch_f1_score

                writer.add_scalar('training loss', 
                    mini_batch_loss.item(), cur_epoch)
                writer.add_scalar('training f1', 
                    mini_batch_f1_score, cur_epoch)
                
                evaluate(testloader, cur_epoch, writer, record = True)

            
        epoch_loss = loss_sum / num_of_batches
        train_f1_score = train_f1_score_sum / num_of_batches
        train_acc = running_acc/ num_dataset

        test_loss, test_acc, test_f1_score = evaluate(testloader)

        print("train loss : %-10.5f test loss = %-10.5f" %(epoch_loss, test_loss))
        print("train acc: %-10.5f test accuracy = %-10.5f" %(train_acc, test_acc))
        print("train  f1 mirco average: %-10.5f test  f1 mirco average: %-10.5f" %(test_f1_score, train_f1_score))

    torch.save(model, './models/'+str(EPOCHS)+ "_concat_"+'resnet_18.pt')
    print("epoch = %d done!" %(EPOCHS))
    writer.flush()
    writer.close()

def evaluate(testloader, cur_train_epoch = None, writer = None, record = False):
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
        writer.add_scalar("test loss", test_loss, cur_train_epoch)
        writer.add_scalar("test acc", test_acc,  cur_train_epoch)
        writer.add_scalar("test f1 score", test_f1_score,  cur_train_epoch)



    return test_loss, test_acc, test_f1_score

if __name__ == "__main__" :
    train()
    # evaluate(test_data)
