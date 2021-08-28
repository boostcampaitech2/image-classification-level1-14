
from sklearn.metrics import f1_score

from Evaluete import evaluate




from utils import K_FOLD, K_FOLD_EPOCH, IS_LOG, BATCH_SIZE
from tqdm import tqdm
import torch

def train(model, dataloader, device, loss, opt, EPOCHS, testloader, num_testset,ex_num_name, fold = None ):
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
