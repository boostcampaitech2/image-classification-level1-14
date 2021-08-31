import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import logging

def evaluate(model, loss, device, val_loader, num_val_set, cur_train_epoch = None, name = None, writer = None, record = False, final = False):
    model.eval()
    num_acc = 0

    y_true = []
    y_pred = []

    val_loss_sum = 0
    num_of_batches = len(val_loader)

    patience = 10
    counter = 0
    best_val_f1 = best_val_loss = -np.inf

    early_stop = False

    for batch_in, batch_out in tqdm(val_loader):
        with torch.no_grad():
            X = batch_in.to(device)
            Y = batch_out.to(device)
            
            logit = model(X)
            _, preds = torch.max(logit, 1)
            
            val_mini_match_loss = loss(logit, Y)
            val_loss_sum += val_mini_match_loss.item()

            num_acc += torch.sum(preds == Y.data).item()
            
            y_pred += preds.cpu().detach().numpy().tolist()
            y_true += Y.cpu().detach().numpy().tolist()
    
    val_loss = val_loss_sum / num_of_batches
    val_acc = num_acc / num_val_set
    val_f1_score = f1_score(y_true, y_pred, average = 'macro')

        # Callback1: validation accuracy가 향상될수록 모델을 저장합니다.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
    if val_f1_score > best_val_f1:
        logging.warning('New best model for val accuracy! saving the model..')
        print("New best model for val accuracy! saving the model..")
        torch.save(model.state_dict(), f"./checkpoints/{name}/{cur_train_epoch:03}_accuracy_{val_acc:4.2%}.ckpt")
        best_val_f1 = val_f1_score
        counter = 0
    else:
        counter += 1
    # Callback2: patience 횟수 동안 성능 향상이 없을 경우 학습을 종료시킵니다.
    if counter > patience:
        logging.warning('Early Stopping...')
        print("Early Stopping...")
        early_stop = True
        return early_stop, None

    if record == True:
        if writer == None:
            raise Exception('writer object is None')
        elif cur_train_epoch == None:
            raise Exception('cur epoch num is None')
            
        writer.add_scalar("loss/test loss", val_loss, cur_train_epoch)
        writer.add_scalar("acc/test acc", val_acc,  cur_train_epoch)
        writer.add_scalar("f1/test f1 score", val_f1_score,  cur_train_epoch)


    eval_result = {'loss':val_loss, 'acc':val_acc, 'f1':val_f1_score}

    return early_stop, eval_result

