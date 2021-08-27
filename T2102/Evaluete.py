import torch
from sklearn.metrics import f1_score

def evaluate(model, loss, device, testloader, num_testset, cur_train_epoch = None, writer = None, record = False, final = False):
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

