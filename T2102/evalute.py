model.eval()
num_acc = 0

y_true = []
y_pred = []
# print(len(false_testset))
# print(len(concat_testloader))
# for idx, (batch_in, batch_out) in enumerate(tqdm(concat_testloader)):
for idx, (batch_in, batch_out) in enumerate(tqdm(DataLoader(concat_testset))):
    with torch.no_grad():
        X = batch_in.to(device)
        Y = batch_out.to(device)
        
        logit = model(X)
        _, preds = torch.max(logit, 1)
        
        num_acc += torch.sum(preds == Y.data)
        
        y_pred += preds.cpu().detach().numpy().tolist()
        y_true += Y.cpu().detach().numpy().tolist()
        
print("accuracy = ", num_acc.item() / len(concat_testset))
score = f1_score(y_true, y_pred, average = 'micro')
print("mirco average: ", score)
score = f1_score(y_true, y_pred, average = 'macro')
print("marco average: ", score)
# print(y_pred)
# print(y_true)