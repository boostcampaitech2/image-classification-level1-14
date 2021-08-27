def get_loss_and_opt(ex_num_name, device, model):
    import torch.optim as optim
    from utils import weigted_loss, get_class_weight
    import torch.nn as nn
    import torch
    if weigted_loss:
        weights = get_class_weight()
        weights = list(map(lambda x:x*5, weights))
        print("cross entropy weights : ", weights)
        ex_num_name += "_weighted_loss_true_mult_5"
        # print("weighted cross entropy: weights = ", ' '.join(str(w) for w in weights))
        loss = nn.CrossEntropyLoss(torch.tensor(weights).to(device))
    else:
        loss = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr = 1e-3,)

    return loss, opt, ex_num_name

if __name__ == "__main__":
    get_loss_and_opt()