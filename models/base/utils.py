from attr import frozen
import numpy as np
from sklearn.metrics import precision_score
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_single_epoch_with_dataloader(model, device, dataloader: DataLoader,
                                       opt: Optimizer, loss_fn):
    """训练一个epoch
    Returns: 返回一个epoch产生的loss,np.float64类型
    """
    model.train()
    loss_per_epoch = []
    for batch_id, batch in enumerate(dataloader):
        user, item, rating = batch[0].to(device), batch[1].to(
            device), batch[2].to(device)
        y_real = rating.reshape(-1, 1)
        
        opt.zero_grad()
        y_pred = model(user, item)
        loss = loss_fn(y_pred, y_real)
        
        loss.backward()
        opt.step()
        loss_per_epoch.append(loss.item())

    return np.average(loss_per_epoch)


def freeze_specific_params(model,alive_layer_name):
    for name,param in model.named_parameters():
        if alive_layer_name in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # for name,param in model.named_parameters():

    #     if param.requires_grad:
    #         print(name,param.size())
    # raise Exception()

def unfreeze_params(model):
    for name,param in model.named_parameters():
        param.requires_grad = True


def train_mult_epochs_with_dataloader(epochs, *args, **kwargs):
    train_loss_list = []
    for epoch in range(epochs):
        loss_per_eopch = train_single_epoch_with_dataloader(*args, **kwargs)
        train_loss_list.append(loss_per_eopch)
    return np.average(train_loss_list), train_loss_list
