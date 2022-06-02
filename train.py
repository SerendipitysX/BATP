import torch
import torch.utils as utils
import torch.nn as nn
import numpy as np
import os
import random
import tqdm
import torch.optim as optim
import argparse
from data.dataset.data_utils import data_transform
from utils.Earlystopping import EarlyStopping
from model1 import *
from torch.utils.tensorboard import SummaryWriter

layout = {
    'PLOT': {
        "loss": ["Multiline", ['Loss/train', 'Loss/val']],
        "Metrics": ["Multiline", ['Metrics/mae', 'Metrics/lr']],
    },
}
writer = SummaryWriter()
writer.add_custom_scalars(layout)

parser = argparse.ArgumentParser(description='BUS')
parser.add_argument('--enable_cuda', type=bool, default='True', help='enable CUDA, default as True')
parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilize experiment results')
parser.add_argument('--kt', type=int, default=3, help='kenel size of temporal convolution')
parser.add_argument('--ks', type=int, default=3, help='kenel size of spatial convolution')
parser.add_argument('--epoches', type=int, default=1000)
parser.add_argument('--step_size', type=int, default=10)
parser.add_argument('--patience', type=int, default=50, help='early stop,How long to wait after last time validation loss improved.')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--loss', type=str, default='mse', choices=['mse'])
parser.add_argument('--bz', type=int, default=16, help='batchsize')
parser.add_argument('--max_len', type=int, default=50, help='the max length of route')
parser.add_argument('--days', type=int, default=29)
parser.add_argument('--in_f', type=int, default=1)
parser.add_argument('--out_f', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--out_dim', type=int, default=256)
args = parser.parse_args()


# =============================== environment ================================
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
def set_env(seed):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False ## find fittest convolution
    torch.backends.cudnn.deterministic = True ## keep experiment result stable
set_env(seed=args.seed)

# =============================== dataset ================================
trainFile = 'D:/A-bus/bus_pytorch/data/dataset/partial-prediction/train.txt'
valFile = 'D:/A-bus/bus_pytorch/data/dataset/partial-prediction/val.txt'
testFile = 'D:/A-bus/bus_pytorch/data/dataset/partial-prediction/test.txt'
train_iter, zero_value1, mean1, std1 = data_transform(trainFile, max_len=args.max_len, batch_size=args.bz)
val_iter, zero_value2, mean2, std2 = data_transform(valFile, max_len=args.max_len, batch_size=args.bz)
test_iter, zero_value3, mean3, std3 = data_transform(testFile, max_len=args.max_len, batch_size=args.bz)


# =============================== model ==============================
model = TemperalModel(Transformer, Time2Vec, LSTM, bz=args.bz, stations=args.max_len, days=args.days,
                      in_features=args.in_f, out_features=args.out_f, hidden_dim=args.hidden_dim, out_dim=args.out_dim).to(device)
# model.load_state_dict(torch.load('D:/A-bus/bus_pytorch/checkpoint/302-9.452553033641857_model.pth'))
print(model.count_parameters())

# ========================== optimizer & loss ========================
# optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
scheduler2 = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.2)
scheduler3 = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
scheduler4 = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.2)
scheduler5 = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
scheduler6 = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.2)
scheduler7 = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
scheduler8 = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.2)
scheduler9 = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
scheduler10 = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.2)
scheduler11 = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
loss = nn.MSELoss()

# =============================== train ================================
def train(epochs, optimizer, scheduler, loss, early_stopping, model, train_iter, val_iter):
    min_val_loss = np.inf
    for epoch in range(epochs):
        model.train()
        l_sum = []
        for x, y in tqdm.tqdm(train_iter):
            # print(x.shape, y.shape,)
            y_pred = model(x).permute(0, 2, 1)
            mask = torch.logical_not(y == zero_value1)
            y_pred_without_padding = torch.masked_select(y_pred, mask)
            y_without_padding = torch.masked_select(y, mask)
            # print(y.shape, y_pred.shape)
            # print(y_pred_without_padding.shape, y_without_padding.shape)
            l = loss(y_pred_without_padding, y_without_padding)
            writer.add_scalar("Loss/train", np.mean(l.item()), epoch)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # print(l)
            l_sum.append(np.mean(l.item()))
        if epoch <= 100:
            scheduler1.step()
        elif epoch > 100 and epoch <= 200:
            scheduler2.step()
        else:
            scheduler3.step()
        val_loss, mae = val(model, val_iter, epoch=epoch)
        writer.add_scalar("Metrics/lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("Metrics/mae", mae, epoch)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        # early_stopping(mae, model)
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.6f} |Train loss: {:.6f} | Val loss: {:.6f} | MAE: {:.4f} |GPU occupy: {:.6f} MiB'. \
              format(epoch + 1, optimizer.param_groups[0]['lr'], np.mean(l_sum), val_loss, mae, gpu_mem_alloc))

        # torch.save(model.state_dict(), 'D:/A-bus/bus_pytorch/checkpoint/'+str(epoch+1)+'-'+str(mae)+'_model.pth')
        # if early_stopping.early_stop:
        #     print("Early stopping.")
        #     break

    print('\nTraining finished.\n')


def val(model, val_loader, mean=mean2, std=std2, epoch=0):
    model.eval()
    l_sum = []
    total_y_pred, total_y = [], []
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in val_loader:
            y_pred = model(x).permute(0, 2, 1)
            mask = torch.logical_not(y == zero_value2)
            y_pred_without_padding = torch.masked_select(y_pred, mask).cpu()
            y_without_padding = torch.masked_select(y, mask).cpu()
            l = loss(y_pred_without_padding, y_without_padding)
            l_sum.append(np.mean(l.item()))
            writer.add_scalar("Loss/val", np.mean(l.item()), epoch)
            ### score
            y_without_padding = mean + std * y_without_padding
            y_pred_without_padding = mean + std * y_pred_without_padding
            d = np.abs(y_without_padding - y_pred_without_padding)/60
            mae += d.tolist()
            sum_y += y_without_padding.tolist()
            mape += (d / y_without_padding).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))
        return np.mean(l_sum), MAE

def test(model_save_path, model, test_loader, score):
    model.load_state_dict(torch.load(model_save_path))


if __name__ == '__main__':
    set_env(args.seed)
    train(args.epoches, optimizer, scheduler1, loss, EarlyStopping(patience=args.patience), model, train_iter, val_iter)
    writer.flush()
    # test('/home/xss/NER/BiLSTM/data/checkpoint/bilstm.pth', model, test_loader, score)