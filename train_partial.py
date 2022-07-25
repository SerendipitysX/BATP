import os
import random
import tqdm
import torch.optim as optim
import argparse
from data.dataset.data_utils import data_transform
from utils.Earlystopping import EarlyStopping
from model_partial import *
import pickle


# Parameter should change ####
# route: '1/00740'
# max_len
# d1 (15,38) [bz,stations]

parser = argparse.ArgumentParser(description='BUS')
parser.add_argument('--enable_cuda', type=bool, default='True', help='enable CUDA, default as True')
parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilize experiment results')
parser.add_argument('--kt', type=int, default=3, help='kenel size of temporal convolution')
parser.add_argument('--ks', type=int, default=3, help='kenel size of spatial convolution')
parser.add_argument('--epoches', type=int, default=200)
parser.add_argument('--step_size', type=int, default=10)
parser.add_argument('--patience', type=int, default=20, help='early stop,How long to wait after last time validation loss improved.')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--loss', type=str, default='mse', choices=['mse'])
parser.add_argument('--bz', type=int, default=15, help='batchsize')
parser.add_argument('--max_len', type=int, default=50, help='the max length of route')
parser.add_argument('--days', type=int, default=29)
parser.add_argument('--in_f', type=int, default=1)
parser.add_argument('--out_f', type=int, default=16)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--out_dim', type=int, default=32)
args = parser.parse_args()


# =============================== environment ================================
def set_env(seed):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False ## find fittest convolution
    torch.backends.cudnn.deterministic = True ## keep experiment result stable


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
set_env(seed=args.seed)

# =============================== dataset ================================
trainFile = 'D:/A-bus/bus_pytorch/data/dataset/1-M3723/train.txt'
valFile = 'D:/A-bus/bus_pytorch/data/dataset/1-M3723/val.txt'
testFile = 'D:/A-bus/bus_pytorch/data/dataset/1-M3723/test.txt'
train_iter, zero_value1, mean1, std1 = data_transform(trainFile, max_len=args.max_len, batch_size=args.bz)
val_iter, zero_value2, mean2, std2 = data_transform(valFile, max_len=args.max_len, batch_size=args.bz)
test_iter, zero_value3, mean3, std3 = data_transform(testFile, max_len=args.max_len, batch_size=args.bz)

# =============================== model ==============================
model = TemperalModel(Transformer, Time2Vec, LSTM, bz=args.bz, stations=args.max_len, days=args.days,
                      in_features=args.in_f, out_features=args.out_f, hidden_dim=args.hidden_dim, out_dim=args.out_dim).to(device)
print(model.count_parameters())

# ========================== optimizer & loss ========================
# optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
loss = nn.MSELoss()

# =============================== train ================================
def train(epochs, optimizer, scheduler, loss, early_stopping, model, train_iter, val_iter):
    min_val_loss = np.inf
    for epoch in range(epochs):
        model.train()
        l_sum = []
        for x, y in tqdm.tqdm(train_iter):
            # print(x.shape, y.shape) #(15,29,50) (15,1,50)
            y_pred = model(x, y.long().permute(0, 2, 1)).permute(0, 2, 1)  # [bz, days,stations/max_len]
            mask = torch.logical_not(y == zero_value1)
            y_pred_without_padding = torch.masked_select(y_pred, mask)
            y_without_padding = torch.masked_select(y, mask)
            # print(y.shape, y_pred.shape)
            # print(y_pred_without_padding.shape, y_without_padding.shape)
            l = loss(y_pred_without_padding, y_without_padding)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # print(l)
            l_sum.append(np.mean(l.item()))
        scheduler.step()
        val_loss, mae, d1 = val(model, val_iter)
        with open('D:/A-bus/bus_pytorch/result/tl-mae.txt', 'wb') as handle:
            pickle.dump(d1, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
        # early_stopping(val_loss, model)
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.6f} |Train loss: {:.6f} | Val loss: {:.6f} | MAE: {:.4f} |GPU occupy: {:.6f} MiB'. \
              format(epoch + 1, optimizer.param_groups[0]['lr'], np.mean(l_sum), val_loss, mae, gpu_mem_alloc))

        torch.save(model.state_dict(), 'D:/A-bus/bus_pytorch/checkpoint2/'+str(epoch+1)+'-'+str(mae)+'_model.pth')
        if early_stopping.early_stop:
            print("Early stopping.")
            break

    print('\nTraining finished.\n')



def val(model, val_loader, mean=mean2, std=std2):
    model.eval()
    l_sum = []
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        d1 = np.zeros((38))
        for x, y in val_loader:
            y_pred = model(x, y.long().permute(0, 2, 1)).permute(0, 2, 1)
            mask = torch.logical_not(y == zero_value2)
            y_pred_without_padding = torch.masked_select(y_pred, mask).cpu()
            y_without_padding = torch.masked_select(y, mask).cpu()
            l = loss(y_pred_without_padding, y_without_padding)
            l_sum.append(np.mean(l.item()))
            ### score
            y_without_padding = mean + std * y_without_padding
            y_pred_without_padding = mean + std * y_pred_without_padding
            d = np.abs(y_without_padding - y_pred_without_padding)/60
            mae += d.tolist()
            sum_y += y_without_padding.tolist()
            mape += (d / y_without_padding).tolist()
            mse += (d ** 2).tolist()
            d = d.reshape(15, -1).mean(axis=0)
            d1 = np.concatenate([d1, d]).mean(axis=0)
        MAE = np.array(mae).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))
        scheduler.step()
        return np.mean(l_sum), MAE, d1

def test(model_save_path, model, test_loader, score):
    model.load_state_dict(torch.load(model_save_path))


if __name__ == '__main__':
    set_env(args.seed)
    train(args.epoches, optimizer, scheduler, loss, EarlyStopping(patience=args.patience), model, train_iter, val_iter)
    # test('/home/xss/NER/BiLSTM/data/checkpoint/bilstm.pth', model, test_loader, score)