import torch
import torch.utils as utils
import torch.nn.functional as F
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def read_data(filename):
    """
    Read data from a file.
    """
    with open(filename, 'r') as f:
        data = f.readlines()
    return data


def convert_str_to_list(line):
    """
    Convert string to list.
    """
    return np.array(json.loads(line))


def generate_data(datafile, max_len=50):
    """
    Generate data. txt --> list --> tensor(padding to max_len)
    return: [len(data), 30, max_len]
    """
    data = read_data(datafile)
    data_input = torch.zeros(len(data), 30, max_len)
    num = 0
    for line in data:
        line = convert_str_to_list(line)
        route_len = int(len(line) / 30)
        data_line = line.reshape(30, route_len)
        data_line = torch.from_numpy(data_line)
        data_line[25, :] = (data_line[24, :] + data_line[26, :])/2
        data_line = F.pad(input=data_line, pad=(0, max_len - route_len, 0, 0), mode='constant', value=0)
        data_input[num] = data_line
        num += 1
    return data_input


def z_score(x):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where $\mu$ and $\sigma$ are the mean and standard deviation of the data.
    '''
    mean, std = torch.std_mean(x)
    x_out = (x - mean) / std
    zero_value = torch.min(x_out)
    return x_out, mean, std, zero_value
    # scaler = StandardScaler()
    # scaled = scaler.fit_transform(x)
    # return scaled


def data_transform(datafile, max_len, batch_size):
    """
    produce data slices for x_data and y_data
    :param data: [len(data), 30, max_len]
    :return: x:[len(data), :29, max_len] , y:[len(data), 30, max_len]
    """
    data = generate_data(datafile, max_len)
    data, mean, std, zero_value = z_score(data)
    x = torch.zeros([len(data), 29, max_len])
    y = torch.zeros([len(data),  1, max_len])
    # mask_x = torch.zeros([len(data), 29, max_len])
    # mask_y = torch.zeros([len(data),  1, max_len])
    mask_x = torch.empty([len(data), 29, max_len])
    mask_y = torch.zeros([len(data), 1, max_len])

    for i in range(len(data)):
        x[i] = data[i][:29]   # [1,29,50]
        y[i] = data[i][29]    # [1,1,50]
        # a = torch.logical_not(x[i] == zero_value)
        mask_x[i] = torch.logical_not(x[i] == zero_value)
        mask_y[i] = torch.logical_not(y[i] == zero_value)

    # data = utils.data.TensorDataset(x.cuda(), y.cuda(), mask_x.cuda(), mask_y.cuda())
    data = utils.data.TensorDataset(x.cuda(), y.cuda())
    iter_data = utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, drop_last=True)
    return iter_data, zero_value, mean, std

# trainFile = 'D:/A-bus/bus_pytorch/data/dataset/train.txt'
# valFile = 'D:/A-bus/bus_pytorch/data/dataset/val.txt'
# testFile = 'D:/A-bus/bus_pytorch/data/dataset/test.txt'
# train_iter, zero_value, mean, std = data_transform(trainFile, max_len=50, batch_size=16)
# # val_iter = data_transform(valFile, max_len=50)
# # test_iter = data_transform(testFile, max_len=50)


# trainFile = 'D:/A-bus/bus_pytorch/data/dataset/1-M3723/train.txt'
# data = generate_data(trainFile, max_len=50)  # [len, 30, max_len]
# plt.plot(list(range(1, 31)), data[0, :, 0], label='1')
# plt.plot(list(range(1, 31)), data[1, :, 0], label='2')
# plt.plot(list(range(1, 31)), data[2, :, 0], label='3')
# plt.plot(list(range(1, 31)), data[3, :, 0], label='4')
# plt.plot(list(range(1, 31)), data[4, :, 0], label='5')
# plt.plot(list(range(1, 31)), data[5, :, 0], label='6')
# plt.plot(list(range(1, 31)), data[6, :, 0], label='7')
# plt.legend()
# plt.show()
