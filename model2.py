import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# param x: tensor, [bs, 29(his_days), 50(max_len)]

# == single model ==
def t2v(x, f, w0, b0, w, b):
    # One Non-periodic feature
    v1 = torch.matmul(x, w0) + b0
    # k-1 periodic features
    v2 = f(torch.matmul(x, w) + b)
    return torch.cat([v1, v2], dim=2)


class Time2Vec(nn.Module):
    '''
    Using sine from time point to vector, which consists of 2 parts:
    1. periodic 2.non-periodic
    x: bz, days, stations
    :return bz, stations, days, embeding_dim(out_features)
    '''
    def __init__(self, in_features, out_features, bz, stations,days):
        super(Time2Vec, self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(bz, days, in_features))
        self.b0 = nn.parameter.Parameter(torch.randn(bz, stations, in_features))
        self.w = nn.parameter.Parameter(torch.randn(bz, days, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(bz, stations, out_features - 1))
        self.f = torch.sin

    def forward(self, x):
        return t2v(x, self.f, self.w0, self.b0, self.w, self.b)


class Transformer(nn.Module):
    '''
    use Transformer to model the same stop in different days
    :x bz, days, stations
    '''
    def __init__(self, time2vector, in_features, out_features, bz, stations, days):
        super(Transformer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bz = bz
        self.stations = stations
        self.embedding = nn.Linear(days, out_features)
        self.T2V = time2vector(in_features, out_features, bz, stations, days)
        self.positional_embedding = self.T2V
        self.encoder = nn.TransformerEncoderLayer(d_model=out_features, nhead=8, dropout=0.1)
        self.decoder = nn.TransformerEncoder(self.encoder, num_layers=3)
        self.linear = nn.Linear(out_features, out_features*2)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in')

    def forward(self, x):
        # X [bz, days, stations]
        x = x.permute(0, 2, 1)
        x_input = self.embedding(x)                             # bz, stations, 29->d_models(out_features)
        x_pos = self.positional_embedding(x)                    # bz, stations, out_features
        x_time = x_input + x_pos                                # bz, stations, out_features
        x_out = torch.zeros_like(x_time)
        for i in range(self.stations):                          # self.stations = x_time.shape[1]
            x_feature = self.decoder(x_time[:, i, :])           # bz, stations, out_features
            x_out[:, i, :] = x_feature
        return self.linear(x_out)                               # bz, stations, out_features*2


class LSTM(nn.Module):
    def __init__(self, hidden_dim, out_dim, days):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        assert self.hidden_dim == self.out_dim
        self.lstm = nn.LSTM(self.hidden_dim, self.out_dim, num_layers=3, batch_first=True)
        self.linear = nn.Linear(self.out_dim, 1)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in')

    def forward(self, x):
        teacher_forcing_ratio = 0.5
        input = x[:, 0, :]
        out = torch.zeros_like(x)
        for i in range(x.shape[1]):
            out[:, i, :], _ = self.lstm(input)
            input = out[:, i, :] + (x[:, i, :] - out[:, i, :]) * teacher_forcing_ratio
        output = self.linear(out)
        return output


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class TemperalModel(nn.Module):
    def __init__(self, Transformer, time2vector, LSTM, bz, stations, days, in_features, out_features, hidden_dim, out_dim):
        super(TemperalModel, self).__init__()
        # self.T = Transformer(time2vector, in_features, out_features, bz, stations, days)
        self.embedding = nn.Linear(days, out_features)
        self.T = time2vector(in_features, out_features, bz, stations, days)
        self.L = LSTM(hidden_dim, out_dim, days)
        self.layer_norm = nn.LayerNorm(out_features)

    def count_parameters(self):
        T_params = sum(p.numel() for p in self.T.parameters() if p.requires_grad)
        L_params = sum(p.numel() for p in self.L.parameters() if p.requires_grad)
        return T_params+L_params

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.T(x)              # bz, stations, out_features
        x = self.layer_norm(x)
        x = self.L(x)
        return x


# x = torch.randn((16, 29, 50)).to(torch.float)  # (15,29,50) [bz, days, stations]
# # x1 = Time2Vec(in_features=1, out_features=128, bz=15, stations=50, days=29)(x)
# # t = Transformer(Time2Vec, 1, 8, 2, 3)
# # t_f = t(x1)
# # lstm = LSTM(16, 16, 4)
# # lstm_f = lstm(t_f)
# model = TemperalModel(Transformer, Time2Vec, LSTM, 16, 50, 29, 1, 128, 256, 256)
# print(model(x).shape)


