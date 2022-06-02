import torch
import torch.nn as nn
import numpy as np

# param x: tensor, [bs, 29(his_days), 50(max_len)]

# == single model ==


def t2v(x, f, w0, b0, w, b):
    # One Non-periodic feature
    v1 = torch.mul(x, w0) + b0
    # k-1 periodic features
    v2 = f(torch.mul(x.permute(0, 2, 1).unsqueeze(-1), w) + b)
    return torch.cat([v1.permute(0, 2, 1).unsqueeze(-1), v2], dim=3)


class Time2Vec(nn.Module):
    '''
    Using sine from time point to vector, which consists of 2 parts:
    1. periodic 2.non-periodic
    x: bz, days, stations
    :return bz, stations, days, embeding_dim(out_features)
    '''
    def __init__(self, in_features, out_features, bz, stations):
        super(Time2Vec, self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(bz, in_features, stations))
        self.b0 = nn.parameter.Parameter(torch.randn(bz, in_features, stations))
        self.w = nn.parameter.Parameter(torch.randn(bz, stations, in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(bz, stations, in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, x):
        return t2v(x, self.f, self.w0, self.b0, self.w, self.b)


class Transformer(nn.Module):
    '''
    use Transformer to model the same stop in different days
    :x bz, days, stations
    '''
    def __init__(self, time2vector, in_features, out_features, bz, stations):
        super(Transformer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bz = bz
        self.stations = stations
        self.T2V = time2vector(in_features, out_features, bz, stations)
        # self.positional_embedding = nn.Embedding(bz*stations*30, out_features)
        self.encoder = nn.TransformerEncoderLayer(d_model=out_features, nhead=8, dropout=0.1)
        self.decoder = nn.TransformerEncoder(self.encoder, num_layers=2)
        self.linear = nn.Linear(out_features, out_features*2)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in')

    def forward(self, x):
        x_time = self.T2V(x)                                    # bz, stations, days, out_features
        # x_pos = self.positional_embedding(x1)                 # bz, stations, days, out_features
        x_out = torch.zeros_like(x_time)
        for i in range(self.stations):                          # self.stations = x_time.shape[1]
            x_feature = self.decoder(x_time[:, i, :, :])        # bz, stations, days, out_features
            x_out[:, i, :, :] = x_feature
        return self.linear(x_out)                               # bz, stations, days, out_features*2

class LSTM(nn.Module):
    def __init__(self, hidden_dim, out_dim, days):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        assert self.hidden_dim == self.out_dim
        self.lstm = nn.LSTM(self.hidden_dim, self.out_dim, batch_first=True)
        self.linear1 = nn.Linear(self.out_dim, 1)
        self.linear2 = nn.Linear(days, 1)
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in')

    def forward(self, x):
        teacher_forcing_ratio = 0.5
        input = x[:, 0, :, :]
        out = torch.zeros_like(x)
        for i in range(x.shape[1]):
            out[:, i, :], _ = self.lstm(input)
            input = out[:, i, :, :] + (x[:, i, :, :] - out[:, i, :, :]) * teacher_forcing_ratio
        output = self.linear1(out)
        output = self.linear2(output.squeeze(-1))
        return output

class TemperalModel(nn.Module):
    def __init__(self, Transformer, time2vector, LSTM, bz, stations, days, in_features, out_features, hidden_dim, out_dim):
        super(TemperalModel, self).__init__()
        self.T = Transformer(time2vector, in_features, out_features, bz, stations)
        self.L = LSTM(hidden_dim, out_dim, days)

    def count_parameters(self):
        T_params = sum(p.numel() for p in self.T.parameters() if p.requires_grad)
        L_params = sum(p.numel() for p in self.L.parameters() if p.requires_grad)
        return T_params+L_params

    def forward(self, x):
        x = self.T(x)
        x = self.L(x)
        return x


# x = torch.randn((15, 29, 50)).to(torch.int64)  # (15,29,50) [bz, days, stations]
# x1 = Time2Vec(in_features=1, out_features=16, bz=15, stations=50)(x)
# print(x1.shape)

# t = Transformer(Time2Vec, 1, 8, 2, 3)
# t_f = t(x1)
# lstm = LSTM(16, 16, 4)
# lstm_f = lstm(t_f)
# model = TemperalModel(Transformer, Time2Vec, LSTM, 2, 3, 4, 1, 8, 16, 16)
# print(model(x1).shape)


