import time
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
# from tensorflow.keras.layers.wrappers import *
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping


def build_model(input_timesteps, output_timesteps, num_links):
    model = Sequential()
    model.add(BatchNormalization(name='batch_norm_0', input_shape=(input_timesteps, num_links, 1, 1)))
    model.add(ConvLSTM2D(name='conv_lstm_1',
                         filters=64, kernel_size=(10, 1),
                         padding='same',
                         return_sequences=True))

    model.add(Dropout(0.21, name='dropout_1'))
    model.add(BatchNormalization(name='batch_norm_1'))

    model.add(ConvLSTM2D(name='conv_lstm_2',
                         filters=64, kernel_size=(preds, 1),
                         padding='same',
                         return_sequences=False))

    model.add(Dropout(0.20, name='dropout_2'))
    model.add(BatchNormalization(name='batch_norm_2'))

    model.add(Flatten())
    model.add(RepeatVector(output_timesteps))
    model.add(Reshape((output_timesteps, num_links, 1, 64)))

    model.add(ConvLSTM2D(name='conv_lstm_3',
                         filters=64, kernel_size=(10, 1),
                         padding='same',
                         return_sequences=True))

    model.add(Dropout(0.20, name='dropout_3'))
    model.add(BatchNormalization(name='batch_norm_3'))

    model.add(ConvLSTM2D(name='conv_lstm_4',
                         filters=64, kernel_size=(preds, 1),
                         padding='same',
                         return_sequences=True))

    model.add(TimeDistributed(Dense(units=1, name='dense_1', activation='relu')))
    # model.add(Dense(units=1, name = 'dense_2'))

    optimizer = RMSprop()  # lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.9)
    model.compile(loss="mse", optimizer=optimizer)
    return model

lags = 38
preds = 5
X_train, Y_train = [], []
train = pd.read_csv('D:/A-bus/BATP/data/dataset/1-00720-step/train.csv')
for _,i in enumerate(list(train.date.unique())):
    for _,j in enumerate(list(train.global_order.unique())):
        time_df = train.loc[(train.date==i)&(train.global_order==j), 'timestamp']
        X = np.stack([np.roll(time_df, i, axis = 0) for i in range(lags, 0, -1)], axis = 1)
        Y = np.stack([np.roll(time_df, -i, axis = 0) for i in range(0, preds, 1)], axis = 1)
        if i == min(train.date) and j == min(train.global_order):
            X_train,Y_train = X,Y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            Y_train = np.concatenate((Y_train, Y), axis=0)

X_val, Y_val = [], []
val = pd.read_csv('D:/A-bus/BATP/data/dataset/1-00720-step/val.csv')
for _,i in enumerate(list(val.date.unique())):
    for _,j in enumerate(list(val.global_order.unique())):
        time_df = val.loc[(val.date==i)&(val.global_order==j), 'timestamp']
        X = np.stack([np.roll(time_df, i, axis = 0) for i in range(lags, 0, -1)], axis = 1)
        Y = np.stack([np.roll(time_df, -i, axis = 0) for i in range(0, preds, 1)], axis = 1)
        if i == min(val.date) and j == min(val.global_order):
            X_val,Y_val = X,Y
        else:
            X_val = np.concatenate((X_val, X), axis=0)
            Y_val = np.concatenate((Y_val, Y), axis=0)

X_test, Y_test = [], []
test = pd.read_csv('D:/A-bus/BATP/data/dataset/1-00720-step/test.csv')
for _,i in enumerate(list(test.date.unique())):
    for _,j in enumerate(list(test.global_order.unique())):
        time_df = test.loc[(test.date==i)&(test.global_order==j), 'timestamp']
        X = np.stack([np.roll(time_df, i, axis = 0) for i in range(lags, 0, -1)], axis = 1)
        Y = np.stack([np.roll(time_df, -i, axis = 0) for i in range(0, preds, 1)], axis = 1)
        if i == min(test.date) and j == min(test.global_order):
            X_test,Y_test = X,Y
        else:
            X_test = np.concatenate((X_test, X), axis=0)
            Y_test = np.concatenate((Y_test, Y), axis=0)

X_train = X_train.reshape(-1,38,38)
X_train = X_train[:,:,:,np.newaxis,np.newaxis]
Y_train = Y_train.reshape(-1,5,38)
Y_train = Y_train[:,:,:,np.newaxis,np.newaxis]
X_test = X_test.reshape(-1,38,38)
X_test = X_test[:,:,:,np.newaxis,np.newaxis]
Y_test = Y_test.reshape(-1,5,38)
Y_test = Y_test[:,:,:,np.newaxis,np.newaxis]
X_val = X_val.reshape(-1,38,38)
X_val = X_val[:,:,:,np.newaxis,np.newaxis]
Y_val = Y_val.reshape(-1,5,38)
Y_val = Y_val[:,:,:,np.newaxis,np.newaxis]

print("TRAIN:", X_train.shape, Y_train.shape)
print("VAL:", X_val.shape, Y_val.shape)
print("TEST:", X_test.shape, Y_test.shape)

model = build_model(lags, preds, lags)

# Train
history = model.fit(X_train, Y_train,
                    batch_size=64, epochs=30,
                    shuffle=False, validation_data=(X_val, Y_val), verbose=2)
model.save('models/ConvLSTM.h5')

Y_rm_mean_test = 41950.91110048
Y_scale_test = 1.12781839e+08
Y_true = Y_test.squeeze() * Y_scale_test + Y_rm_mean_test
Y_naive = Y_rm_mean_test
Y_pred = model.predict(X_test).squeeze() * Y_scale_test + Y_rm_mean_test

# Y_true_total = np.sum(Y_true * Y_w_test, axis=2).squeeze()
# Y_naive_total = np.sum(Y_naive * Y_w_test, axis=2).squeeze()
# Y_pred_total = np.sum(Y_pred * Y_w_test, axis=2).squeeze()

for t in range(preds):
    # mask = Y_true[:, t, :] > 0
    # Y_true_total_t = Y_true[mask, t, :] / 60
    # Y_naive_total_t = Y_naive[mask, t, :] / 60
    # Y_pred_total_t = Y_pred[mask, t, :] / 60
    Y_true_total_t = Y_true[:, t, :] / 60
    Y_naive_total_t = Y_naive / 60
    Y_pred_total_t = Y_pred[:, t, :] / 60

    error_naive_total_t = (Y_naive_total_t - Y_true_total_t)
    error_lstm_total_t = (Y_pred_total_t - Y_true_total_t)

    mae_ha = np.mean(np.abs(error_naive_total_t))
    rmse_ha = np.sqrt(np.mean((error_naive_total_t) ** 2))
    mape_ha = np.mean(np.abs(error_naive_total_t) / Y_true_total_t) * 100

    mae_lstm = np.mean(np.abs(error_lstm_total_t))
    rmse_lstm = np.sqrt(np.mean((error_lstm_total_t) ** 2))
    mape_lstm = np.mean(np.abs(error_lstm_total_t) / Y_true_total_t) * 100

    print("- t + %d - HA       - MAE: %s - RMSE: %.2f - MAPE: %.2f" % (t + 1, mae_ha, rmse_ha, mape_ha))
    print("- t + %d - ConvLSTM - MAE: %s - RMSE: %.2f - MAPE: %.2f" % (t + 1, mae_lstm, rmse_lstm, mape_lstm))