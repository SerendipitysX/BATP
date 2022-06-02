import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import pickle
import matplotlib.pyplot as plt
import tqdm


def euclidean_dist(o1, o2):
    '''
    Calculate the euclidean distance between two objects
    :param o1: [[x1,y1],[x2,y2],[x3,y3]...]
    :param o2:
    :return:
    '''
    dist = []
    for i in range(len(o1)):
        x1, y1 = o1[i][0], o1[i][1]
        x2, y2 = o2[i][0], o2[i][1]
        dist.append(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
    return np.mean(dist)


def knn_model(data, k, order, trip):
    data_c = data.copy()
    data_c.coord = data_c.coord.apply(lambda x: x[:order])
    # print(len(data.coord[0]))
    dist_all = []
    for i in range(len(data_c)):
        dist = euclidean_dist(data_c.coord[i], trip)
        dist_all.append(dist)
    dist_top_k = np.argsort(dist_all)[:k]
    # print(dist_top_k)
    return dist_top_k


def knn_predict(data, dist_top_k, order):
    predict_candidates = []
    for i in dist_top_k:
        predict_candidates.append(data.coord[i][order][1])
    return np.mean(predict_candidates)


l = [380,
 200,
 280,
 470,
 2890,
 780,
 750,
 560,
 560,
 320,
 280,
 240,
 1460,
 460,
 690,
 380,
 570,
 320,
 360,
 620,
 440,
 510,
 1330,
 1270,
 1320,
 550,
 350,
 380,
 410,
 340,
 640,
 220,
 360,
 410,
 780,
 140,
 70]
l = [0]+[int(i/10) for i in l]
l = np.cumsum(l)   # test: l[2:]
Filename = 'D:/A-bus/bus_pytorch/data/dataset/knn-dataset/train.csv'
data = pd.read_csv(Filename)
data['coord'] = data['coord'].apply(eval)
train, test = train_test_split(data, test_size=0.1, random_state=42)
# kf5 = KFold(n_splits=5, shuffle=False)

# for train_index, test_index in kf5.split(data):
#     train, test = data.iloc[train_index], data.iloc[test_index]
train, test = train.reset_index(drop=True), test.reset_index(drop=True)
predict_list = []
error_list = []
truth_list = []
for sample in tqdm.tqdm(range(len(test))):
    predict_sample = []
    error_sample = []
    truth_sample = []
    for i in range(2, len(l)):
        trip = test['coord'][sample]
        dist_top_k = knn_model(train, k=25, order=4*i, trip=trip[:4*i])
        predict = knn_predict(train, dist_top_k, 4*i)
        predict_sample.append(predict)
        error = np.abs(trip[4*i][1] - predict)/60
        error_sample.append(error)
        truth_sample.append(trip[4*i][1])
    predict_list.append(predict_sample)
    error_list.append(error_sample)
    truth_list.append(truth_sample)

with open('D:/A-bus/bus_pytorch/result/knn-predict.txt', 'wb') as handle:
    pickle.dump(predict_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('/result/knn-mae.txt', 'wb') as handle:
    pickle.dump(error_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('D:/A-bus/bus_pytorch/result/knn-truth.txt', 'wb') as handle:
    pickle.dump(truth_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)

error_list = np.array(error_list)
error_list = np.mean(error_list, axis=0)
# Make the plot
plt.stem(error_list, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.show()