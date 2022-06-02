import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import tqdm

with open('/result/knn-mae.txt', 'rb') as handle:
    error_list = pickle.load(handle)

# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)

error_list = np.array(error_list)
error_list = np.mean(error_list, axis=0)
my_range=range(1, len(error_list)+1)
# Make the plot
plt.stem(error_list, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.show()