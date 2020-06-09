import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def load_data(rootdir='./'):
    print('load data \n')
    x_train = np.loadtxt(rootdir + 'x_train.txt', dtype=str).astype(float)
    y_train = np.loadtxt(rootdir + 'y_train.txt', dtype=str).astype(int)

    print('x_train: [%d, %d], y_train:[%d,]' % (
        x_train.shape[0], x_train.shape[1], y_train.shape[0]))

    return x_train, y_train
