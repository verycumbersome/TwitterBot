import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec

sns.set()

import argparse

from utils import load_data
from decompose import PCA
from clustering import KMeans


def load_args():
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--pca', default=0, type=int,
                        help='set to 1 if we desire running pca, otherwise 0')
    parser.add_argument('--kmeans', default=1, type=int,
                        help='set to 1 if we desire running kmeans, otherwise 0')

    parser.add_argument('--pca_retain_ratio', default=.9, type=float)
    parser.add_argument('--kmeans_max_k', default=10, type=int)
    parser.add_argument('--kmeans_max_iter', default=20, type=int)
    parser.add_argument('--root_dir', default='../data/', type=str)
    args = parser.parse_args()

    return args


def plot_y_vs_x_list(y_vs_x, x_label, y_label, save_path):
    fld = os.path.join(args.root_dir, save_path)
    if not os.path.exists((fld)):
        os.mkdir(fld)

    plots_per_fig = 2

    ks_sses_keys = list(range(0, len(y_vs_x)))
    js = list(range(0, len(ks_sses_keys), plots_per_fig))

    for j in js:
        pp = ks_sses_keys[j:j + plots_per_fig]
        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(len(pp), 1, figure=fig)
        i = 0
        for k in pp:
            ax = fig.add_subplot(gs[i, :])
            ax.set_ylabel('%s (k=%d)' % (y_label, k))
            ax.set_xlabel(x_label)
            ax.plot(range(1, len(y_vs_x[k]) + 1), [x for x in y_vs_x[k]], linewidth=2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            i += 1

        fig.savefig(os.path.join(fld, '%d_%d.png' % (pp[0], pp[-1])))

    print('Saved at : %s' % fld)


def plot_y_vs_x(ys_vs_x, x_label, y_label, save_path):
    fld = os.path.join(args.root_dir, save_path)
    if not os.path.exists((fld)):
        os.mkdir(fld)

    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, :])
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.plot(range(1, len(ys_vs_x) + 1), ys_vs_x, linewidth=2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(os.path.join(fld, 'plot.png'))

    print('Saved at : %s' % fld)


def visualize(x_train, y_train):
    ##################################
    #      YOUR CODE GOES HERE       #
    ##################################
    fld = os.path.join(args.root_dir, 'outputs for part 2')
    if not os.path.exists((fld)):
        os.mkdir(fld)
    
    colors = {1: 'red', 2: 'green', 3: 'blue', 4: 'yellow', 5: 'orange', 6: 'cyan'}
    pc1 = x_train[:,0]
    pc2 = x_train[:,1]

    fig, ax = plt.subplots()
    fig.suptitle('2 component PCA')
    plt.xlabel('Principle component 1')
    plt.ylabel('Principle component 2')
    gs = gridspec.GridSpec(1, 1, figure=fig)
    
    for g in np.unique(y_train):
        ix = np.where(y_train == g)
        ax.scatter(pc1[ix], pc2[ix], c = colors[g], label = g, s = 5)
    ax.legend()
    
    fig.savefig(os.path.join(fld, 'plot.png'))
    print('Saved at : %s' % fld)

def apply_kmeans(do_pca, x_train, y_train, kmeans_max_iter, kmeans_max_k):
    print('kmeans\n')
    train_sses_vs_iter = []
    train_sses_vs_k = []
    train_purities_vs_k = []

    ##################################
    #      YOUR CODE GOES HERE       #
    ##################################

    for k in range(1, kmeans_max_k):
        sses = None
        avg_purity = 0.

        # do five tests to reduce effect of random start
        for i in range(5):
            kmeans = KMeans(k, kmeans_max_iter)
            sse = kmeans.fit(x_train)
            if (sses == None):
                sses = sse
            else:
                for j in range(len(sse)):
                    sses[j] = (sses[j] + sse[j])

            avg_purity += kmeans.get_purity(x_train, y_train)

        avg_purity = avg_purity / 5.

        for j in range(len(sses)):
            sses[j] = sses[j]/5.0
        # avg_sses = np.sum(np.array(sses), 0) / 5

        train_sses_vs_iter.append(sses)
        train_purities_vs_k.append(avg_purity)
        train_sses_vs_k.append(min(sses))


    plot_y_vs_x_list(train_sses_vs_iter, x_label='iter', y_label='sse',
                     save_path='plot_sse_vs_k_subplots_%d'%do_pca)
    plot_y_vs_x(train_sses_vs_k, x_label='k', y_label='sse',
                save_path='plot_sse_vs_k_%d'%do_pca)
    plot_y_vs_x(train_purities_vs_k, x_label='k', y_label='purities',
                save_path='plot_purity_vs_k_%d'%do_pca)



if __name__ == '__main__':
    args = load_args()
    x_train, y_train = load_data(args.root_dir)

    if args.pca == 1:
        pca = PCA(args.pca_retain_ratio)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        visualize(x_train, y_train)

    if args.kmeans == 1:
        apply_kmeans(args.pca, x_train, y_train, args.kmeans_max_iter, args.kmeans_max_k)

    print('Done')
