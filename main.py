import time
import numpy as np
import pandas as pd

from prism import Prism
from data import load_hepatitis, load_cmc, load_nursery

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import click
from click import Choice

import warnings
warnings.filterwarnings('ignore') 


@click.command()
@click.option('-d', '--dataset', type=Choice(['hepatitis', 'cmc', 'nursery']), default='hepatitis',
              help='Dataset name.')
@click.option('-s', '--seed', type=int, default=101, help='Random seed used when shuffling data before'
              ' splitting it into train and test sets.')
def main(dataset, seed):

    # load the corresponding dataset into a dataframe
    if dataset == 'hepatitis':
        df = load_hepatitis()
        target = 'class'
        n_bins = 3

    elif dataset == 'cmc':
        df = load_cmc()
        target = 'Contraceptive_method_used'
        n_bins = 3

    elif dataset == 'nursery':
        df = load_nursery()
        target = 'class'
        n_bins = None

    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=seed)
    df_train = df_train.copy()
    df_test = df_test.copy()

    # preprocessing - missing values
    num_attr = df.select_dtypes(include=['number']).columns
    means = df_train[num_attr].mean()
    df_train.fillna(means, inplace=True)
    df_test.fillna(means, inplace=True)

    categ_attr = df.select_dtypes(include=['category', object, 'bool']).columns
    df_train.replace('?', np.nan, inplace=True)
    df_test.replace('?', np.nan, inplace=True)
    modes = df_train[categ_attr].mode().iloc[0]
    df_train.fillna(modes, inplace=True)
    df_test.fillna(modes, inplace=True)

    prism = Prism()
    t0 = time.time()
    prism.fit(df_train, target=target, n_bins=n_bins)
    t1 = time.time()

    print(prism)
    print('\nCompute time:', round(t1 - t0, 2))

    y_pred = prism.predict(df_test)
    y_true = df_test[target]
    print('Accuracy:', round(accuracy_score(y_true, y_pred), 4))


if __name__ == '__main__':
    main()