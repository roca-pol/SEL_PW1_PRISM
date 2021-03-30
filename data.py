import os
import pandas as pd


def load_hepatitis():
    path = os.path.join('datasets', 'hepatitis.csv')
    df = pd.read_csv(path)
    return df


def load_cmc():
    path = os.path.join('datasets', 'cmc.csv')
    df = pd.read_csv(path)
    df = df.astype(str)

    num_attr = ['Wifes_age', 'Number_of_children_ever_born']
    df[num_attr] = df[num_attr].astype(int)
    return df


def load_nursery():
    path = os.path.join('datasets', 'nursery.csv')
    df = pd.read_csv(path)
    return df
