# 3rd party
import pandas as pd

def data(dataset : str = 'prueba 1.csv'):
    '''
    Glucopy includes a few datasets to test the package, this function downloads
    and returns one of them

    The following datasets are available:
    * prueba 1.csv
    * prueba 2.csv

    Parameters
    ----------
    dataset : str, optional
        Name of the dataset to download, by default 'prueba 1.csv'

    Returns
    -------
    Gframe
        Gframe object
    '''
    dataset = dataset.lower()

    path = 'https://raw.githubusercontent.com/Deigoodle/GlucoPy/main/data/'

    return pd.read_csv(path + dataset)