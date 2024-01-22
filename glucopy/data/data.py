# local
from ..io import read_csv

def data(dataset : str = 'prueba_1'):
    '''
    Glucopy includes a few datasets to test the package, this function downloads
    and returns one of them as a Gframe object.

    The following datasets are available:
    * prueba_1
    * prueba_2

    Parameters
    ----------
    dataset : str, default 'prueba_1'
        Name of the dataset to download

    Returns
    -------
    Gframe
        Gframe object

    Examples
    --------
    >>> import glucopy as gp
    >>> gf = gp.data('prueba_1')
    >>> gf.data.head()
                Timestamp         Day      Time    CGM
    0 2020-11-27 21:29:00  2020-11-27  21:29:00  235.0
    1 2020-11-27 21:44:00  2020-11-27  21:44:00  242.0
    2 2020-11-27 21:59:00  2020-11-27  21:59:00  257.0
    3 2020-11-27 22:14:00  2020-11-27  22:14:00  277.0
    4 2020-11-27 22:29:00  2020-11-27  22:29:00  299.0
    '''
    dataset = dataset.lower()

    path = 'https://raw.githubusercontent.com/Deigoodle/GlucoPy/main/data/'

    if dataset in ['prueba_1', 'prueba 1', 'prueba1','prueba_1.csv', 'prueba 1.csv', 'prueba1.csv']:
        path += 'prueba_1.csv'

    elif dataset in ['prueba_2', 'prueba 2', 'prueba2','prueba_2.csv', 'prueba 2.csv', 'prueba2.csv']:
        path += 'prueba_2.csv'

    return read_csv(path = path,
                    date_column='Sello de tiempo del dispositivo',
                    cgm_column='Historial de glucosa mg/dL',
                    skiprows=2,
                    date_format='%d-%m-%Y %H:%M',
                    )