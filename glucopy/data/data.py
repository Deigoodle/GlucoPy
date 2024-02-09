# local
from ..io import read_csv

def data(dataset : str = 'prueba_1'):
    '''
    GlucoPy includes a few datasets to test the package, this function downloads
    and returns one of them as a Gframe object.

    The following datasets are available:
    
    - prueba_1
    - prueba_2

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
    .. ipython:: python

        import glucopy as gp
        gf = gp.data('prueba_1')
        gf
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