# 3rd party
import pandas as pd
import numpy as np

def dt(df: pd.DataFrame
       ):
    '''
    Calculates the Distance Travelled (DT) for each day.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' column present in
        :attr:`glucopy.Gframe.data`.

    Returns
    -------
    dt : float
        Distance Travelled of CGM.

    Notes
    -----
    This function is meant to be used by :meth:`glucopy.Gframe.dt`
    '''

    return np.sum(np.abs(np.diff(df['CGM'])))