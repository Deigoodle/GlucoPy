# 3rd party
import pandas as pd
import numpy as np

def dt(df: pd.DataFrame
       ):
    '''
    Calculates the Distance Travelled (DT).

    .. math::

        DT = \\sum_{i=1}^{N-1} | X_{i+1} - X_i |

    - :math:`X_i` is the glucose value at time i.
    - :math:`N` is the number of glucose readings.

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