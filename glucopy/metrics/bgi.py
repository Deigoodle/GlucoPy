# 3rd party
import pandas as pd
import numpy as np

# Local
from glucopy.utils import mmoll_to_mgdl

def bgi(df: pd.DataFrame,
        unit: str = 'mg/dL',
        index_type:str = 'h',
        maximum: bool = False
        ):
    '''
    Calculates the Low Blood Glucose Index (LBGI) or the High Blood Glucose Index (LBGI).

    .. math::

        LBGI = \\frac{1}{N} \\sum_{i=1}^N rl(X_i)

    .. math::

        HBGI = \\frac{1}{N} \\sum_{i=1}^N rh(X_i)

    - :math:`N` is the number of glucose readings.
    - :math:`rl(X_i) = 22.77 * f(X_i)^2` if :math:`f(X_i) < 0` and :math:`0` otherwise.
    - :math:`rh(X_i) = 22.77 * f(X_i)^2` if :math:`f(X_i) > 0` and :math:`0` otherwise.
    - :math:`f(X_i) = 1.509 * (\\ln(X_i)^{1.084} - 5.381)` for glucose readings in mg/dL.
    - :math:`X_i` is the glucose value in mg/dL at time i.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' column present in
        :attr:`glucopy.Gframe.data`.
    unit : str, default 'mg/dL'
        Unit of the CGM values. Can be 'mg/dL' or 'mmol/L'.
    index_type : str, default 'h'
        Type of index to calculate. Can be 'h' (High Blood Glucose Index) or 'l' (Low Blood Glucose Index).
    maximum : bool, default False
        If True, returns the maximum LBGI or HBGI. If False, returns the mean LBGI or HBGI.

    Returns
    -------
    bgi : float 
        Low Blood Glucose Index (LBGI) or High Blood Glucose Index (HBGI).

    Notes
    -----
    This function is meant to be used by :meth:`glucopy.Gframe.bgi`
    '''
    # Check input
    index_type.lower()
    if index_type != 'h' and index_type != 'l':
        raise ValueError('index_type must be "h" or "l"')
    if unit != 'mg/dL' and unit != 'mmol/L':
        raise ValueError('unit must be "mg/dL" or "mmol/L"')
    
    def f(x):
        result = ( np.power(np.log(x), 1.084) - 5.381 ) * 1.509
        if result >= 0 and index_type == 'l':
            result = 0
        elif result <= 0 and index_type == 'h':
            result = 0
        return result

    values = df['CGM'].values
    if unit == 'mmol/L':
        values = mmoll_to_mgdl(values)

    f_values = np.vectorize(f,otypes=[float])(values)

    risk = 22.77 * np.square(f_values)
    
    if maximum:
        bgi = np.max(risk)
    else:
        bgi = np.mean(risk)

    return bgi