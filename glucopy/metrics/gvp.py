# 3rd party
import pandas as pd
import numpy as np

def gvp(df: pd.DataFrame
        ):
    '''
    Calculates the Glucose Variability Percentage (GVP), with time in minutes.

    .. math::

        GVP = \\left( \\frac{L}{T_0} - 1\\right) * 100

    - :math:`L = \\sum_{i=1}^N \\sqrt{\\Delta X_i^2 + \\Delta T_i^2}`
    - :math:`T_0 = \\sum_{i=1}^N \\Delta T_i`
    - :math:`N` is the number of glucose readings.
    - :math:`\\Delta X_i` is the difference between glucose values at time i and i-1.
    - :math:`\\Delta T_i` is the difference between times at time i and i-1.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' and 'Timestamp' columns present in
        :attr:`glucopy.Gframe.data`.

    Returns
    -------
    gvp : float
        Glucose Variability Percentage.

    Notes
    -----
    This function is meant to be used by :meth:`glucopy.Gframe.gvp`
    '''
    # Calculate the difference between consecutive timestamps
    timeStamp_diff = pd.Series(np.diff(df['Timestamp']))

    # Calculate the difference between consecutive CGM values
    cgm_diff = pd.Series(np.diff(df['CGM']))

    line_length  = np.sum( np.sqrt( np.square(cgm_diff) \
                                  + np.square(timeStamp_diff.dt.total_seconds() / 60) ) )
    
    t0 = pd.Timedelta(df['Timestamp'].tail(1).values[0] \
                        -df['Timestamp'].head(1).values[0]).total_seconds() / 60
    
    gvp = (line_length/t0 - 1) * 100

    return gvp