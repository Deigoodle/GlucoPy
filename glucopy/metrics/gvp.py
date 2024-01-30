# 3rd party
import pandas as pd
import numpy as np

def gvp(df: pd.DataFrame
        ):
    '''
    Calculates the Glucose Variability Percentage (GVP), with time in minutes.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' and 'Timestamp' columns present in
        :class:`glucopy.Gframe.data`.

    Returns
    -------
    gvp : float
        Glucose Variability Percentage.

    Notes
    -----
    This function is meant to be used by :class:`glucopy.Gframe.gvp`
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