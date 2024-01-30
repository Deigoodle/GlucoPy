# 3rd party
import pandas as pd
import numpy as np

# Local
from glucopy.utils import time_factor

def mag(df: pd.DataFrame,
        time_unit: str = 'm'
        ):
    '''
    Calculates the Mean Absolute Glucose Change per unit of time (MAG).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' and 'Timestamp' columns present in
        :class:`glucopy.Gframe.data`.
    time_unit : str, default 'm' (minutes)
        The time time_unit for the x-axis. Can be 's (seconds)', 'm (minutes)', or 'h (hours)'.
    
    Returns
    -------
    mag : float
        Mean Absolute Glucose Change per unit of time.

    Notes
    -----
    This function is meant to be used by :class:`glucopy.Gframe.mag`
    '''
    # Determine the factor to multiply the total seconds by
    factor = time_factor(time_unit)
    
    # Calculate the difference between consecutive timestamps
    timeStamp_diff = pd.Series(np.diff(df['Timestamp']))

    # Calculate the difference between consecutive CGM values
    cgm_diff = pd.Series(np.abs(np.diff(df['CGM'])))

    # Calculate the MAG
    mag = np.sum(np.abs(cgm_diff)) / (timeStamp_diff.dt.total_seconds().sum()/factor)
        
    return mag