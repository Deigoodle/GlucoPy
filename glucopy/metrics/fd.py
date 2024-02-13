# 3rd party
import pandas as pd
import numpy as np

def fd(df: pd.DataFrame,
       interval: list = [0,70,180],
       decimals: int = 2,
       count: bool = False
       ):
    '''
    Calculates the Frequency Distribution (FD) for a given target range of glucose.

    .. math::

        FD = \\frac{n_i}{N}

    - :math:`n_i` is the number of observations within the `i`-th interval.
    - :math:`N` is the total number of observations.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' column present in
        :attr:`glucopy.Gframe.data`.
    interval : list of int|float, default [0,70,180]
        Target range in CGM unit. It must have at least 2 values, for the "normal"
        range, low and high values will be values outside that range.
    decimals : int, default 2
        Number of decimal places to round to. Use None for no rounding.
    count : bool, default False
        If True, returns the count of observations for each range. If False, returns the percentage of observations

    Returns
    -------
    fd : pandas.Series 
        Series of Frequency Distribution for each range.

    Notes
    -----
    This function is meant to be used by :meth:`glucopy.Gframe.fd`
    '''
    # Check input, Ensure interval is a list or numpy array of numbers
    if not isinstance(interval, (list, np.ndarray)):
        raise ValueError("interval must be a list or numpy array of numbers")
    
    # Convert interval to a list if it's a numpy array
    if isinstance(interval, np.ndarray):
        interval = interval.tolist()
    
    # Add 0 to the target range if it is not present to count the time below the target range
    if 0 not in interval:
        interval = [0] + interval

    # Add the max value of the data to the target range if it is not present to count the time above the target range
    max_value = max(df['CGM'])
    if max_value <= interval[-1]:
        max_value = interval[-1] + 1
    if max_value > interval[-1]:
        interval = interval + [max_value]

    result = pd.cut(df['CGM'], bins=interval).groupby(pd.cut(df['CGM'], bins=interval), observed=False).count()

    if not count:
        result = result / result.sum()
    if decimals is not None:
        fd = (result).round(decimals=decimals)
    else:
        fd = result
        
    return fd