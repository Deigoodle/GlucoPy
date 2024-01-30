# 3rd party
import pandas as pd
import numpy as np

def fd(df: pd.DataFrame,
       target_range: list = [0,70,180],
       decimals: int = 2,
       count: bool = False
       ):
    '''
    Calculates the Frequency Distribution (FD) for a given target range of glucose.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' column present in
        :class:`glucopy.Gframe.data`.
    target_range : list of int|float, default [0,70,180]
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
    This function is meant to be used by :class:`glucopy.Gframe.fd`
    '''
    # Check input, Ensure target_range is a list with 0 and the max value of the data
    if not isinstance(target_range, list) or not all(isinstance(i, (int, float)) for i in target_range):
        raise ValueError("target_range must be a list of numbers")
    
    # Add 0 to the target range if it is not present to count the time below the target range
    if 0 not in target_range:
        target_range = [0] + target_range

    # Add the max value of the data to the target range if it is not present to count the time above the target range
    max_value = max(df['CGM'])
    if max_value <= target_range[-1]:
        max_value = target_range[-1] + 1
    if max_value > target_range[-1]:
        target_range = target_range + [max_value]

    result = pd.cut(df['CGM'], bins=target_range).groupby(pd.cut(df['CGM'], bins=target_range), observed=False).count()

    if not count:
        result = result / result.sum()
    if decimals is not None:
        fd = (result).round(decimals=decimals)
    else:
        fd = result
        
    return fd