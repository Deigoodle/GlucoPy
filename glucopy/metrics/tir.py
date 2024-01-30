# 3rd party
import pandas as pd
import numpy as np

def tir(df: pd.DataFrame, 
        target_range:list= [0,70,180],
        percentage: bool = True,
        decimals: int = 2
        ):
    '''
    Calculates the Time in Range (TIR) for a given target range of glucose.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' and 'Timestamp' columns present in
        :class:`glucopy.Gframe.data`.
    target_range : list of int|float, default [0,70,180]
        Target range in CGM unit for low, normal and high glycaemia. It must have at least 2 values, for the "normal"
        range, low and high values will be values outside that range.
    percentage : bool, default True
        If True, returns the TIR as a percentage. If False, returns the TIR as time.
    decimals : int, default 2
        Number of decimal places to round to. Use None for no rounding.

    Returns
    -------
    tir : pandas.Series 
        Series of TIR for each day.

    Notes
    -----
    This function is meant to be used by :class:`glucopy.Gframe.tir`
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
                
    data_copy = df.copy()
    # Calculate time difference between consecutive timestamps
    data_copy['Time_Diff'] = data_copy['Timestamp'].diff().dt.total_seconds()
    # Create a column with the range that each CGM value belongs to
    data_copy['ranges'] = pd.cut(data_copy['CGM'], bins=target_range)
    # Group data by range and sum the time difference
    time_count = data_copy.groupby('ranges', observed=False)['Time_Diff'].sum()

    if percentage:
        result = time_count / time_count.sum() * 100
        if decimals is not None:
            tir = np.round(result, decimals=decimals)

    else:
        tir = time_count.apply(lambda x: pd.to_timedelta(x, unit='s'))
    
    return tir