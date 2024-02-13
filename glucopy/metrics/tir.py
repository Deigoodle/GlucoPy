# 3rd party
import pandas as pd
import numpy as np

def tir(df: pd.DataFrame, 
        interval:list= [0,70,180],
        percentage: bool = True,
        decimals: int = 2
        ):
    '''
    Calculates the Time in Range (TIR) for a given target range of glucose.

    .. math::

        TIR = \\frac{\\tau}{T} * 100

    - :math:`\\tau` is the time spent within the target range.
    - :math:`T` is the total time of observation.

    Parameters
    ----------
    per_day : bool, default False
        If True, returns a pandas Series with the TIR for each day. If False, returns the TIR for all days combined.
    interval : list of int|float, default [0,70,180]
        Interval of glucose concentration to calculate :math:`\\tau`. Can be a list of 1 number, in that case the 
        time will be calculated below and above that number. It will always try to calculate the time below the first 
        number and above the last number.
    percentage : bool, default True
        If True, returns the TIR as a percentage. If False, returns the TIR as timedelta (:math:`TIR=\\tau`).
    decimals : int, default 2
        Number of decimal places to round to. Use None for no rounding.

    Returns
    -------
    tir : pandas.Series 
        Series of TIR.

    Notes
    -----
    This function is meant to be used by :meth:`glucopy.Gframe.tir`
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
                
    data_copy = df.copy()
    # Calculate time difference between consecutive timestamps
    data_copy['Time_Diff'] = data_copy['Timestamp'].diff().dt.total_seconds()
    # Create a column with the range that each CGM value belongs to
    data_copy['ranges'] = pd.cut(data_copy['CGM'], bins=interval)
    # Group data by range and sum the time difference
    time_count = data_copy.groupby('ranges', observed=False)['Time_Diff'].sum()

    if percentage:
        tir = time_count / time_count.sum() * 100
        if decimals is not None:
            tir = np.round(tir, decimals=decimals)

    else:
        tir = time_count.apply(lambda x: pd.to_timedelta(x, unit='s'))

    # Rename tir
    tir.name = 'Time in Range'
    
    return tir