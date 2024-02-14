# 3rd party
import pandas as pd

# Local
from .quantile import quantile

def iqr(df: pd.DataFrame,
        per_day: bool = False,
        interpolation:str = 'linear',
        **kwargs
        ):
    '''
    Calculates the Interquartile Range (IQR) of the CGM values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' and 'Day' columns present in
        :attr:`glucopy.Gframe.data`.
    per_day : bool, default False
        If True, returns a pandas.Series with the interquartile range for each day. If False, returns the
        interquartile range for the entire dataset.
    interpolation : str, default 'linear'
        This optional parameter specifies the interpolation method to use, when the desired quantile lies between
        two data points i and j. Default is 'linear'. 
    **kwargs : dict
        Additional keyword arguments to be passed to the function. For more information view the documentation for
        pandas.DataFrameGroupBy.quantile().

    Returns
    -------
    iqr : float | pandas.Series
        Interquartile range of the CGM values.

    Notes
    -----
    This function is meant to be used by :meth:`glucopy.Gframe.iqr`
    '''
        
    q1 = quantile(df=df, per_day=per_day, q=0.25, interpolation=interpolation, **kwargs)
    q3 = quantile(df=df, per_day=per_day, q=0.75, interpolation=interpolation, **kwargs)

    iqr = q3 - q1

    # Rename the series
    if per_day:
        iqr.name = 'Interquartile Range'
        
    return iqr