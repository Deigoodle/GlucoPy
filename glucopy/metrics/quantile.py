# 3rd party
import pandas as pd

def quantile(df: pd.DataFrame,
             per_day: bool = False,
             q:float = 0.5,
             interpolation:str = 'linear',
             **kwargs
             ):
    '''
    Calculates the quantile of the CGM values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' and 'Day' columns present in
        :class:`glucopy.Gframe.data`.
    per_day : bool, default False
        If True, returns a pandas.Series with the quantile for each day. If False, returns the quantile for all
        days combined.
    q : float, default 0.5
        Value between 0 and 1 for the desired quantile.
    interpolation : str, default 'linear'
        This optional parameter specifies the interpolation method to use, when the desired quantile lies between 
        two data points i and j. Default is 'linear'.
    **kwargs : dict
        Additional keyword arguments to be passed to the function. For more information view the documentation for
        pandas.DataFrameGroupBy.quantile().

    Returns
    -------
    quantile : float | pandas.Series
        Quantile of the CGM values.

    Notes
    -----
    This function is meant to be used by :class:`glucopy.Gframe.quantile`
    '''
    if per_day:
        # Group data by day
        day_groups = df.groupby('Day')
        quantile = day_groups['CGM'].quantile(q=q, interpolation=interpolation, **kwargs)
    
    else:
        quantile = df['CGM'].quantile(q=q, interpolation=interpolation, **kwargs)
    
    return quantile