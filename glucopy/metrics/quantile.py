# 3rd party
import pandas as pd

def quantile(df: pd.DataFrame,
             per_day: bool = False,
             q:float = 0.5,
             interpolation:str = 'linear',
             ):
    '''
    Calculates the quantile of the CGM values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' and 'Day' columns present in
        :attr:`glucopy.Gframe.data`.
    per_day : bool, default False
        If True, returns a pandas.Series with the quantile for each day. If False, returns the quantile for all
        days combined.
    q : float, default 0.5
        Value between 0 and 1 for the desired quantile.
    interpolation : str, default 'linear'
        This optional parameter specifies the interpolation method to use, when the desired quantile lies between
        two data points i and j. Can be one of the following:

        - 'linear': i + (j - i) * fraction, where fraction is the fractional part of the index surrounded by i and j.
        - 'lower': i.
        - 'higher': j.
        - 'nearest': i or j, whichever is nearest.
        - 'midpoint': (i + j) / 2.

    Returns
    -------
    quantile : float | pandas.Series
        Quantile of the CGM values.

    Notes
    -----
    This function is meant to be used by :meth:`glucopy.Gframe.quantile`
    '''
    if per_day:
        # Group data by day
        day_groups = df.groupby('Day')
        quantile = day_groups['CGM'].quantile(q=q, interpolation=interpolation)

        # Convert the index to string for easier access
        quantile.index = quantile.index.map(str)

        # Rename the series
        quantile.name = f'Quantile {q}'
    
    else:
        quantile = df['CGM'].quantile(q=q, interpolation=interpolation)
    
    return quantile