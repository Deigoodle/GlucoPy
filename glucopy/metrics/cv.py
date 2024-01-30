# 3rd party
import pandas as pd

# Local
from .std import std
from .mean import mean

def cv(df: pd.DataFrame,
       per_day: bool = False,
       ddof: int = 1,
       **kwargs
       ):
    '''
    Calculates the Coefficient of Variation (CV) of the CGM values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' and 'Day' columns present in
        :attr:`glucopy.Gframe.data`.
    per_day : bool, default False
        If True, returns the an array with the coefficient of variation for each day. If False, returns
        the coefficient of variation for all days combined.
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of 
        elements. By default ddof is 1.
    **kwargs : dict
        Additional keyword arguments to be passed to the function. For more information view the documentation for
        pandas.DataFrameGroupBy.mean() and std().

    Returns
    -------
    cv : float | pandas.Series
        Coefficient of variation of the CGM values.    

    Notes
    -----
    This function is meant to be used by :meth:`glucopy.Gframe.cv`
    '''

    return std(df=df, per_day=per_day, ddof=ddof, **kwargs) / mean(df=df, per_day=per_day, **kwargs)