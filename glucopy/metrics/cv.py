# 3rd party
import pandas as pd

# Local
from .std import std
from .mean import mean

def cv(df: pd.DataFrame,
       per_day: bool = False,
       ddof: int = 1,
       ):
    '''
    Calculates the Coefficient of Variation (CV) of the CGM values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' and 'Day' columns present in
        :attr:`glucopy.Gframe.data`.
    per_day : bool, default False
        If True, returns a :py:class:`pandas.Series` with the coefficient of variation for each day. If False, returns
        the coefficient of variation for the entire dataset.
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of 
        elements. By default ddof is 1.

    Returns
    -------
    cv : float | pandas.Series
        Coefficient of variation of the CGM values.    

    Notes
    -----
    This function is meant to be used by :meth:`glucopy.Gframe.cv`
    '''
    cv = std(df=df, per_day=per_day, ddof=ddof) / mean(df=df, per_day=per_day)

    # Rename the series
    if per_day:
        cv.name = 'Coefficient of Variation'

    return cv