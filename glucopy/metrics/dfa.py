# 3rd party
import pandas as pd
import neurokit2 as nk
import numpy as np

def dfa(df: pd.DataFrame,
        scale = 'default',
        overlap: bool = True,
        integrate: bool = True,
        order: int = 1,
        show: bool = False,
        **kwargs
        ):
    '''
    Calculates the Detrended Fluctuation Analysis (DFA) using neurokit2.fractal_dfa().

    For more information on the parameters and details see :py:func:`neurokit2.complexity.fractal_dfa`.

    Parameters
    ----------
    per_day : bool, default False
        If True, returns a :py:class:`pandas.Series` with the DFA for each day. If False, returns the DFA for the entire dataset. If
        a day has very few data points, the DFA for that day will be NaN.
    others: 
        For more information on the rest of the parameters see :py:func:`neurokit2.complexity.fractal_dfa`.

    Returns
    -------
    dfa : float
        Detrended fluctuation analysis.

    Notes
    -----
    This function is meant to be used by :meth:`glucopy.Gframe.dfa`
    '''
    try:
        dfa, _ = nk.fractal_dfa(df['CGM'].values,
                                scale=scale,
                                overlap=overlap,
                                integrate=integrate,
                                order=order,
                                show=show,
                                **kwargs)
    except:
        dfa = np.nan
        
    return dfa