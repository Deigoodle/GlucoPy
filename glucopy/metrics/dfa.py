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

    For more information on the parameters and details of the neurokit2.fractal_dfa() method, 
    see the neurokit2 documentation: 
    `neurokit2.fractal_dfa() <https://neuropsychology.github.io/NeuroKit/functions/complexity.html#neurokit2.complexity.fractal_dfa>`_.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' column present in
        :class:`glucopy.Gframe.data`.

    Returns
    -------
    dfa : float
        Detrended fluctuation analysis.

    Notes
    -----
    This function is meant to be used by :class:`glucopy.Gframe.dfa`
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