# 3rd party
import pandas as pd
import neurokit2 as nk
import numpy as np

def mse(df: pd.DataFrame,
        scale = 'default',
        dimension = 3,
        tolerance = 'sd',
        method = 'MSEn',
        show = False,
        **kwargs
        ):
    '''
    Calculates the Multiscale Sample Entropy using neurokit2.entropy_multiscale()

    For more information on the parameters and details of the neurokit2.entropy_sample() method, 
    see the neurokit2 documentation: 
    `neurokit2.entropy_multiscale() <https://neuropsychology.github.io/NeuroKit/functions/complexity.html#entropy-multiscale>`_.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' column present in
        :class:`glucopy.Gframe.data`.
        
    Returns
    -------
    mse : float
        Multiscale Sample Entropy.

    Notes
    -----
    This function is meant to be used by :class:`glucopy.Gframe.mse`
    '''
    # Get glucose values
    signal = df['CGM'].values

    # Estimate optimal parameters for sample entropy
    if dimension is None:
        dimension, _ = nk.complexity_dimension(signal)
    if tolerance is None:
        tolerance, _ = nk.complexity_tolerance(signal, dimension=dimension)

    # Calculate sample entropy
    try:
        with np.errstate(divide='ignore', invalid='ignore'): # ignore divide by zero warning
            mse, _ = nk.entropy_multiscale(signal, 
                                           scale=scale, 
                                           dimension=dimension, 
                                           tolerance=tolerance, 
                                           method=method, 
                                           show=show,
                                           **kwargs)  
    except:
        mse = np.nan      
        
    return mse