# 3rd party
import pandas as pd
import neurokit2 as nk

def samp_en(df: pd.DataFrame,
            delay: int | None = 1,
            dimension: int | None = 2,
            tolerance: float | str | None = 'sd',
            **kwargs
            ):
    '''
    Calculates the Sample Entropy using neurokit2.entropy_sample()

    For more information on the parameters and details of the neurokit2.entropy_sample() method, 
    see the `neurokit2 documentation <https://neuropsychology.github.io/NeuroKit/functions/complexity.html#neurokit2.complexity.entropy_sample>`_.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' column present in
        :attr:`glucopy.Gframe.data`.
    
    Returns
    -------
    samp_en : float
        Entropy Sample.

    Notes
    -----
    This function is meant to be used by :meth:`glucopy.Gframe.samp_en`
    '''
    # Get glucose values
    signal = df['CGM'].values

    # Estimate optimal parameters for sample entropy
    if delay is None:
        delay, _  = nk.complexity_delay(signal)
    if dimension is None:
        dimension, _ = nk.complexity_dimension(signal,delay=delay)
    if tolerance is None:
        tolerance, _ = nk.complexity_tolerance(signal, delay=delay, dimension=dimension)

    # Calculate sample entropy
    samp_en, _ = nk.entropy_sample(signal, 
                                   delay=delay, 
                                   dimension=dimension, 
                                   tolerance=tolerance, 
                                   **kwargs)
    
    return samp_en