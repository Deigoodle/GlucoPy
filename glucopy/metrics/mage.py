# 3rd party
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def mage(df: pd.DataFrame) -> float:
    '''
    Calculates the Mean Amplitude of Glycaemic Excursions (MAGE) for each day.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' column present in
        :class:`glucopy.Gframe.data`.
        
    Returns
    -------
    mage : float
        Mean Amplitude of Glycaemic Excursions (MAGE).

    Notes
    -----
    This function is meant to be used by :class:`glucopy.Gframe.mage`
    '''
    day_std = df['CGM'].std()
    
    # find peaks and nadirs
    peaks, _ = find_peaks(df['CGM'])
    nadirs, _ = find_peaks(-df['CGM'])

    if peaks.size > nadirs.size:
        nadirs = np.append(nadirs, df['CGM'].size - 1)
    elif peaks.size < nadirs.size:
        peaks = np.append(peaks, df['CGM'].size - 1)
    
    # calculate the difference between the peaks and the nadirs
    differences = np.abs(df['CGM'].iloc[peaks].values - df['CGM'].iloc[nadirs].values)

    # get differences greater than std
    differences = differences[differences > day_std]

    # calculate mage
    mage = differences.mean()

    return mage