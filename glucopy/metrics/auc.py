# 3rd party
import pandas as pd
import numpy as np

# Local
from glucopy.utils import time_factor

def auc(df: pd.DataFrame,
        time_unit='m',
        threshold: int | float = 0,
        above: bool = True
        ):
    '''
    Calculates the Area Under the Curve (AUC) using the trapezoidal rule.

    .. math::

        AUC = \\frac{1}{2} \\sum_{i=1}^{N} (X_i + X_{i-1}) * (T_i - T_{i-1})

    - :math:`X_i` is the glucose value at time i.
    - :math:`T_i` is the time at time i.
    - :math:`N` is the number of glucose readings.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' and 'Timestamp' columns present in
        :attr:`glucopy.Gframe.data`.
    time_unit : str, default 'm' (minutes)
        The time unit for the x-axis. Can be 's (seconds)', 'm (minutes)', or 'h (hours)'.
    threshold : int | float, default 0
        The threshold value above which the AUC will be calculated.
    above : bool, default True
        If True, the AUC will be calculated above the threshold. If False, the AUC will be calculated below the
        threshold.

    Returns
    -------
    auc : float
        Area Under the Curve (AUC).

    Notes
    -----
    This function is meant to be used by :meth:`glucopy.Gframe.auc`
    '''
    # Determine the factor to multiply the total seconds by
    factor = time_factor(time_unit)

    # Convert timestamps to the specified time unit
    time_values = (df['Timestamp'] - df['Timestamp'].min()).dt.total_seconds() / factor

    # Get the CGM values and set all values below or above the threshold to the threshold
    if above:
        cgm_values = np.maximum(df['CGM'], threshold) - threshold
    else:
        cgm_values = threshold - np.minimum(df['CGM'], threshold)

    auc = np.trapz(y = cgm_values, x = time_values)

    return auc