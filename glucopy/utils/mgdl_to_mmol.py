# 3rd party
import numpy as np

FACTOR = 18.0

def mgdl_to_mmoll(mgdl: float | int | np.ndarray) -> float | int | np.ndarray:
    '''
    Convert mg/dL to mmol/L.

    Parameters
    ----------
    mgdl : float | int | np.ndarray
        mg/dL value to convert

    Returns
    -------
    mmoll : float | np.ndarray
        mmol/L value
    '''

    return mgdl / FACTOR

