# 3rd party
import numpy as np

FACTOR = 18.0

def mmoll_to_mgdl(mmoll: float | int | np.ndarray) -> float | int | np.ndarray:
    '''
    Convert mmol/L to mg/dL.

    Parameters
    ----------
    mmoll : float | int | np.ndarray
        mmol/L value to convert

    Returns
    -------
    mgdl : float | np.ndarray
        mg/dL value
    '''

    return mmoll * FACTOR