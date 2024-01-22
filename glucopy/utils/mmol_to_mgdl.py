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

    Examples
    --------
    Convert a single value

    .. ipython:: python

        import glucopy as gp
        gp.mmoll_to_mgdl(5.55)

    Convert a numpy array

    .. ipython:: python

        import numpy as np
        gp.mmoll_to_mgdl(np.array([5.55, 11.11, 16.66]))
    '''

    return mmoll * FACTOR