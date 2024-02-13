# 3rd party
import numpy as np

FACTOR = 18.0

def mgdl_to_mmoll(mgdl: float | int | np.ndarray):
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

    Examples
    --------
    Convert a single value

    .. ipython:: python

        import glucopy as gp
        gp.mgdl_to_mmoll(100)

    Convert a numpy array

    .. ipython:: python

        import numpy as np
        gp.mgdl_to_mmoll(np.array([100, 200, 300]))
    '''

    return mgdl / FACTOR

