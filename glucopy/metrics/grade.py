# 3rd party
import pandas as pd
import numpy as np

# Local
from glucopy.utils import mgdl_to_mmoll

def grade(df: pd.DataFrame,
          percentage: bool = True,
          unit: str = 'mg/dL'
          ):
    '''
    Calculates the contributions of the Glycaemic Risk Assessment Diabetes Equation (GRADE) to Hypoglycaemia,
    Euglycaemia and Hyperglycaemia. Or the GRADE scores for each value.

    .. math::

        GRADE = 425 * [\\log_{10}(\\log_{10} (X_i) + 0.16)]^2

    - :math:`X_i` is the glucose value in mmol/L at time i.

    The GRADE contribution percentages are calculated as follows:

    .. math::
        :nowrap:

        \\begin{align*}
        \\text{Hypoglycaemia %} &= 100 * \\frac{\\sum \\text{GRADE}(X_i < 3.9 [\\text{mmol/L}])}{\\sum \\text{GRADE}(X_i)} \\\\
        \\text{Euglycaemia %} &= 100 * \\frac{\\sum \\text{GRADE}(3.9 [\\text{mmol/L}] \\leq X_i \\leq 7.8 [\\text{mmol/L}])}{\\sum \\text{GRADE}(X_i)} \\\\
        \\text{Hyperglycaemia %} &= 100 * \\frac{\\sum \\text{GRADE}(X_i > 7.8 [\\text{mmol/L}])}{\\sum \\text{GRADE}(X_i)}
        \\end{align*}

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' column present in
        :attr:`glucopy.Gframe.data`.
    percentage : bool, default True
        If True, returns a pandas.Series of GRADE score contribution percentage for Hypoglycaemia, Euglycaemia and 
        Hyperglycaemia. If False, returns a list of GRADE scores for each value.
    unit : str, default 'mg/dL'
        Unit of the CGM values. Can be 'mg/dL' or 'mmol/L'.
    

    Returns
    -------
    grade : pandas.Series | numpy.ndarray
        Series of GRADE score contribution percentage for Hypoglycaemia, Euglycaemia and Hyperglycaemia. Or a list of
        GRADE scores for each value.

    Notes
    -----
    This function is meant to be used by :meth:`glucopy.Gframe.grade`
    '''
    values = df['CGM'].values
    if unit == 'mg/dL':
        values = mgdl_to_mmoll(values)

    grade = np.minimum(425 * np.square( np.log10( np.log10(values) ) + 0.16), 50)

    if percentage:
        grade_sum = np.sum(grade)
        hypo = np.sum(grade[values < 3.9]) / grade_sum 
        hyper = np.sum(grade[values > 7.8]) / grade_sum
        eugly = 1 - hypo - hyper
        grade = pd.Series([hypo, eugly, hyper], 
                          index=['Hypoglycaemia', 'Euglycaemia', 'Hyperglycaemia'], 
                          name = 'GRADE') * 100
    
    return grade