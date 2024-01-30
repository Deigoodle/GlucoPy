# 3rd party
import pandas as pd
import numpy as np

# Local
from glucopy.utils import mgdl_to_mmoll

def grade(df: pd.DataFrame,
          percentage: bool = True,
          unit: str = 'mg/dL'):
    '''
    Calculates the contributions of the Glycaemic Risk Assessment Diabetes Equation (GRADE) to Hypoglycaemia,
    Euglycaemia and Hyperglycaemia. Or the GRADE scores for each value.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' column present in
        :class:`glucopy.Gframe.data`.
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
    This function is meant to be used by :class:`glucopy.Gframe.grade`
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
        grade = pd.Series([hypo, eugly, hyper], index=['Hypoglycaemia', 'Euglycaemia', 'Hyperglycaemia']) * 100
    
    return grade