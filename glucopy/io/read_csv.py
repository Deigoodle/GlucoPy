# Local
from ..classes import Gframe

# 3rd party
import pandas as pd

# Built-in
from typing import Callable
from collections.abc import Hashable

def read_csv(path,
             sep: str = ',',
             date_column: list[str] | str | int = 0,
             cgm_column: str | int = 1,
             unit: str = 'mg/dL',
             date_format: str | None = None,
             skiprows: list[int] | int | Callable[[Hashable], bool] | None = None,
             nrows: int | None = None,
             **kwargs):
    '''
    Use pandas.read_csv to read a csv file into a Gframe object

    Parameters
    ----------
    path : str, bytes, ExcelFile, xlrd.Book, path object, or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For more information view
        the documentation for pandas.read_csv
    sep : str, default None
        Character to use as delimiter, if None ',' will be used. For more information view the 
        documentation for pandas.read_csv
    date_column : str or list of str, default None
        Column name(s) of the date values, max 2 columns, if None, the first
        Available cases:
        * Defaults to ``None``: first column will be used as date
        * ``"Date"``: column named "Date" will be used as date
        * ``["Date", "Time"]``: columns named "Date" and "Time" will be used as date
    cgm_column : str or None, default None
        Column name of the CGM values, if None, the second column will be used
    unit : str, default 'mg/dL'
        Unit of the Glucose values
    date_format : str, default None
        Format of the date information, if None, it will be assumed that the date information is in a consistent format
    skiprows : list-like, int or callable, optional
        Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file.
        For more information view the documentation for pandas.read_csv
    nrows : int, default None
        Number of rows to read
    **kwargs : dict, optional
        Any other arguments accepted by pandas.read_csv    

    Returns
    -------
    Gframe
        A Gframe object

    Examples
    --------
    Read a csv file with the first column as date and the second as cgm values (default)

    >>> import glucopy as gp
    >>> gf = gp.read_csv('data.csv')

    Read a csv file with the data column named 'Date' and the cgm column named 'CGM'

    >>> gf = gp.read_csv('data.csv', date_column='Date', cgm_column='CGM')
    '''
    # Create a list of columns to use
    if isinstance(date_column, (int, str)):
        cols = [date_column] + [cgm_column]
    else:
        cols = date_column + [cgm_column]

    # Read the csv file
    df = pd.read_csv(path, 
                     sep=sep, 
                     usecols=cols, 
                     skiprows=skiprows, 
                     nrows=nrows, 
                     **kwargs)

    return Gframe(data=df,
                  unit=unit,
                  date_column=date_column,
                  cgm_column=cgm_column,
                  date_format=date_format)


