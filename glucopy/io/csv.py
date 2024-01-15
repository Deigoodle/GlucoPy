# Local
from ..classes import Gframe

# 3rd party
import pandas as pd

# Built-in
from typing import Callable
from collections.abc import Hashable

def load_csv(path,
             sep: str = ',',
             date_column: list[str] | str | int = 0,
             cgm_column: str | int = 1,
             skiprows: list[int] | int | Callable[[Hashable], bool] | None = None,
             nrows: int | None = None,
             **kwargs):
    '''
    Use pandas.read_csv to load a csv file into a Gframe object

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
    '''
    # Create a list of columns to use
    if isinstance(date_column, (int, str)):
        cols = [date_column] + [cgm_column]
    else:
        cols = date_column + [cgm_column]

    # Load the csv file
    df = pd.read_csv(path, sep=sep, usecols=cols, skiprows=skiprows, nrows=nrows, **kwargs)

    return Gframe(data=df,date_column=date_column,cgm_column=cgm_column)

def save_csv(gframe: Gframe,
             path: str,
             sep: str = ',',
             include_index: bool = False,
             include_time_and_day: bool = False,
             **kwargs):
    '''
    Use pandas.DataFrame.to_csv to save a Gframe object into a csv file

    Parameters
    ----------
    gframe : Gframe
        A Gframe object

    path : str, bytes, ExcelFile, xlrd.Book, path object, or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For more information view
        the documentation for pandas.DataFrame.to_csv
    sep : str, default None
        Character to use as delimiter, if None ',' will be used. For more information view the 
        documentation for pandas.DataFrame.to_csv
    include_index : bool, default False
        Write row names (index). For more information view the documentation for pandas.DataFrame.to_csv
    include_time_and_day : bool, default False
        Write time and day columns.
    **kwargs : dict, optional
        Any other arguments accepted by pandas.DataFrame.to_csv

    Returns
    '''
    # Create a copy of the data
    df = gframe.data.copy()

    if not include_time_and_day:
        # Drop the time and day columns
        df.drop(columns=['Day','Time'], inplace=True)

    # Save the csv file
    df.to_csv(path, sep=sep, index=include_index, **kwargs)


