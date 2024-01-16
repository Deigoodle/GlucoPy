# Local
from ..classes import Gframe

# 3rd party
import pandas as pd

# Built-in
from typing import Callable
from collections.abc import Sequence

def load_excel(path,
               sheet_name: str | int | list | None = 0,
               date_column: list[str] | str | int = 0,
               cgm_column: str | int = 1,
               skiprows: Sequence[int] | int | Callable[[int], object] | None = None,
               nrows: int | None = None,):
    '''
    Use pandas.read_excel to load an excel file into a Gframe object

    Parameters
    ----------
    path : str, bytes, ExcelFile, xlrd.Book, path object, or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For more information view
        the documentation for pandas.read_excel

    sheet_name : str, int, list, or None, default 0
        Strings are used for sheet names. Integers are used in zero-indexed
        sheet positions. Lists of strings/integers are used to request
        multiple sheets. Specify None to get all sheets. For more information view
        the documentation for pandas.read_excel

        Available cases:

        * Defaults to ``0``: 1st sheet as a `GFrame`
        * ``1``: 2nd sheet as a `GFrame`
        * ``"Sheet1"``: Load sheet with name "Sheet1"
        * ``[0, 1, "Sheet5"]``: Load first, second and sheet named "Sheet5"
            as a dict of `GFrame`
        * None: All worksheets.

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
        For more information view the documentation for pandas.read_excel
    
    nrows : int, default None
        Number of rows to read.
        
    Returns
    -------
    Gframe
        Gframe object
    '''
    # Create a list of columns to use
    if isinstance(date_column, (int, str)):
        cols = [date_column] + [cgm_column]
    else:
        cols = date_column + [cgm_column]

    # Load the excel file
    df = pd.read_excel(path, sheet_name=sheet_name, usecols=cols, skiprows=skiprows, nrows=nrows)

    return Gframe(data=df,date_column=date_column,cgm_column=cgm_column)

