# Local
from ..classes import Gframe


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
    -------
    Nothing

    '''
    # Create a copy of the data
    df = gframe.data.copy()

    if not include_time_and_day:
        # Drop the time and day columns
        df.drop(columns=['Day','Time'], inplace=True)

    # Save the csv file
    df.to_csv(path, sep=sep, index=include_index, **kwargs)