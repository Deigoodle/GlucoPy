# Local
from ..classes import Gframe


def save_excel(gframe: Gframe,
             path: str,
             include_index: bool = False,
             include_time_and_day: bool = False,
             **kwargs):
    '''
    Use pandas.DataFrame.to_csv to save a Gframe object into an excel file

    Parameters
    ----------
    gframe : Gframe
        A Gframe object

    path : str, bytes, ExcelFile, xlrd.Book, path object, or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For more information view
        the documentation for pandas.DataFrame.to_csv
    include_index : bool, default False
        Write row names (index). For more information view the documentation for pandas.DataFrame.to_csv
    include_time_and_day : bool, default False
        Write time and day columns.
    **kwargs : dict, optional
        Any other arguments accepted by pandas.DataFrame.to_csv

    Returns
    -------
    Nothing

    Examples
    --------
    Save a Gframe object into a xlsx file

    >>> import glucopy as gp
    >>> gf = gp.data()
    >>> gp.save_excel(gf, 'data.xlsx')
    '''
    # Create a copy of the data
    df = gframe.data.copy()

    if not include_time_and_day:
        # Drop the time and day columns
        df.drop(columns=['Day','Time'], inplace=True)

    # Save the csv file
    df.to_excel(path, index=include_index, **kwargs)