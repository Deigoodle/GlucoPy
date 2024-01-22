# Built-in
import datetime

# 3rd party
import pandas as pd

def str_to_time(time_string:str) -> datetime.time:
    '''
    Converts a string with format 'hh:mm:ss, hh:mm or hh' to a datetime.time object

    Parameters
    ----------
    time_string : str
        String to convert to datetime.time object

    Returns
    -------
    time_result : datetime.time
        Datetime.time object

    Examples
    --------
    Convert a string with format 'hh:mm:ss' to a datetime.time object

    .. ipython:: python

        import glucopy as gp
        gp.str_to_time('12:00:00')

    Convert a string with format 'hh:mm' to a datetime.time object

    .. ipython:: python

        gp.str_to_time('12:00')

    Convert a string with format 'hh' to a datetime.time object

    .. ipython:: python
    
        gp.str_to_time('12')  
    '''
    if time_string.count(':') == 0:  # Format is 'HH'
        time_string += ':00:00'
    elif time_string.count(':') == 1:  # Format is 'HH:MM'
        time_string += ':00'

    try:
        time_result = pd.to_datetime(time_string).time()
    except:
        raise ValueError('time must be a valid time format')
    
    return time_result