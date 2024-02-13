# Built-in
import datetime

def time_to_str(time:datetime.time):
    '''
    Converts a datetime.time object to a string with format 'hh:mm:ss'

    Parameters
    ----------
    time : datetime.time
        The time object to be converted

    Returns
    -------
    str
        The time string

    Examples
    --------
    Convert a datetime.time object to a string

    .. ipython:: python

        import glucopy as gp
        import datetime
        t = datetime.time(10, 0, 0)
        gp.time_to_str(t)
    '''
    return time.strftime('%H:%M:%S')