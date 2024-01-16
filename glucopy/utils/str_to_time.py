# Built-in
import datetime

# 3rd party
import pandas as pd

def str_to_time(time_string:str) -> datetime.time:
    '''
    Converts a string with format 'hh:mm:ss, hh:mm or hh' to a datetime.time object
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