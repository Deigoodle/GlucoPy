# Built-in
import datetime

def time_to_str(time:datetime.time) -> str:
    '''
    Converts a datetime.time object to a string with format 'hh:mm:ss'
    '''
    return time.strftime('%H:%M:%S')