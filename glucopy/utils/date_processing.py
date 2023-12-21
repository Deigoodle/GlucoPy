# 3rd party
import pandas as pd

# Python
import datetime

def disjoin_days_and_hours(df,
                           date_name = None, 
                           cgm_name = None) -> pd.DataFrame:
    
    if date_name is None:
        date_name = df.columns[0]
    if cgm_name is None:
        cgm_name = df.columns[1]

    disjoined_df = pd.DataFrame(columns=['Timestamp','Day','Time','CGM'])

    disjoined_df['Timestamp'] = df[date_name]
    disjoined_df['Day'] = df[date_name].dt.date
    disjoined_df['Time'] = df[date_name].dt.time
    disjoined_df['CGM'] = df[cgm_name]

    return disjoined_df

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