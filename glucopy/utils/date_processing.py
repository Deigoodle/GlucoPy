# 3rd party
import pandas as pd

# Python
import datetime

def disjoin_days_and_hours(df,
                           date_column: str | int = 0, 
                           cgm_column: str | int = 1) -> pd.DataFrame:

    disjoined_df = pd.DataFrame(columns=['Timestamp','Day','Time','CGM'])

    if isinstance(date_column, int):
        disjoined_df['Timestamp'] = df.iloc[:, date_column]
    else:
        disjoined_df['Timestamp'] = df.loc[:, date_column]

    if isinstance(cgm_column, int):
        disjoined_df['CGM'] = df.iloc[:, cgm_column]
    else:
        disjoined_df['CGM'] = df.loc[:, cgm_column]

    disjoined_df['Day'] = disjoined_df['Timestamp'].dt.date
    disjoined_df['Time'] = disjoined_df['Timestamp'].dt.time

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