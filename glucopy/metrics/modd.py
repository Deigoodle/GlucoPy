# 3rd party
import pandas as pd
import numpy as np

# Built-in
import datetime

# Local
from ..utils import str_to_time, time_to_str


def modd(df: pd.DataFrame, 
         target_time: str | datetime.time | None = None, 
         slack: int = 0,
         ignore_na: bool = True
        ):
    '''
    Calculates the Mean of Daily Differences (MODD) for a given time of day.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain all columns present in :attr:`glucopy.Gframe.data`.
    target_time : str | datetime.time | None, default None
        Time of day to calculate the MODD for. If None, calculates the MODD for all available times.
    slack : int, default 0
        Maximum number of minutes that the given time can differ from the actual time in the data.
    ignore_na : bool, default True
        If True, ignores missing values (not found within slack). If False, raises an error 
        if there are missing values.

    Returns
    -------
    modd : float
        Mean of Daily Differences (MODD).   

    Notes
    -----
    This function is meant to be used by :meth:`glucopy.Gframe.modd`
    '''
    # Convert slack to timedelta
    slack = pd.to_timedelta(slack, unit='m')

    if target_time is None: # calculate MODD for all times
        # Make a copy and add a column with the previous day
        data_copy = df.copy()
        data_copy['Prev_Timestamp'] = data_copy['Timestamp'] - pd.Timedelta('1 day')

        # Merge the data with itself, matching the previous day within the slack
        merged_data = pd.merge_asof(data_copy, 
                                    data_copy, 
                                    left_on='Timestamp', 
                                    right_on='Prev_Timestamp', 
                                    suffixes=('', '_Next'),
                                    direction='nearest',
                                    tolerance=slack)
        
        # Drop last day rows because it will never have a next day value
        last_day = merged_data['Day'].max()
        merged_data = merged_data.loc[merged_data['Day'] != last_day]
        
        # Check if there are missing values
        if not ignore_na: 
            unvalid_data = merged_data['CGM_Next'].isna()
            if unvalid_data.any():
                raise ValueError(f"No Next day data found for:\n{merged_data.loc[unvalid_data, 'Timestamp']}")
            
        valid_data = merged_data['CGM_Next'].notna() # Get values that have a next day within the slack

        # Calculate the difference between the current day and the previous day
        differences = np.abs(merged_data.loc[valid_data, 'CGM'] - merged_data.loc[valid_data, 'CGM_Next'])

        # Calculate the mean of the differences
        modd = differences.mean()

    else: # Calculate MODD for a given time
        # Convert time to same format as df['Time']
        if not isinstance(target_time, str) and not isinstance(target_time, datetime.time):
            raise TypeError('time must be a string or a datetime.time')
        elif isinstance(target_time, str):# String -> datetime.time
            target_str = target_time
            target_time = str_to_time(target_str)
        else: # datetime.time
            target_str = time_to_str(target_time)

        cgm_values = []
        # Search given time in each day
        day_groups = df.groupby('Day')
        for day, day_data in day_groups:
            target_time_index = day_data['Time'] == target_time

            # If exact time is found, use it
            if target_time_index.any():
                cgm_values.append(day_data.loc[target_time_index, 'CGM'].values[0])

            # If not, search for closest time within error range
            elif slack > pd.Timedelta('0 min'):
                # Combine "day" and target_time to compare it with Timestamp
                target_date = str(day) + ' ' + target_str
                target_datetime = pd.to_datetime(target_date)

                # Search for closest time within error range
                mask_range = ((day_data['Timestamp'] - target_datetime).abs() <= slack)

                if mask_range.any(): # If there are values within the error range, use the closest one
                    closest_index = (day_data.loc[mask_range, 'Timestamp'] - target_datetime).abs().idxmin()
                    cgm_values.append(day_data.loc[closest_index, 'CGM'])

                else: # If there are no values within the error range, use NaN
                    if ignore_na:
                        cgm_values.append(np.nan)
                    else:
                        raise ValueError(f"No data found for date {day}")
                    
            else:
                if ignore_na:
                    cgm_values.append(np.nan)
                else:
                    raise ValueError(f"No data found for date {day}")
        
        modd = np.nanmean(np.abs(np.diff(cgm_values)))

    return modd