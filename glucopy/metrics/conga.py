# 3rd party
import pandas as pd
import numpy as np

def conga(df: pd.DataFrame,
          m: int = 1,
          slack: int = 0,
          ignore_na: bool = True
          ):
    '''
    Calculates the Continuous Overall Net Glycaemic Action (CONGA).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' and 'Timestamp' columns present in
        :attr:`glucopy.Gframe.data`.
    m : int, default 1
        Number of hours to use for the CONGA calculation.
    slack : int, default 0
        Maximum number of minutes that the given time can differ from the actual time in the data.
    ignore_na : bool, default True
        If True, ignores missing values (not found within slack). If False, raises an error 
        if there are missing values.

    Returns
    -------
    conga : float
        Continuous Overall Net Glycaemic Action (CONGA).

    Notes
    -----
    This function is meant to be used by :meth:`glucopy.Gframe.conga`
    '''
    # Check input
    if m < 0:
        raise ValueError('m must be a positive number')
    if slack < 0:
        raise ValueError('slack must be a positive number or 0')
    
    # Convert slack to timedelta
    slack = pd.to_timedelta(slack, unit='m')
    m = pd.to_timedelta(m, unit='h')

    # Make a copy and add a column with the previous m hours
    data_copy = df.copy()
    data_copy['Prev_Timestamp'] = data_copy['Timestamp'] - m

    # Merge the data with itself, matching the previous m hours
    merged_data = pd.merge_asof(data_copy,
                                data_copy,
                                left_on='Timestamp',
                                right_on='Prev_Timestamp',
                                suffixes=('', '_Next'),
                                direction='nearest',
                                tolerance=slack)
    
    # Drop the rows that have no following m hours
    last_m_hours = merged_data['Timestamp'].max() - m
    merged_data = merged_data.loc[merged_data['Timestamp'] <= last_m_hours]

    # Check if there are missing values
    if not ignore_na: 
        unvalid_data = merged_data['CGM_Next'].isna()
        if unvalid_data.any():
            raise ValueError(f"No Next day data found for:\n{merged_data.loc[unvalid_data, 'Timestamp']}")
    
    # Get values that have value in the next m hours and within the slack
    valid_data = merged_data['CGM_Next'].notna()

    # Calculate the difference between the current value and the previous value
    differences = merged_data.loc[valid_data, 'CGM'] - merged_data.loc[valid_data, 'CGM_Next']

    # Calculate the standard deviation of the differences
    conga = np.std(differences)

    return conga