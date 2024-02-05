# 3rd party
import pandas as pd

def mard(cgm_df: pd.DataFrame,
         smbg_df: pd.DataFrame,
         slack: int = 0,
         interpolate: bool = True
        ):
    '''
    Calculates the Mean Absolute Relative Difference (MARD).

    .. math::

        MARD = \\frac{1}{N} \\sum_{i=1}^N \\frac{|CGM_i - SMBG_i|}{SMBG_i} * 100

    - :math:`N` is the number of SMBG readings.
    - :math:`CGM_i` is the Continuous Glucose Monitoring (CGM) value at time i.
    - :math:`SMBG_i` is the Self Monitoring of Blood Glucose (SMBG) value at time i.

    Parameters
    ----------
    cgm_df : pandas.DataFrame
        DataFrame containing the CGM values. The dataframe must contain 'CGM' and 'Timestamp' columns present in
        :attr:`glucopy.Gframe.data`.
    smbg_df : pandas.DataFrame
        DataFrame containing the SMBG values. The dataframe must contain 'SMBG' and 'Timestamp' columns present in
        :attr:`glucopy.Gframe.data`.
    slack : int, default 0
        Maximum number of minutes that a given CGM value can be from an SMBG value and still be considered a match.
    interpolate : bool, default True
        If True, the SMBG values will be interpolated to the CGM timestamps. If False, Only CGM values that have
        corresponding SMBG values will be used.

    Returns
    -------
    mard : float
        Mean Absolute Relative Difference (MARD).

    Notes
    -----
    This function is meant to be used by :meth:`glucopy.Gframe.mard`
    '''
    # Rename 'Timestamp' in smbg_df to avoid naming conflict
    smbg_df = smbg_df.rename(columns={'Timestamp': 'SMBG_Timestamp'})

    # Merge the CGM and SMBG dataframes
    df = pd.merge_asof(cgm_df, 
                       smbg_df, 
                       left_on='Timestamp',
                       right_on='SMBG_Timestamp',
                       direction='nearest',
                       tolerance=pd.Timedelta(minutes=slack))
    
    # Remove duplicated SMBG values with a partner CGM value within the slack
    df.loc[df.duplicated(subset=['SMBG_Timestamp','SMBG'], keep='first'), 'SMBG'] = pd.NA

    # Set 'Timestamp' as the index
    df.set_index('Timestamp', inplace=True)

    # Interpolate the SMBG values between the first and last non-NA values
    if interpolate:
        first_valid = df['SMBG'].first_valid_index()
        last_valid = df['SMBG'].last_valid_index()
        df.loc[first_valid:last_valid, 'SMBG'] = df.loc[first_valid:last_valid, 'SMBG'].interpolate(method='time')

    # Remove rows with no partner
    df.dropna(subset=['SMBG'],inplace=True)

    # Calculate the MARD
    mard = (df['CGM'] - df['SMBG']).abs().mean() / df['SMBG'].mean() * 100

    return mard
