# 3rd party
import pandas as pd

def std(df: pd.DataFrame,
        per_day: bool = False,
        ddof: int = 1,
        ):
        '''
        Calculates the standard deviation of the CGM values.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the CGM values. The dataframe must contain 'CGM' and 'Day' columns present in
            :attr:`glucopy.Gframe.data`.
        per_day : bool, default False
            If True, returns a pandas Series with the standard deviation for each day. If False, returns the 
            standard deviation for the entire dataset.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of
            elements. By default ddof is 1.

        Returns
        -------
        std : float | pandas.Series
            Standard deviation of the CGM values.     

        Notes
        -----
        This function is meant to be used by :meth:`glucopy.Gframe.std`
        '''
        if per_day:
            # Group data by day
            day_groups = df.groupby('Day')
            std = day_groups['CGM'].std(ddof=ddof)

            # Convert the index to string for easier access
            std.index = std.index.map(str)

            # Rename the series
            std.name = 'Standard Deviation'
        
        else:
            std = df['CGM'].std(ddof=ddof)

        

        return std