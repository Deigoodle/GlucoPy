# 3rd party
import pandas as pd

def mean(df : pd.DataFrame,
         per_day: bool = False,
        ):
        '''
        Calculates the mean of the CGM values.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the CGM values. The dataframe must contain 'CGM' and 'Day' columns present in
            :attr:`glucopy.Gframe.data`.
        per_day : bool, default False
            If True, returns a :py:class:`pandas.Series` with the mean for each day. If False, returns the mean for the entire dataset.

        Returns
        -------
        mean : float | pandas.Series
            Mean of the CGM values.   

        Notes
        -----
        This function is meant to be used by :meth:`glucopy.Gframe.mean`
        '''

        if per_day:
            # Group data by day
            day_groups = df.groupby('Day')
            mean = day_groups['CGM'].mean()

            # Convert the index to string for easier access
            mean.index = mean.index.map(str)

            # Rename the series
            mean.name = 'Mean'

        else:
            mean = df['CGM'].mean()

        return mean