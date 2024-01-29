# 3rd party
import pandas as pd

def mean(df : pd.DataFrame,
         per_day: bool = False,
         **kwargs
        ) -> float | pd.Series:
        '''
        Calculates the mean of the CGM values.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing the CGM values. The dataframe must contain a column named 'CGM' with the CGM values.
        per_day : bool, default False
            If True, returns a pandas Series with the mean for each day. If False, returns the mean for all days combined.
        **kwargs : dict
            Additional keyword arguments to be passed to the function. For more information view the documentation for
            pandas.DataFrameGroupBy.mean() for per_day=True and pandas.DataFrame.mean() for per_day=False.

        Returns
        -------
        mean : float | pandas.Series
            Mean of the CGM values.   

        Notes
        -----
        This function is meant to be used by :class:`glucopy.Gframe.mean`
        '''

        if per_day:
            # Group data by day
            day_groups = df.groupby('Day')
            mean = day_groups['CGM'].mean(**kwargs)

        else:
            mean = df['CGM'].mean(**kwargs)

        return mean