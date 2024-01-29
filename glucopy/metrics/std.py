# 3rd party
import pandas as pd

def std(df: pd.DataFrame,
        per_day: bool = False,
        ddof: int = 1,
        **kwargs
        ) -> float | pd.Series:
        '''
        Calculates the standard deviation of the CGM values.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing the CGM values. The dataframe must contain a column named 'CGM' with the CGM values.
        per_day : bool, default False
            If True, returns a pandas Series with the standard deviation for each day. If False, returns the 
            standard deviation for all days combined.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of
            elements. By default ddof is 1.
        **kwargs : dict
            Additional keyword arguments to be passed to the function. For more information view the documentation for
            pandas.DataFrameGroupBy.std().

        Returns
        -------
        std : float | pandas.Series
            Standard deviation of the CGM values.     

        Notes
        -----
        This function is meant to be used by :class:`glucopy.Gframe.std`
        '''
        if per_day:
            # Group data by day
            day_groups = df.groupby('Day')
            std = day_groups['CGM'].std(ddof=ddof,**kwargs)
        
        else:
            std = df['CGM'].std(ddof=ddof,**kwargs)

        return std