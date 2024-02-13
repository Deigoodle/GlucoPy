#3rd party
import numpy as np
import pandas as pd

# Built-in
from collections.abc import Sequence
import datetime

# Local
from glucopy.utils import (disjoin_days_and_hours,
                           mgdl_to_mmoll,
                           mmoll_to_mgdl)
import glucopy.metrics as metrics

class Gframe:
    '''
    Class for the analysis of CGM data. it uses a pandas DataFrame as the main data structure.

    To create a Gframe object from a csv file or an excel file, check Input/Output glucopy.read_csv() 
    and glucopy.read_excel().

    Parameters
    -----------
    data : pandas DataFrame 
        DataFrame containing the CGM signal information, it will be saved into a DataFrame with the columns 
        ['Timestamp','Day','Time','CGM']
    unit : str, default 'mg/dL'
        CGM signal measurement unit.
    date_column : str or str array, default None
        The name or names of the column(s) containing the date information
        If it's a str, it will be the name of the single column containing the date information
        If it's a str array, it will be the 2 names of the columns containing the date information, eg. ['Date','Time']
        If it's None, it will be assumed that the date information is in the first column
    cgm_column : str, default None
        The name of the column containing the CGM signal information
        If it's None, it will be assumed that the CGM signal information is in the second column
    date_format : str, default None
        Format of the date information, if None, it will be assumed that the date information is in a consistent format
    dropna : bool, default True
        If True, removes all rows with NaN values

    Attributes
    ----------
    data : pandas DataFrame
        DataFrame containing the CGM signal information, it will be saved into a DataFrame with the columns:

        - 'Timestamp' : datetime64[ns]  # pandas datetime
        - 'Day' : datetime.date 
        - 'Time' : datetime.time
        - 'CGM' : number 

    unit : str
        CGM signal measurement unit. Can be 'mg/dL' or 'mmol/L'.
    n_samples : int
        Number of samples in the data.
    n_days : int
        Number of days in the data.
    max : float
        Maximum value of the CGM signal.
    min : float
        Minimum value of the CGM signal.

    Examples
    --------
    Creating a Gframe object from a pandas DataFrame:

    .. ipython:: python

        import glucopy as gp
        import pandas as pd
        df = pd.DataFrame({'Timestamp':['2020-01-01 12:00:00','2020-01-01 12:05:00','2020-01-01 12:10:00'],
                           'CGM':[100,110,120]})
        gf = gp.Gframe(df)
        gf

    Creating a Gframe object from a pandas DataFrame with extra columns:

    .. ipython:: python

        df = pd.DataFrame({'Timestamp':['2020-01-01 12:00:00','2020-01-01 12:05:00','2020-01-01 12:10:00'],
                            'Extra':[1,2,3],
                            'CGM':[100,110,120]})
        gf = gp.Gframe(df, cgm_column='CGM', date_column='Timestamp')
        gf
    '''

    # Constructor
    def __init__(self, 
                 data=None, 
                 unit:str = 'mg/dL',
                 date_column: list[str] | str | int = 0,
                 cgm_column: str | int = 1,
                 date_format: str | None = None,
                 dropna:bool = True):
        
        # Check data is a dataframe
        if isinstance(data, pd.DataFrame):
            # Check date_column
            if isinstance(date_column, str) or isinstance(date_column, int):
                self.data = disjoin_days_and_hours(data, date_column, cgm_column, date_format)

            # if date_column is a list of 2 strings
            elif isinstance(date_column, Sequence) and len(date_column) == 2:
                self.data = pd.DataFrame(columns=['Timestamp','Day','Time','CGM'])
                combined_timestamp = pd.to_datetime(data[date_column[0]].astype(str) + ' ' + data[date_column[1]].astype(str))

                self.data['Timestamp'] = combined_timestamp
                self.data['Day'] = combined_timestamp.dt.date
                self.data['Time'] = combined_timestamp.dt.time
                self.data['CGM'] = data[cgm_column]

            else:
                raise ValueError('date_column must be a String or a sequence of 2 Strings')
            
        if dropna: # remove all rows with NaN values
            self.data.dropna(inplace=True)


        # atributes
        self.unit = unit
        self.n_samples = self.data.shape[0]
        self.n_days = len(self.data['Day'].unique())
        self.max = self.data['CGM'].max()
        self.min = self.data['CGM'].min()


    # String representation
    def __repr__(self):
        return str(self.data)
    
    # Glucose Unit Conversion
    def convert_unit(self,
                     new_unit: str = 'mmol/L'):
        '''
        Converts the unit of the CGM signal.

        Parameters
        ----------
        new_unit : str, default 'mmol/L'
            New unit for the CGM signal. Can be 'mg/dL' or 'mmol/L'.

        Returns
        -------
        None

        Examples
        --------
        Converting the unit of the CGM signal to mmol/L:

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.convert_unit('mmol/L')
        '''
        # Check input
        if new_unit not in ['mg/dL', 'mmol/L']:
            raise ValueError("new_unit must be 'mg/dL' or 'mmol/L'")
        
        # Convert unit
        if new_unit == 'mmol/L':
            if self.unit == 'mg/dL':
                self.data['CGM'] = mgdl_to_mmoll(self.data['CGM'])
                self.unit = new_unit
            else:
                raise ValueError('The data is already in mmol/L')
        else:
            if self.unit == 'mmol/L':
                self.data['CGM'] = mmoll_to_mgdl(self.data['CGM'])
                self.unit = new_unit
            else:
                raise ValueError('The data is already in mg/dL')
    
    # Metrics 
    # -------
    # 1. Joint data analysis metrics for glycaemia dynamics

    # Sample Mean
    def mean(self,
             per_day: bool = False,
             **kwargs):
        '''
        Calculates the mean of the CGM values.

        Parameters
        ----------
        per_day : bool, default False
            If True, returns a pandas Series with the mean for each day. If False, returns the mean for all days combined.
        **kwargs : dict
            Additional keyword arguments to be passed to the function. For more information view the documentation for
            pandas.DataFrameGroupBy.mean() for per_day=True and pandas.DataFrame.mean() for per_day=False.

        Returns
        -------
        mean : float | pandas.Series
            Mean of the CGM values.   

        Examples
        --------
        Calculating the mean for the entire dataset:

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.mean()

        Calculating the mean for each day:

        .. ipython:: python

            gf.mean(per_day=True)     
        '''

        return metrics.mean(df=self.data, per_day=per_day, **kwargs)
    
    # Standard Deviation, by default ddof=1, so its divided by n-1
    def std(self,
            per_day: bool = False,
            ddof:int = 1,
            **kwargs):
        '''
        Calculates the standard deviation of the CGM values.

        Parameters
        ----------
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

        Examples
        --------
        Calculating the standard deviation for the entire dataset:

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.std()       

        Calculating the standard deviation for each day:

        .. ipython:: python

            gf.std(per_day=True)
        '''
        
        return metrics.std(df=self.data, per_day=per_day, ddof=ddof, **kwargs)
    
    # Coefficient of Variation
    def cv(self,
           per_day: bool = False,
           ddof:int = 1,
           **kwargs):
        '''
        Calculates the Coefficient of Variation (CV) of the CGM values.

        Parameters
        ----------
        per_day : bool, default False
            If True, returns the an array with the coefficient of variation for each day. If False, returns
            the coefficient of variation for all days combined.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of 
            elements. By default ddof is 1.
        **kwargs : dict
            Additional keyword arguments to be passed to the function. For more information view the documentation for
            pandas.DataFrameGroupBy.mean() and std().

        Returns
        -------
        cv : float | pandas.Series
            Coefficient of variation of the CGM values.    

        Examples
        --------
        Calculating the coefficient of variation for the entire dataset:

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.cv()

        Calculating the coefficient of variation for each day:

        .. ipython:: python

            gf.cv(per_day=True)       
        '''

        return metrics.cv(df=self.data, per_day=per_day, ddof=ddof, **kwargs)
            
    # % Coefficient of Variation
    def pcv(self,
            per_day: bool = False,
            ddof:int = 1,
            **kwargs):
        '''
        Calculates the Percentage Coefficient of Variation (%CV) of the CGM values.
        
        Parameters
        ----------
        per_day : bool, default False
            If True, returns the a pandas.Series with the percentage coefficient of variation for each day. If False,
            returns the percentage coefficient of variation for all days combined.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of 
            elements. By default ddof is 1.
        **kwargs : dict
            Additional keyword arguments to be passed to the function. For more information view the documentation for
            pandas.DataFrameGroupBy.mean() and std().

        Returns
        -------
        pcv : float | pandas.Series
            Percentage coefficient of variation of the CGM values.

        Examples
        --------
        Calculating the percentage coefficient of variation for the entire dataset:

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.pcv()

        Calculating the percentage coefficient of variation for each day:

        .. ipython:: python

            gf.pcv(per_day=True)
        '''
        if per_day:
            pcv = self.cv(per_day=True,ddof=ddof,**kwargs) * 100

            # Rename the series
            pcv.name = '% Coefficient of Variation'
          
        else:
            pcv = self.cv(ddof=ddof,**kwargs) * 100

        return pcv
    
    # Quantiles
    def quantile(self,
                 per_day: bool = False,
                 q:float = 0.5,
                 interpolation:str = 'linear',
                 **kwargs):
        '''
        Calculates the quantile of the CGM values.

        Parameters
        ----------
        per_day : bool, default False
            If True, returns a pandas.Series with the quantile for each day. If False, returns the quantile for all
            days combined.
        q : float, default 0.5
            Value between 0 and 1 for the desired quantile.
        interpolation : str, default 'linear'
            This optional parameter specifies the interpolation method to use, when the desired quantile lies between 
            two data points i and j. Default is 'linear'.
        **kwargs : dict
            Additional keyword arguments to be passed to the function. For more information view the documentation for
            pandas.DataFrameGroupBy.quantile().

        Returns
        -------
        quantile : float | pandas.Series
            Quantile of the CGM values.

        Examples
        --------
        Calculating the median for the entire dataset:

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.quantile()

        Calculating the first quartile for the entire dataset:

        .. ipython:: python

            gf.quantile(q=0.25)

        Calculating the median for each day:

        .. ipython:: python

            gf.quantile(per_day=True)
        '''
        
        return metrics.quantile(df=self.data, per_day=per_day, q=q, interpolation=interpolation, **kwargs)
    
    # Interquartile Range
    def iqr(self,
            per_day: bool = False,
            interpolation:str = 'linear',
            **kwargs):
        '''
        Calculates the Interquartile Range (IQR) of the CGM values.

        Parameters
        ----------
        per_day : bool, default False
            If True, returns a pandas.Series with the interquartile range for each day. If False, returns the
            interquartile range for all days combined.
        interpolation : str, default 'linear'
            This optional parameter specifies the interpolation method to use, when the desired quantile lies between
            two data points i and j. Default is 'linear'. 
        **kwargs : dict
            Additional keyword arguments to be passed to the function. For more information view the documentation for
            pandas.DataFrameGroupBy.quantile().

        Returns
        -------
        iqr : float | pandas.Series
            Interquartile range of the CGM values.

        Examples
        --------
        Calculating the interquartile range for the entire dataset:

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.iqr()

        Calculating the interquartile range for each day:

        .. ipython:: python

            gf.iqr(per_day=True)
        '''
        
        return metrics.iqr(df=self.data, per_day=per_day, interpolation=interpolation, **kwargs)
    
    # Mean of Daily Differences
    def modd(self, 
             target_time: str | datetime.time | None = None, 
             slack: int = 0,
             ignore_na: bool = True) -> float:
        '''
        Calculates the Mean of Daily Differences (MODD).

        .. math::

            MODD = \\frac{1}{T} \\sum_{t=1}^T | X_t - X_{t-1} |

        - :math:`X_t` is the glucose value at time t.
        - :math:`X_{t-1}` is the glucose value 24 hours before time t.
        - :math:`T` is the number of observations with a previous 24-hour observation.

        Parameters
        ----------
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

        Examples
        --------
        Calculating the MODD for a target time with a slack of 5 minutes:

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.modd(target_time='08:00', slack=5)

        Calculating the MODD for all times with a slack of 10 minutes:

        .. ipython:: python

            gf.modd(slack=10) 
        '''
        return metrics.modd(df=self.data, target_time=target_time, slack=slack, ignore_na=ignore_na)
        
    # Time in Range
    def tir(self, 
            per_day: bool = False,
            interval:list= [0,70,180],
            percentage: bool = True,
            decimals: int = 2):
        '''
        Calculates the Time in Range (TIR) for a given target range of glucose.

        .. math::

            TIR = \\frac{\\tau}{T} * 100

        - :math:`\\tau` is the time spent within the target range.
        - :math:`T` is the total time of observation.

        Parameters
        ----------
        per_day : bool, default False
            If True, returns a pandas Series with the TIR for each day. If False, returns the TIR for all days combined.
        interval : list of int|float, default [0,70,180]
            Interval of glucose concentration to calculate :math:`\\tau`. Can be a list of 1 number, in that case the 
            time will be calculated below and above that number. It will always try to calculate the time below the first 
            number and above the last number.
        percentage : bool, default True
            If True, returns the TIR as a percentage. If False, returns the TIR as timedelta (:math:`TIR=\\tau`).
        decimals : int, default 2
            Number of decimal places to round to. Use None for no rounding.

        Returns
        -------
        tir : pandas.Series 
            Series of TIR.

        Examples
        --------
        Calculating the TIR for the entire dataset and the default range (0,70,180), this will return an array with the
        tir between 0 and 70, between 70 and 180 and above 180:

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.tir()

        Calculating the TIR for a custom range:

        .. ipython:: python

            gf.tir(interval=[0,70,150,180,230])

        Calculating the TIR for each day and the default range (0,70,180):

        .. ipython:: python

            gf.tir(per_day=True)
        '''
        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')

            # Initialize tir as an empty Series
            tir = pd.Series(dtype=float, name='Time in Range')
            tir.index.name = 'Day'

            # Calculate TIR for each day
            for day, day_data in day_groups:
                tir[str(day)] = metrics.tir(df=day_data, interval=interval, percentage=percentage, decimals=decimals).tolist()

        else: # Calculate TIR for all data
            tir = metrics.tir(df=self.data, interval=interval, percentage=percentage, decimals=decimals)

        return tir
    
    # 2. Analysis of distribution in the plane for glycaemia dynamics.

    # Frecuency distribution : counts the amount of observations given certain intervals of CGM
    def fd(self,
           per_day: bool = False,
           interval: list = [0,70,180],
           decimals: int = 2,
           count: bool = False):
        '''
        Calculates the Frequency Distribution (FD) for a given target range of glucose.

        .. math::

            FD = \\frac{n_i}{N}

        - :math:`n_i` is the number of observations within the `i`-th interval.
        - :math:`N` is the total number of observations.

        Parameters
        ----------
        per_day : bool, default False
            If True, returns a pandas Series with the FD for each day. If False, returns the FD for all days combined.
        interval : list of int|float, default [0,70,180]
            Interval of glucose concentration to calculate `FD`. Can be a list of 1 number, in that case the time will
            be calculated below and above that number. It will always try to calculate the time below the first number
            and above the last number.
        decimals : int, default 2
            Number of decimal places to round to. Use None for no rounding.
        count : bool, default False
            If True, returns the count of observations for each range. If False, returns the percentage of observations

        Returns
        -------
        fd : pandas.Series 
            Series of fd for each day.

        Examples
        --------
        Calculating the FD for the entire dataset and the default range (0,70,180):

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.fd()

        Calculating the FD for a custom range:

        .. ipython:: python

            gf.fd(interval=[0,70,150,180,230])
        
        Calculating the FD for each day and the default range (0,70,180):

        .. ipython:: python

            gf.fd(per_day=True)
        '''
        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')

            fd = pd.Series(dtype=float, name = 'Time in Range')
            fd.index.name = 'Day'

            for day, day_data in day_groups:
                fd[str(day)] = metrics.fd(df=day_data, interval=interval, decimals=decimals, count=count).values
        
        else:
            fd = metrics.fd(df=self.data, interval=interval, decimals=decimals, count=count)

        return fd

    # Area Under the Curve (AUC)
    def auc(self,
            per_day: bool = False,
            time_unit='m',
            threshold: int | float = 0,
            above: bool = True):
        '''
        Calculates the Area Under the Curve (AUC) using the trapezoidal rule.

        .. math::

            AUC = \\frac{1}{2} \\sum_{i=1}^{N} (X_i + X_{i-1}) * (T_i - T_{i-1})

        - :math:`X_i` is the glucose value at time i.
        - :math:`T_i` is the time at time i.
        - :math:`N` is the number of glucose readings.
            
        Parameters
        ----------
        per_day : bool, default False
            If True, returns a pandas Series with the AUC for each day. If False, returns the AUC for all days combined.
        time_unit : str, default 'm' (minutes)
            The time unit for the x-axis. Can be 's (seconds)', 'm (minutes)', or 'h (hours)'.
        threshold : int | float, default 0
            The threshold value above which the AUC will be calculated.
        above : bool, default True
            If True, the AUC will be calculated above the threshold. If False, the AUC will be calculated below the
            threshold.

        Returns
        -------
        auc : float | pandas.Series
            Area Under the Curve (AUC).

        Examples
        --------
        Calculating the AUC for the entire dataset and minutes as the time unit (default):

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.auc()

        Calculating the AUC for each day and minutes as the time unit (default):

        .. ipython:: python

            gf.auc(per_day=True)

        Calculating the AUC for the entire dataset, hours as the time unit, and below the threshold (100):

        .. ipython:: python

            gf.auc(time_unit='h', threshold=100, above=False)
        '''
        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')

            # Initialize auc as an empty Series
            auc = pd.Series(dtype=float, name = 'AUC')
            auc.index.name = 'Day'

            # Calculate AUC for each day
            for day, day_data in day_groups:
                auc[str(day)] = metrics.auc(df=day_data, time_unit=time_unit, threshold=threshold, above=above)

        else:
            auc = metrics.auc(df=self.data, time_unit=time_unit, threshold=threshold, above=above)

        return auc


    # 3. Amplitude and distribution of frequencies metrics for glycaemia dynamics.

    # Mean Amplitude of Glycaemic Excursions (MAGE)
    def mage(self,
             per_day: bool = False):
        '''
        Calculates the Mean Amplitude of Glycaemic Excursions (MAGE).

        .. math::

            MAGE = \\frac{1}{K} \\sum_{i=1}^K \\lambda_i * I(\\lambda_i > s)

        - :math:`\\lambda_i` is the difference between a peak and a nadir of glycaemia (or nadir-peak).
        - :math:`s` is the standar deviation of the glucose values.
        - :math:`I(\\lambda_i > s)` is the indicator function that returns 1 if :math:`\\lambda_i > s` and 0 otherwise.
        - :math:`K` is the number of events such that :math:`\\lambda_i > s`

        Parameters
        ----------
        per_day : bool, default False
            If True, returns a pandas Series with the MAGE for each day. If False, returns the MAGE for all days combined.

        Returns
        -------
        mage : float | pandas.Series
            Mean Amplitude of Glycaemic Excursions.

        Examples
        --------
        Calculating the MAGE for the entire dataset:

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.mage()

        Calculating the MAGE for each day:

        .. ipython:: python

            gf.mage(per_day=True)
        '''
        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')

            # Initialize mage as an empty Series
            mage = pd.Series(dtype=float, name = 'MAGE')
            mage.index.name = 'Day'

            # Calculate MAGE for each day
            for day, day_data in day_groups:
                mage[str(day)] = metrics.mage(df=day_data)
        
        else:
            mage = metrics.mage(df=self.data)

        return mage

    # Distance Travelled (DT)
    def dt(self,
           per_day: bool = False):
        '''
        Calculates the Distance Travelled (DT).

        .. math::

            DT = \\sum_{i=1}^{N-1} | X_{i+1} - X_i |

        - :math:`X_i` is the glucose value at time i.
        - :math:`N` is the number of glucose readings.

        Parameters
        ----------
        per_day : bool, default False
            If True, returns a pandas Series with the DT for each day. If False, returns the DT for all days combined.

        Returns
        -------
        dt : float | pandas.Series
            Distance Travelled of Glucose Values.

        Examples
        --------
        Calculating the DT for the entire dataset (default):

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.dt()

        Calculating the DT for each day:

        .. ipython:: python

            gf.dt(per_day=True)
        '''
        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')

            dt = pd.Series(dtype=float, name='Distance Travelled')
            dt.index.name = 'Day'

            # Calculate DT for each day
            for day, day_data in day_groups:
                dt[str(day)] = metrics.dt(df=day_data)

        else:
            dt = metrics.dt(df=self.data)

        return dt
    
    # 4. Metrics for the analysis of glycaemic dynamics using scores of glucose values

    # Low Blood Glucose Index (LBGI) and High Blood Glucose Index (HBGI)
    def bgi(self,
            per_day: bool = False,
            index_type:str = 'h',
            maximum: bool = False):
        '''
        Calculates the Low Blood Glucose Index (LBGI) or the High Blood Glucose Index (LBGI).

        .. math::

            LBGI = \\frac{1}{N} \\sum_{i=1}^N rl(X_i)

        .. math::

            HBGI = \\frac{1}{N} \\sum_{i=1}^N rh(X_i)

        - :math:`N` is the number of glucose readings.
        - :math:`rl(X_i) = 22.77 * f(X_i)^2` if :math:`f(X_i) < 0` and :math:`0` otherwise.
        - :math:`rh(X_i) = 22.77 * f(X_i)^2` if :math:`f(X_i) > 0` and :math:`0` otherwise.
        - :math:`f(X_i) = 1.509 * (\\ln(X_i)^{1.084} - 5.381)` for glucose readings in mg/dL.
        - :math:`X_i` is the glucose value at time i.

        Parameters
        ----------
        per_day : bool, default False
            If True, returns a pandas Series with the LBGI for each day. If False, returns the BGI for all days combined.
        index_type : str, default 'h'
            Type of index to calculate. Can be 'h' (High Blood Glucose Index) or 'l' (Low Blood Glucose Index).
        maximum : bool, default False
            If True, returns the maximum LBGI or HBGI. If False, returns the mean LBGI or HBGI.

        Returns
        -------
        bgi : float | pandas.Series
            Low Blood Glucose Index or High Blood Glucose Index.

        Notes
        -----
        * Using :meth:`glucopy.Gframe.lbgi` is equivalent to using bgi(index_type='l').
        * Using :meth:`glucopy.Gframe.hbgi` is equivalent to using bgi(index_type='h').

        Examples
        --------
        Calculating the LBGI for the entire dataset:

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.bgi(index_type='l')

        Calculating the HBGI for each day:

        .. ipython:: python

            gf.bgi(index_type='h', per_day=True)
        '''        
        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')

            bgi = pd.Series(dtype=float, name = 'HBGI' if index_type == 'h' else 'LBGI')
            bgi.index.name = 'Day'

            for day, day_data in day_groups:
                bgi[str(day)] = metrics.bgi(df=day_data, unit=self.unit, index_type=index_type, maximum=maximum)

        else: 
            bgi = metrics.bgi(df=self.data, unit=self.unit, index_type=index_type, maximum=maximum)

        return bgi
    
    # BGI Aliases
    def lbgi(self, 
             per_day: bool = False,
             maximum: bool = False):
        '''
        This is an alias for :meth:`glucopy.Gframe.bgi` with `index_type='l'`.
        '''
        return self.bgi(per_day=per_day, index_type='l', maximum=maximum)
    
    def hbgi(self,
             per_day: bool = False,
             maximum: bool = False):
        '''
        This is an alias for :meth:`glucopy.Gframe.bgi` with `index_type='h'`.
        '''
        return self.bgi(per_day=per_day, index_type='h', maximum=maximum)
        
    # Average Daily Risk Range (ADRR)
    def adrr(self):
        '''
        Calculates the Average Daily Risk Range (ADRR).

        .. math::

            ADRR = \\frac{1}{D} \\sum_{d=1}^D LR_d + HR_d

        - :math:`D` is the number of days.        
        - :math:`LR_d = max(rl(X_1),...,rl(X_N))` for glucose readings :math:`X_1,...X_N` taken within a day :math:`d = 1,...,D`
        - :math:`HR_d = max(rh(X_1),...,rh(X_N))` for glucose readings :math:`X_1,...X_N` taken within a day :math:`d = 1,...,D`
        - :math:`N` is the number of glucose readings within a day.
        - The definition of :math:`rl(X_i)` and :math:`rh(X_i)` is the same as in :meth:`glucopy.Gframe.bgi`.

        Parameters
        ----------
        None

        Returns
        -------
        adrr : float
            ADRR.

        Examples
        --------
        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.adrr()  
        ''' 

        adrr = self.bgi(per_day=True,index_type='h', maximum=True) \
             + self.bgi(per_day=True, index_type='l', maximum=True)
        
        return np.mean(adrr)

    # Glycaemic Risk Assessment Diabetes Equation (GRADE)
    def grade(self,
              percentage: bool = True):
        '''
        Calculates the contributions of the Glycaemic Risk Assessment Diabetes Equation (GRADE) to Hypoglycaemia,
        Euglycaemia and Hyperglycaemia. Or the GRADE scores for each value.

        .. math::

            GRADE = 425 * [\\log_{10}(\\log_{10} (X_i) + 0.16)]^2

        - :math:`X_i` is the glucose value at time i in mmol/L.

        The GRADE contribution percentages are calculated as follows:

        .. math::

            Hypoglycaemia \\% = 100 * \\frac{\\sum GRADE(X_i < 3.9 [mmol/L])}{\\sum GRADE(X_i)}

        .. math::

            Euglycaemia \\% = 100 * \\frac{\\sum GRADE(3.9 [mmol/L] <= X_i <= 7.8 [mmol/L])}{\\sum GRADE(X_i)}

        .. math::

            Hyperglycaemia \\% = 100 * \\frac{\\sum GRADE(X_i > 7.8 [mmol/L])}{\\sum GRADE(X_i)}

        Parameters
        ----------
        percentage : bool, default True
            If True, returns a pandas.Series of GRADE score contribution percentage for Hypoglycaemia, Euglycaemia and 
            Hyperglycaemia. If False, returns a list of GRADE scores for each value.

        Returns
        -------
        grade : pandas.Series
            Series of GRADE for each day.

        Examples
        --------
        Calculating the contributions of GRADE to Hypoglycaemia, Euglycaemia and Hyperglycaemia:

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.grade()
        
        Calculating the GRADE scores for each value:

        .. ipython:: python

            gf.grade(percentage=False)
        '''
        return metrics.grade(df=self.data, percentage=percentage, unit=self.unit)

    # [3.9,8.9] mmol/L -> [70.2,160.2] mg/dL
    # Q-Score Glucose=180.15588[g/mol] | 1 [mg/dL] -> 0.05551 [mmol/L] | 1 [mmol/L] -> 18.0182 [mg/dL]
    def qscore(self,
               slack: int = 0):
        '''
        Calculates the Q-Score.

        .. math::

            Q{-}score = 8 + \\frac{\\bar x -7.8}{1.7} + \\frac{Range - 7.5}{2.9} + \\frac{t_{G<3.9} - 0.6}{2.9} + 
                            \\frac{t_{G>8.9} - 6.2}{5.7} + \\frac{MODD - 1.8}{0.9}

        - :math:`\\bar x` is the mean glucose.
        - :math:`Range` is the mean of the differences between the maximum and minimum glucose for each day.
        - :math:`t_{G<3.9}` is the mean time [h] spent under 3.9 mmol/L in each day.
        - :math:`t_{G>8.9}` is the mean time [h] spent over 8.9 mmol/L in each day.
        - :math:`MODD` is the Mean of Daily Differences (:meth:`glucopy.Gframe.modd`).

        Parameters
        ----------
        slack : int, default 0
            Maximum number of minutes that the given time can differ from the actual time in the data in the calculation
            of :meth:`glucopy.Gframe.modd`.

        Returns
        -------
        qscore : float
            Q-Score.

        Examples
        --------
        Calculating the Q-Score with a 5 minutes slack for MODD:

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.qscore(slack=5)
        '''
        # Time in range [70.2,160.2] mg/dL = [3.9,8.9] mmol/L
        if self.unit == 'mmol/L':
            interval = [3.9,8.9]
        elif self.unit == 'mg/dL':
            interval = [70.2,160.2]

        tir_per_day = self.tir(per_day=True, interval=interval, percentage=False)

        # List with the Timedelta corresponding to the time spent under 3.9 mmol/L for each day
        tir_per_day_minus_3_9 = [time[0].total_seconds() for time in tir_per_day] 
        # Mean of the previous list in hours
        hours_minus_3_9 = np.mean(tir_per_day_minus_3_9) / 3600
        
        # List with the Timedelta corresponding to the time spent under 8.9 mmol/L for each day
        tir_per_day_plus_8_9 = [time[2].total_seconds() for time in tir_per_day] 
        # Mean of the previous list in hours
        hours_plus_8_9 = np.mean(tir_per_day_plus_8_9) / 3600
        
        # Calculate the difference between max and min for each day (range)
        differences = self.data.groupby('Day')['CGM'].apply(lambda x: x.max() - x.min())

        # Mean of Max-Min for each day
        mean_difference = differences.mean()

        # Mean
        mean = self.mean()

        # MODD
        modd = self.modd(slack=slack)

        # Convert to mmol/L
        if self.unit == 'mg/dL':
            mean = mgdl_to_mmoll(mean)
            mean_difference = mgdl_to_mmoll(mean_difference)
            modd = mgdl_to_mmoll(modd)

        # fractions
        f1 = (mean - 7.8 ) / 1.7

        f2 = (mean_difference - 7.5) / 2.9

        f3 = (hours_minus_3_9 - 0.6) / 1.2
        
        f4 = (hours_plus_8_9 - 6.2) / 5.7

        f5 = (modd - 1.8) / 0.9

        return 8 + f1 + f2 + f3 + f4 + f5


    # 5. Metrics for the analysis of glycaemic dynamics using variability estimation.

    # Mean absolute relative deviation (MARD)
    def mard(self,
             smbg_df: pd.DataFrame,
             slack: int = 0,
             interpolate: bool = False):
        '''
        Calculates the Mean Absolute Relative Difference (MARD).

        .. math::

            MARD = \\frac{1}{N} \\sum_{i=1}^N \\frac{|CGM_i - SMBG_i|}{SMBG_i} * 100

        - :math:`N` is the number of SMBG readings.
        - :math:`CGM_i` is the Continuous Glucose Monitoring (CGM) value at time i.
        - :math:`SMBG_i` is the Self Monitoring of Blood Glucose (SMBG) value at time i.

        Parameters
        ----------
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

        Examples
        --------
        Calculating the MARD with a 5 minutes slack and without interpolation:

        .. ipython:: python

            import glucopy as gp
            import pandas as pd
            gf = gp.data('prueba_1')
            smbg_timestamps = pd.to_datetime(['2020-11-27 22:00:00', 
                                              '2020-11-28 01:00:00', 
                                              '2020-11-28 04:00:00'])
            smbg_df = pd.DataFrame({'Timestamp': smbg_timestamps,
                                    'SMBG': [260, 239, 135]})
            gf.mard(smbg_df=smbg_df, slack=5, interpolate=False)

        Calculating the MARD with a 5 minutes slack and with interpolation:

        .. ipython:: python

            gf.mard(smbg_df=smbg_df, slack=5, interpolate=True)
        '''
        return metrics.mard(cgm_df=self.data, smbg_df=smbg_df, slack=slack, interpolate=interpolate)


    # Continuous Overall Net Glycaemic Action (CONGA)    
    def conga(self,
              per_day: bool = False,
              m: int = 1,
              slack: int = 0,
              ignore_na: bool = True,
              ddof: int = 1):
        '''
        Calculates the Continuous Overall Net Glycaemic Action (CONGA).

        .. math::

            CONGA = \\sqrt{\\frac{1}{k-ddof} \\sum_{t=t1} (D_t - \\bar D)^2}

        - :math:`ddof` is the Delta Degrees of Freedom.
        - :math:`D_t` is the difference between glycaemia at time `t` and `t` minus `m` hours ago.
        - :math:`\\bar D` is the mean of the differences (:math:`D_t`).
        - :math:`k` is the number of differences.

        Parameters
        ----------
        per_day : bool, default False
            If True, returns the CONGA for each day separately. If False, returns the CONGA for all days combined.
        m : int, default 1
            Number of hours to use for the CONGA calculation.
        slack : int, default 0
            Maximum number of minutes that the given time can differ from the actual time in the data.
        ignore_na : bool, default True
            If True, ignores missing values (not found within slack). If False, raises an error 
            if there are missing values.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations of standard deviation is N - ddof, where N 
            represents the number of elements.

        Returns
        -------
        conga : list
            List of CONGA for each day.

        Examples
        --------
        Calculating the CONGA for the entire dataset (default):

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.conga()

        Calculating the CONGA for each day with a 5 minutes slack:

        .. ipython:: python

            gf.conga(per_day=True, slack=5)
        '''
        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')
            conga = pd.Series(dtype=float, name='CONGA')
            conga.index.name = 'Day'

            for day, day_data in day_groups:
                conga[str(day)] = metrics.conga(df=day_data, m=m, slack=slack, ignore_na=ignore_na, ddof=ddof)
        
        else:
            conga = metrics.conga(df=self.data, m=m, slack=slack, ignore_na=ignore_na, ddof=ddof)

        return conga

    # Glucose Variability Percentage (GVP)
    def gvp(self):
        '''
        Calculates the Glucose Variability Percentage (GVP), with time in minutes.

        .. math::

            GVP = \\left( \\frac{L}{T_0} - 1\\right) * 100

        - :math:`L = \\sum_{i=1}^N \\sqrt{\\Delta X_i^2 + \\Delta T_i^2}`
        - :math:`T_0 = \\sum_{i=1}^N \\Delta T_i`
        - :math:`N` is the number of glucose readings.
        - :math:`\\Delta X_i` is the difference between glucose values at time i and i-1.
        - :math:`\\Delta T_i` is the difference between times at time i and i-1.

        Parameters
        ----------
        None

        Returns
        -------
        gvp : float
            Glucose Variability Percentage.

        Examples
        --------
        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.gvp()
        '''
        return metrics.gvp(df=self.data)
    
    # Mean Absolute Glucose Change per unit of time (MAG)
    def mag(self,
            per_day: bool = False,
            time_unit: str = 'm'):
        '''
        Calculates the Mean Absolute Glucose Change per unit of time (MAG).

        .. math::

            MAG = \\sum_{i=1}^{N} \\frac{|\\Delta X_i|}{\\Delta T_i}

        - :math:`N` is the number of glucose readings.
        - :math:`\\Delta X_i` is the difference between glucose values at time i and i-1.
        - :math:`\\Delta T_i` is the difference between times at time i and i-1.

        Parameters
        ----------
        per_day : bool, default False
            If True, returns the an array with the MAG for each day. If False, returns the MAG for all days combined.
        time_unit : str, default 'm' (minutes)
            The time time_unit for the x-axis. Can be 's (seconds)', 'm (minutes)', or 'h (hours)'.
        
        Returns
        -------
        mag : float
            Mean Absolute Glucose Change per unit of time.

        Examples
        --------
        Calculating the MAG for the entire dataset and minutes as the time unit (default):

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.mag()

        Calculating the MAG for the entire dataset and hours as the time unit:

        .. ipython:: python

            gf.mag(time_unit='h')

        Calculating the MAG for each day and minutes as the time unit:

        .. ipython:: python

            gf.mag(per_day=True)
        '''
        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')
            mag = pd.Series(dtype=float, name='MAG')
            mag.index.name = 'Day'

            for day, day_data in day_groups:
                mag[str(day)] = metrics.mag(df=day_data, time_unit=time_unit)
        
        else:
            mag = metrics.mag(df=self.data, time_unit=time_unit)
            
        return mag


    # 6. Computational methods for the analysis of glycemic dynamics
        
    # Detrended fluctuation analysis (DFA)
    def dfa(self,
            per_day: bool = False,
            scale = 'default',
            overlap: bool = True,
            integrate: bool = True,
            order: int = 1,
            show: bool = False,
            **kwargs):
        '''
        Calculates the Detrended Fluctuation Analysis (DFA) using neurokit2.fractal_dfa().

        For more information on the parameters and details of the neurokit2.fractal_dfa() method, 
        see the neurokit2 documentation: 
        `neurokit2.fractal_dfa() <https://neuropsychology.github.io/NeuroKit/functions/complexity.html#neurokit2.complexity.fractal_dfa>`_.

        Parameters
        ----------
        per_day : bool, default False
            If True, returns the an array with the DFA for each day. If False, returns the DFA for all days combined. If
            a day has very few data points, the DFA for that day will be NaN.

        Returns
        -------
        dfa : float | pandas.Series
            Detrended fluctuation analysis.

        Examples
        --------
        Calculating the DFA for the entire dataset:

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.dfa()

        Calculating the DFA for each day:

        .. ipython:: python

            gf.dfa(per_day=True)
        
        Calculating and showing the DFA for the entire dataset:

        .. ipython:: python

            gf.dfa(show=True)

        .. plot::
           :context: close-figs

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.dfa(show=True)
        '''

        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')

            dfa = pd.Series(dtype=float,name='DFA')
            dfa.index.name = 'Day'

            for day, day_data in day_groups:
                dfa[str(day)] = metrics.dfa(df=day_data,
                                       scale=scale,
                                       overlap=overlap,
                                       integrate=integrate,
                                       order=order,
                                       show=show,
                                       **kwargs)
        
        else:
            dfa = metrics.dfa(df=self.data,
                              scale=scale,
                              overlap=overlap,
                              integrate=integrate,
                              order=order,
                              show=show,
                              **kwargs)
            
        return dfa
            
    # Entropy Sample (SampEn)
    def samp_en(self,
                per_day: bool = False,
                delay: int | None = 1,
                dimension: int | None = 2,
                tolerance: float | str | None = 'sd',
                **kwargs):
        '''
        Calculates the Sample Entropy using neurokit2.entropy_sample()

        For more information on the parameters and details of the neurokit2.entropy_sample() method, 
        see the `neurokit2 documentation <https://neuropsychology.github.io/NeuroKit/functions/complexity.html#neurokit2.complexity.entropy_sample>`_.

        Parameters
        ----------
        per_day : bool, default False
            If True, returns the an array with the Sample Entropy for each day. If False, returns the Sample Entropy for
            all days combined.
        
        Returns
        -------
        samp_en : float | pandas.Series
            Entropy Sample.

        Examples
        --------
        Calculating the Sample Entropy for the entire dataset:

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.samp_en()

        Calculating the Sample Entropy for each day:

        .. ipython:: python

            gf.samp_en(per_day=True)
        '''
        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')

            samp_en = pd.Series(dtype=float,name='Entropy Sample')
            samp_en.index.name = 'Day'

            for day, day_data in day_groups:
                samp_en[str(day)] = metrics.samp_en(df=day_data,
                                               delay=delay,
                                               dimension=dimension,
                                               tolerance=tolerance,
                                               **kwargs)   

        else:
            samp_en = metrics.samp_en(df=self.data,
                                      delay=delay,
                                      dimension=dimension,
                                      tolerance=tolerance,
                                      **kwargs)
        
        return samp_en
        
    # Multiscale Sample Entropy (MSE)
    def mse(self,
            per_day: bool = False,
            scale = 'default',
            dimension = 3,
            tolerance = 'sd',
            method = 'MSEn',
            show = False,
            **kwargs):
        '''
        Calculates the Multiscale Sample Entropy using neurokit2.entropy_multiscale()

        For more information on the parameters and details of the neurokit2.entropy_sample() method, 
        see the neurokit2 documentation: 
        `neurokit2.entropy_multiscale() <https://neuropsychology.github.io/NeuroKit/functions/complexity.html#entropy-multiscale>`_.

        Parameters
        ----------
        per_day : bool, default False
            If True, returns the an array with the Multiscale Sample Entropy for each day. If False, returns the 
            Multiscale Sample Entropy for all days combined.        
            
        Returns
        -------
        mse : float | pandas.Series
            Multiscale Sample Entropy.

        Examples
        --------
        Calculating the Multiscale Sample Entropy for the entire dataset:

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.mse()

        Calculating the Multiscale Sample Entropy for each day:

        .. ipython:: python

            gf.mse(per_day=True)
        
        Calculating and showing the Multiscale Sample Entropy for the entire dataset:

        .. ipython:: python

            gf.mse(show=True)

        .. plot::
            :context: close-figs

                import glucopy as gp
                gf = gp.data('prueba_1')
                gf.mse(show=True)

        '''
        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')

            mse = pd.Series(dtype=float,name='MSE')
            mse.index.name = 'Day'

            for day, day_data in day_groups:
                mse[str(day)] = metrics.mse(df=day_data,
                                       scale=scale,
                                       dimension=dimension,
                                       tolerance=tolerance,
                                       method=method,
                                       show=show,
                                       **kwargs)

        else:
            mse = metrics.mse(df=self.data,
                              scale=scale,
                              dimension=dimension,
                              tolerance=tolerance,
                              method=method,
                              show=show,
                              **kwargs)
            
        return mse

    # Summary
    def summary(self,
                auc_time_unit: str = 'm',
                mag_time_unit: str = 'h',
                slack: int = 0,
                decimals: int | None = 2):
        '''
        Calculates a summary of the metrics for the entire dataset or for each day separately.

        Parameters
        ----------
        auc_time_unit : str, default 'm' (minutes)
            The time unit for the calculation of AUC. Can be 's (seconds)', 'm (minutes)', or 'h (hours)'.
        mag_time_unit : str, default 'h' (hours)
            The time unit for the calculation of MAG. Can be 's (seconds)', 'm (minutes)', or 'h (hours)'.
        slack : int, default 0
            Maximum number of minutes that the given time can differ from the actual time in the data in the calculation
            of MODD, CONGA and Q-Score (uses MODD).
        
        decimals : int | None, default 2
            Number of decimals to round the values to. If None, the values will not be rounded.
        
        Returns
        -------
        summary : pandas.DataFrame
            Summary of the metrics.

        Examples
        --------
        Calculating the summary for the entire dataset and minutes as the time unit (default):

        .. ipython:: python

            import glucopy as gp
            gf = gp.data('prueba_1')
            gf.summary()
        '''
        # Time in range [70,180] mg/dL
        tir_interval = np.array([0, 70, 180])
        if self.unit == 'mmol/L':
            tir_interval = mgdl_to_mmoll(tir_interval)

        # Metrics that return Series
        tir = self.tir(interval=tir_interval)
        grade = self.grade()

        # Summary
        summary = [['Mean', self.mean()],
                   ['Standard Deviation', self.std()],
                   ['Coefficient of Variation', self.cv()],
                   ['IQR', self.iqr()],
                   ['MODD', self.modd(slack=slack)],
                   ['% Time below 70 [mg/dL]', tir.iloc[0]],
                   ['% Time in between (70,180] [mg/dL]', tir.iloc[1]],
                   ['% Time above 180 [mg/dL]', tir.iloc[2]],
                   ['AUC', self.auc(time_unit=auc_time_unit)],
                   ['MAGE', self.mage()],
                   ['Distance Traveled', self.dt()],
                   ['LBGI', self.lbgi()],
                   ['HBGI', self.hbgi()],
                   ['ADRR', self.adrr()],
                   ['GRADE Hypoglycaemia %', grade.iloc[0]],
                   ['GRADE Euglycaemia %', grade.iloc[1]],
                   ['GRADE Hyperglycaemia %', grade.iloc[2]],
                   ['Q-Score', self.qscore(slack=slack)],
                   ['CONGA', self.conga(slack=slack)],
                   ['GVP', self.gvp()],
                   ['MAG', self.mag(time_unit=mag_time_unit)],
                   ['DFA', self.dfa()],
                   ['SampEn', self.samp_en()],
                   ['MSE', self.mse()]
                ]

        return pd.DataFrame(data=summary, columns = ['Metric', 'Value']).round(decimals=decimals)




    

    
