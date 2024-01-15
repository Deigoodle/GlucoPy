#3rd party
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import find_peaks
from scipy.stats import linregress

# Built-in
from collections.abc import Sequence
import datetime
from typing import List

# Local
from glucopy.utils import disjoin_days_and_hours
from glucopy.utils import (str_to_time,
                           time_to_str,
                           mgdl_to_mmoll, 
                           mmoll_to_mgdl)

class Gframe:
    '''
    Description

    Parameters
    -----------
    data : pandas Dataframe 
        Dataframe containing the CGM signal information, it will be saved into a Dataframe with the columns 
        ['Timestamp','Day','Time','CGM']
    unit : String, default 'mg/dL'
        CGM signal measurement unit.
    date_column : String or String array, default None
        The name or names of the column(s) containing the date information
        If it's a String, it will be the name of the single column containing the date information
        If it's a String array, it will be the 2 names of the columns containing the date information, eg. ['Date','Time']
        If it's None, it will be assumed that the date information is in the first column
    cgm_column : String, default None
        The name of the column containing the CGM signal information
        If it's None, it will be assumed that the CGM signal information is in the second column
    dropna : bool, default True
        If True, removes all rows with NaN values
    date_format : String, default None
        Format of the date information, if None, it will be assumed that the date information is in a consistent format
    '''

    # Constructor
    def __init__(self, 
                 data=None, 
                 unit:str = 'mg/dL',
                 date_column: list[str] | str | int = 0,
                 cgm_column: str | int = 1,
                 dropna:bool = True):
        
        # Check data is a dataframe
        if isinstance(data, pd.DataFrame):
            # Check date_column
            if isinstance(date_column, str) or isinstance(date_column, int):
                self.data = disjoin_days_and_hours(data, date_column, cgm_column)

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
    def __str__(self):
        return str(self.data)
    
    # Metrics 
    # -------
    # 1. Joint data analysis metrics for glycaemia dynamics

    # Sample Mean
    def mean(self,
             per_day: bool = False,
             **kwargs):
        '''
        Parameters
        ----------
        per_day : bool, default False
            If True, returns a pandas Series with the mean for each day. If False, returns the mean for all days combined.
        **kwargs : dict
            Additional keyword arguments to be passed to the function. For more information view the documentation for
            pandas.DataFrameGroupBy.mean().

        Returns
        -------
        mean : float | pandas.Series
            Mean of the CGM values.            
        '''

        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')
            mean = day_groups['CGM'].mean(**kwargs)

        else:
            mean = self.data['CGM'].mean(**kwargs)

        return mean
    
    # Standard Deviation, by default ddof=1, so its divided by n-1
    def std(self,
            per_day: bool = False,
            ddof:int = 1,
            **kwargs):
        '''
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
        '''
        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')
            std = day_groups['CGM'].std(ddof=ddof,**kwargs)
        
        else:
            std = self.data['CGM'].std(ddof=ddof,**kwargs)

        return std
    
    # Coefficient of Variation
    def cv(self,
           per_day: bool = False,
           ddof:int = 1,
           **kwargs):
        '''
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
        '''
        if per_day:
            cv = self.std(per_day=True,ddof=ddof,**kwargs)/self.mean(per_day=True,**kwargs)
          
        else:
            cv = self.std(ddof=ddof,**kwargs)/self.mean(**kwargs)

        return cv
            
    # % Coefficient of Variation
    def pcv(self,
            per_day: bool = False,
            ddof:int = 1,
            **kwargs):
        '''
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
        '''
        if per_day:
            pcv = self.cv(per_day=True,ddof=ddof,**kwargs) * 100
          
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
        '''
        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')
            quantile = day_groups['CGM'].quantile(q=q, interpolation=interpolation, **kwargs)
        
        else:
            quantile = self.data['CGM'].quantile(q=q, interpolation=interpolation, **kwargs)
        
        return quantile
    
    # Interquartile Range
    def iqr(self,
            per_day: bool = False,
            interpolation:str = 'linear',
            **kwargs):
        '''
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
        iqr : float | list
            Interquartile range of the CGM values.

        '''
        
        q1 = self.quantile(per_day=per_day,q=0.25, interpolation=interpolation, **kwargs)
        q3 = self.quantile(per_day=per_day,q=0.75, interpolation=interpolation, **kwargs)
        
        return q3 - q1
    
    # Mean of Daily Differences
    def modd(self, 
             target_time: str | datetime.time | None = None, 
             slack: int = 0,
             ignore_na: bool = True) -> float:
        '''
        Calculates the Mean of Daily Differences (MODD) for a given time of day.

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
        '''

        # Check input
        if slack < 0:
            raise ValueError('slack must be a positive number or 0')

        if target_time is None: # calculate MODD for all times
            unique_times = self.data['Time'].unique()
            modd_values = []
            for time in unique_times:
                modd_values.append(self.modd(time, slack))

            modd = np.mean(modd_values)
        
        else: # calculate MODD for a given time
            # convert time to same format as self.data['Time']
            if not isinstance(target_time, str) and not isinstance(target_time, datetime.time):
                raise TypeError('time must be a string or a datetime.time')
            elif isinstance(target_time, str):# String -> datetime.time
                target_str = target_time
                target_time = str_to_time(target_str)
            else: # datetime.time
                target_str = time_to_str(target_time)

            # convert slack to timedelta
            slack = pd.to_timedelta(slack, unit='m')

            cgm_values: List[float] = []

            # search given time in each day
            day_groups = self.data.groupby('Day')
            for day, day_data in day_groups:
                target_time_index = day_data['Time'] == target_time
                # if exact time is found, use it
                if target_time_index.any():
                    cgm_values.append(day_data.loc[target_time_index, 'CGM'].values[0])
                # if not, search for closest time within error range
                elif slack > pd.Timedelta('0 min'):
                    # combine "day" and target_time to compare it with Timestamp
                    target_date = str(day) + ' ' + target_str
                    target_datetime = pd.to_datetime(target_date)
                    # search for closest time within error range
                    mask_range = ((day_data['Timestamp'] - target_datetime).abs() <= slack)
                    if mask_range.any():
                        closest_index = (day_data.loc[mask_range, 'Timestamp'] - target_datetime).abs().idxmin()
                        cgm_values.append(day_data.loc[closest_index, 'CGM'])
                    else:
                        if not ignore_na:
                            raise ValueError(f"No data found for date {day}")

            modd = np.sum(np.abs(np.diff(cgm_values))) / (self.n_days)
        
        return modd

    # Time in Range
    def tir(self, 
            per_day: bool = True,
            target_range:list= [0,70,180,350],
            percentage: bool = True,
            decimals: int = 2):
        '''
        Calculates the Time in Range (TIR) for a given target range of glucose for each day.

        Parameters
        ----------
        per_day : bool, default False
            If True, returns a pandas Series with the TIR for each day. If False, returns the TIR for all days combined.
        target_range : list of int|float, default [0,70,180,350]
            Target range in CGM unit for low, normal and high glycaemia. It must have at least 2 values, for the "normal"
            range, low and high values will be values outside that range.
        percentage : bool, default True
            If True, returns the TIR as a percentage. If False, returns the TIR as time.
        decimals : int, default 2
            Number of decimal places to round to. Use None for no rounding.

        Returns
        -------
        tir : pandas.Series 
            Series of TIR for each day, indexed by day.
        '''
        # Check input, Ensure target_range is a list with 0 and the max value of the data
        if not isinstance(target_range, list) or not all(isinstance(i, (int, float)) for i in target_range):
            raise ValueError("target_range must be a list of numbers")
        if 0 not in target_range:
            target_range = [0] + target_range
        if max(self.data['CGM']) > target_range[-1]:
            target_range = target_range + [max(self.data['CGM'])]

        if per_day:
            day_groups = self.data.groupby('Day')

            tir = pd.Series(dtype=float)
            for day, day_data in day_groups:
                day_data['Time_Diff'] = day_data['Timestamp'].diff().dt.total_seconds() 
                day_data['ranges'] = pd.cut(day_data['CGM'], bins=target_range)
                time_count = day_data.groupby('ranges', observed=False)['Time_Diff'].sum()
                if percentage:
                    result = np.array(time_count / time_count.sum()) * 100
                    if decimals is not None:
                        result = np.round(result, decimals=decimals)
                else:
                    
                    result = np.array(time_count.apply(lambda x: str(datetime.timedelta(seconds=x))))
                tir[day] = result
                    
        else:
            data_copy = self.data.copy()
            data_copy['Time_Diff'] = data_copy['Timestamp'].diff().dt.total_seconds()
            data_copy['ranges'] = pd.cut(data_copy['CGM'], bins=target_range)
            time_count = data_copy.groupby('ranges', observed=False)['Time_Diff'].sum()
            if percentage:
                result = np.array(time_count / time_count.sum()) * 100
                if decimals is not None:
                    tir = np.round(result, decimals=decimals)
            else:
                tir = time_count.apply(lambda x: str(datetime.timedelta(seconds=x)))
        
        return tir

    
    # 2. Analysis of distribution in the plane for glycaemia dynamics.

    # Frecuency distribution : counts the amount of observations given certain intervals of CGM
    def fd(self,
           per_day: bool = True,
           target_range: list = [0,70,180,350],
           decimals: int = 2):
        '''
        Calculates the Frequency Distribution (FD) for a given target range of glucose.

        Parameters
        ----------
        per_day : bool, default False
            If True, returns a pandas Series with the FD for each day. If False, returns the FD for all days combined.
        target_range : list of int|float, default [0,70,180,350]
            Target range in CGM unit. It must have at least 2 values, for the "normal"
            range, low and high values will be values outside that range.
        decimals : int, default 2
            Number of decimal places to round to. Use None for no rounding.

        Returns
        -------
        tir : pandas.Series 
            Series of TIR for each day, indexed by day.
        '''
        # Check input, Ensure target_range is a list with 0 and the max value of the data
        if not isinstance(target_range, list) or not all(isinstance(i, (int, float)) for i in target_range):
            raise ValueError("target_range must be a list of numbers")
        if 0 not in target_range:
            target_range = [0] + target_range
        if max(self.data['CGM']) > target_range[-1]:
            target_range = target_range + [max(self.data['CGM'])]

        if per_day:
            day_groups = self.data.groupby('Day')

            # Initialize fd as an empty Series
            fd = pd.Series(dtype=float)

            for day, day_data in day_groups:
                day_data['ranges'] = pd.cut(day_data['CGM'], bins=target_range)
                result = day_data.groupby('ranges', observed=False)['ranges'].count()
                if decimals is not None:
                    fd[day] = np.round(np.array(result / result.sum()), decimals=decimals)
                else:
                    fd[day] = np.array(result / result.sum())

        
        else:
            result = (pd.cut(self.data['CGM'], bins=target_range)
                        .groupby(pd.cut(self.data['CGM'], bins=target_range), observed=False).count())
            summed_results = result.sum()
            if decimals is not None:
                fd = (result / summed_results).round(decimals=decimals)
            else:
                fd = result / summed_results
            
        return fd


    # Ambulatory Glucose Profile (AGP)
    def agp(self):
        pass

    # Area Under the Curve (AUC)
    def auc(self, time_unit='m'):
        '''
        Calculates the Area Under the Curve (AUC) for each day.

        Parameters
        ----------
        time_unit : str, default 'm' (minutes)
            The time unit for the x-axis. Can be 's (seconds)', 'm (minutes)', or 'h (hours)'.

        Returns
        -------
        auc : list 
            List of AUC for each day.
        '''
        # Determine the factor to multiply the total seconds by
        if time_unit == 's':
            factor = 1
        elif time_unit == 'm':
            factor = 60
        elif time_unit == 'h':
            factor = 3600
        else:
            return "Error: Invalid time unit. Must be 's', 'm', or 'h'."

        # Group data by day
        day_groups = self.data.groupby('Day')

        # Initialize auc as an empty Series
        auc = pd.Series(dtype=float)

        # Calculate AUC for each day
        for day, day_data in day_groups:
            # Convert timestamps to the specified time unit
            time_values = (day_data['Timestamp'] - day_data['Timestamp'].min()).dt.total_seconds() / factor
            auc[day] = np.trapz(y=day_data['CGM'], x=time_values)

        return auc


    # 3. Amplitude and distribution of frequencies metrics for glycaemia dynamics.

    # Mean Amplitude of Glycaemic Excursions (MAGE)
    def mage(self):
        '''
        Calculates the Mean Amplitude of Glycaemic Excursions (MAGE) for each day.

        Parameters
        ----------
        None

        Returns
        -------
        mage : list 
            List of MAGE for each day.
        '''
        # Group data by day
        day_groups = self.data.groupby('Day')

        # Initialize mage as an empty Series
        mage = pd.Series(dtype=float)

        # Calculate MAGE for each day
        
        for day, day_data in day_groups:
            day_std = day_data['CGM'].std()
            
            # find peaks and nadirs
            peaks, _ = find_peaks(day_data['CGM'])
            nadirs, _ = find_peaks(-day_data['CGM'])

            if peaks.size > nadirs.size:
                nadirs = np.append(nadirs, day_data['CGM'].size - 1)
            elif peaks.size < nadirs.size:
                peaks = np.append(peaks, day_data['CGM'].size - 1)
            
            # calculate the difference between the peaks and the nadirs
            differences = np.abs(day_data['CGM'].iloc[peaks].values - day_data['CGM'].iloc[nadirs].values)
            # get differences greater than std
            differences = differences[differences > day_std]
            # calculate mage
            mage[day] = differences.mean()

        return mage

    # Distance Travelled (DT)
    def dt(self,
           per_day: bool = True):
        '''
        Calculates the Distance Travelled (DT) for each day.

        Parameters
        ----------
        per_day : bool, default False
            If True, returns a pandas Series with the DT for each day. If False, returns the DT for all days combined.

        Returns
        -------
        dt : list 
            List of DT for each day.
        '''
        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')

            dt = pd.Series(dtype=float)

            # Calculate DT for each day
            for day, day_data in day_groups:
                dt[day] = np.sum(np.abs(np.diff(day_data['CGM'])))

        else:
            dt = np.sum(np.abs(np.diff(self.data['CGM'])))

        return dt
    
    # 4. Metrics for the analysis of glycaemic dynamics using scores of glucose values

    # Low Blood Glucose Index (LBGI) and High Blood Glucose Index (HBGI)
    def bgi(self,
            per_day: bool = True,
            index_type:str = 'h',
            maximum: bool = False):
        '''
        Calculates the Low Blood Glucose Index (LBGI) for each day. Only works for CGM values in mg/dL.

        Parameters
        ----------
        per_day : bool, default False
            If True, returns a pandas Series with the LBGI for each day. If False, returns the BGI for all days combined.
        index_type : str, default 'h'
            Type of index to calculate. Can be 'h' (High Blood Glucose Index) or 'l' (Low Blood Glucose Index).
        maximum : bool, default False
            If True, returns the maximum LBGI or HBGI for each day. If False, returns the mean LBGI or HBGI for each day.

        Returns
        -------
        bgi : list 
            List of LBGI or HBGI for each day.
        '''
        index_type.lower()
        if index_type != 'h' and index_type != 'l':
            raise ValueError('index_type must be "h" or "l"')
        
        def f(x,index_type):
            result = np.power(np.log(x), 1.084) - 5.381
            if result >= 0 and index_type == 'l':
                result = 0
            elif result <= 0 and index_type == 'h':
                result = 0
            return result
        
        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')

            bgi = pd.Series(dtype=float)
            for day, day_data in day_groups:
                values = day_data['CGM'].values
                if self.unit == 'mmol/L':
                    values = mmoll_to_mgdl(values)

                f_values = np.vectorize(f)(values,index_type)
                risk = 22.77 * np.square(f_values)
                if maximum:
                    bgi[day] = np.max(risk)
                else:
                    bgi[day] = np.mean(risk)

        else: 
            values = self.data['CGM'].values
            if self.unit == 'mmol/L':
                values = mmoll_to_mgdl(values)

            f_values = np.vectorize(f)(values,index_type)
            risk = 22.77 * np.square(f_values)
            if maximum:
                bgi = np.max(risk)
            else:
                bgi = np.mean(risk)

        return bgi
    
    # BGI Aliases
    def lbgi(self, 
             per_day: bool = True,
             maximum: bool = False):
        return self.bgi(per_day=per_day, index_type='l', maximum=maximum)
    
    def hbgi(self,
             per_day: bool = True,
             maximum: bool = False):
        return self.bgi(per_day=per_day, index_type='h', maximum=maximum)
        
    # Average Daily Risk Range (ADRR)
    def adrr(self):
        '''
        Calculates the Average Daily Risk Range (ADRR) for each day. Only works for CGM values in mg/dL.

        Parameters
        ----------
        None

        Returns
        -------
        adrr : list
            List of ADRR for each day.
        ''' 

        adrr = self.bgi(index_type='h', maximum=True) + self.bgi(index_type='l', maximum=True)
        
        return np.mean(adrr)

    # Glycaemic Risk Assessment Diabetes Equation (GRADE)
    def grade(self,
              percentage: bool = True):
        '''
        Calculates the Glycaemic Risk Assessment Diabetes Equation (GRADE) for each day. Only works for CGM values in mg/dL.

        Parameters
        ----------
        percentage : bool, default True
            If True, returns a pandas.Series of GRADE score contribution percentage for Hypoglycaemia, Euglycaemia and 
            Hyperglycaemia. If False, returns a list of GRADE scores for each value.

        Returns
        -------
        grade : list
            List of GRADE for each day.
        '''
        values = self.data['CGM'].values
        if self.unit == 'mg/dL':
            values = mgdl_to_mmoll(values)
        grade = np.minimum(425 * np.square( np.log10( np.log10(values) ) + 0.16), 50)

        if percentage:
            grade_sum = np.sum(grade)
            hypo = np.sum(grade[values < 3.9]) / grade_sum 
            hyper = np.sum(grade[values > 7.8]) / grade_sum
            eugly = 1 - hypo - hyper
            grade = pd.Series([hypo, eugly, hyper], index=['Hypoglycaemia', 'Euglycaemia', 'Hyperglycaemia']) * 100
        
        return grade

    # [3.9,8.9] mmol/L -> [70.2,160.2] mg/dL
    # Q-Score Glucose=180.15588[g/mol] | 1 [mg/dL] -> 0.05551 [mmol/L] | 1 [mmol/L] -> 18.0182 [mg/dL]
    def qscore(self):
        '''
        Calculates the Q-Score.

        Parameters
        ----------
        None

        Returns
        -------
        qscore : float
            Q-Score.
        '''
        # Time spent under 3.9 mmol/L in hours
        time_minus_3_9 = self.tir(target_range=[70.2], percentage=False)\
                             .apply(lambda x: pd.to_timedelta(x[0]))\
                             .mean().total_seconds() / 3600
        
        # Time spent over 8.9 mmol/L in hours
        time_plus_8_9 = self.tir(target_range=[160.2], percentage=False)\
                            .apply(lambda x: pd.to_timedelta(x[1]))\
                            .mean().total_seconds() / 3600
        
        # Calculate the difference between max and min for each day (range)
        differences = self.data.groupby('Day')['CGM'].apply(lambda x: x.max() - x.min())
        mean_difference = mgdl_to_mmoll(differences.mean())

        # fractions
        f1 = (mgdl_to_mmoll(self.mean()) - 7.8 ) / 1.7

        f2 = (mean_difference - 7.5) / 2.9

        f3 = (time_minus_3_9 - 0.6) / 1.2
        
        f4 = (time_plus_8_9 - 6.2) / 5.7

        f5 = (mgdl_to_mmoll(self.modd()) - 1.8) / 0.9

        return 8 + f1 + f2 + f3 + f4 + f5


    # 5. Metrics for the analysis of glycaemic dynamics using variability estimation.

    # Continuous Overall Net Glycaemic Action (CONGA)
    def conga(self,
              per_day: bool = True,
              m: int = 1,
              slack: int = 0,
              method: str = 'closest'):
        '''
        Calculates the Continuous Overall Net Glycaemic Action (CONGA).

        Parameters
        ----------
        per_day : bool, default True
            If True, returns the CONGA for each day separately. If False, returns the CONGA for all days combined.
        m : int, default 1
            Number of hours to use for the CONGA calculation.
        slack : int, default 0
            Maximum number of minutes that the given time can differ from the actual time in the data.
        method : str, default 'closest'
            Method to use if there are multiple timestamps that are m hours before the current timestamp and within the
            slack range. Can be 'closest' or 'mean'.

        Returns
        -------
        conga : list
            List of CONGA for each day.
        '''
        # Check input
        if m < 0:
            raise ValueError('m must be a positive number')
        if slack < 0:
            raise ValueError('slack must be a positive number or 0')
        if method != 'closest' and method != 'mean':
            raise ValueError('method must be "closest" or "mean"')
        
        # Convert m and slack to timedelta
        m = pd.to_timedelta(m, unit='h')
        slack = pd.to_timedelta(slack, unit='m')

        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')
            conga = pd.Series(dtype=float)
            for day, day_data in day_groups:
                differences = []
                for i in range(1, day_data.shape[0]):
                    # find a previous timestamp that is m hours before the current timestamp and within the slack range
                    previous_index = (day_data['Timestamp'] >= day_data['Timestamp'].iloc[i] - m - slack) \
                                    &(day_data['Timestamp'] <= day_data['Timestamp'].iloc[i] - m + slack)
                    if previous_index.any():
                        if method == 'mean':
                            previous_value = day_data.loc[previous_index, 'CGM'].mean()
                        elif method == 'closest':
                            closest_index = (day_data.loc[previous_index, 'Timestamp'] - day_data['Timestamp'].iloc[i]).abs().idxmin()
                            previous_value = day_data.loc[closest_index, 'CGM']
                        # calculate the difference between the current value and the previous value
                        differences.append(day_data['CGM'].iloc[i] - previous_value)
                conga[day] = np.std(differences)
        
        else:
            differences = []
            for i in range(1, self.data.shape[0]):
                # find a previous timestamp that is m hours before the current timestamp and within the slack range
                previous_index = (self.data['Timestamp'] >= self.data['Timestamp'].iloc[i] - m - slack) \
                                &(self.data['Timestamp'] <= self.data['Timestamp'].iloc[i] - m + slack)
                if previous_index.any():
                    if method == 'mean':
                        previous_value = self.data.loc[previous_index, 'CGM'].mean()
                    elif method == 'closest':
                        closest_index = (self.data.loc[previous_index, 'Timestamp'] - self.data['Timestamp'].iloc[i]).abs().idxmin()
                        previous_value = self.data.loc[closest_index, 'CGM']
                    # calculate the difference between the current value and the previous value
                    differences.append(self.data['CGM'].iloc[i] - previous_value)

            conga = np.std(differences)

        return conga
                    

    # Glucose Variability Percentage (GVP)
    def gvp(self):
        '''
        Calculates the Glucose Variability Percentage (GVP), with time in minutes.

        Parameters
        ----------
        None

        Returns
        -------
        gvp : float
            Glucose Variability Percentage.
        '''
        # Calculate the difference between consecutive timestamps
        timeStamp_diff = pd.Series(np.diff(self.data['Timestamp']))
        # Calculate the difference between consecutive CGM values
        cgm_diff = pd.Series(np.diff(self.data['CGM']))

        line_length  = np.sum( np.sqrt( np.square(cgm_diff) \
                                      + np.square(timeStamp_diff.dt.total_seconds()/60) ) )
        
        t0 = pd.Timedelta(self.data['Timestamp'].tail(1).values[0] \
                         -self.data['Timestamp'].head(1).values[0]).total_seconds()/60
        
        gvp = (line_length/t0 - 1) *100
        return gvp
    
    # Mean Absolute Glucose Change per unit of time (MAG)
    def mag(self,
            per_day: bool = False,
            time_unit: str = 'm'):
        '''
        Calculates the Mean Absolute Glucose Change per unit of time (MAG).

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
        '''
        # Determine the factor to multiply the total seconds by
        if time_unit == 's':
            factor = 1
        elif time_unit == 'm':
            factor = 60
        elif time_unit == 'h':
            factor = 3600
        else:
            return "Error: Invalid time_unit. Must be 's', 'm', or 'h'."
        
        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')
            mag = pd.Series(dtype=float)
            for day, day_data in day_groups:
                # Calculate the difference between consecutive timestamps
                timeStamp_diff = pd.Series(np.diff(day_data['Timestamp']))
                # Calculate the difference between consecutive CGM values
                cgm_diff = pd.Series(np.abs(np.diff(day_data['CGM'])))
                # Calculate the MAG
                mag[day] = np.sum(np.abs(cgm_diff)) / (timeStamp_diff.dt.total_seconds().sum()/factor)
        
        else:
            # Calculate the difference between consecutive timestamps
            timeStamp_diff = pd.Series(np.diff(self.data['Timestamp']))
            # Calculate the difference between consecutive CGM values
            cgm_diff = pd.Series(np.abs(np.diff(self.data['CGM'])))
            # Calculate the MAG
            mag = np.sum(np.abs(cgm_diff)) / (timeStamp_diff.dt.total_seconds().sum()/factor)
            
        return mag


    # 6. Computational methods for the analysis of glycemic dynamics
        
    # Detrended fluctuation analysis (DFA)
    def dfa(self,
            per_day: bool = False):
        '''
        Calculates the Detrended Fluctuation Analysis (DFA).

        Parameters
        ----------
        per_day : bool, default False
            If True, returns the an array with the DFA for each day. If False, returns the DFA for all days combined.

        Returns
        -------
        dfa : float
            Detrended fluctuation analysis.
        '''
        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')

            dfa = pd.Series(dtype=float)
            for day, day_data in day_groups:
                # Convert the timestamp values to seconds since the start of the dataset
                x = (day_data['Timestamp'] - day_data['Timestamp'].min()).dt.total_seconds().values

                # Integrated data
                y = np.cumsum(day_data['CGM'].values - day_data['CGM'].mean())

                # Generate segment_sizes
                segment_sizes = np.logspace(start=1, stop=np.log2(x.size), num=int(np.log2(x.size))+1, base=2, dtype=int)

                rms_values = []
                for segment_size in segment_sizes:
                    # Divide y into segments
                    y_segments = np.array_split(y, x.size // segment_size)
                    x_segments = np.array_split(x, x.size // segment_size)

                    # Perform linear regression on each segment and calculate predicted values
                    y_predicted = [linregress(x_segment, y_segment).slope * x_segment + linregress(x_segment, y_segment).intercept \
                                   for x_segment, y_segment in zip(x_segments, y_segments)]
                    y_predicted = np.concatenate(y_predicted)

                    # Calculate the root mean square of the differences
                    rms = np.sqrt(np.mean(np.square(y - y_predicted)))
                    rms_values.append(rms)

                # Perform linear regression between log(segment_sizes) and rms_values
                dfa[day] = linregress(np.log(segment_sizes), np.log(rms_values)).slope

        else:
            # Convert the timestamp values to seconds since the start of the dataset
            x = (self.data['Timestamp'] - self.data['Timestamp'].min()).dt.total_seconds().values

            # Integrated data
            y = np.cumsum(self.data['CGM'].values - self.mean())

            # Generate segment_sizes
            # Generate segment_sizes
            segment_sizes = np.logspace(start=1, stop=np.log2(x.size), num=int(np.log2(x.size))+1, base=2, dtype=int)

            rms_values = []
            for segment_size in segment_sizes:
                # Divide y into segments
                y_segments = np.array_split(y, x.size // segment_size)
                x_segments = np.array_split(x, x.size // segment_size)

                # Perform linear regression on each segment and calculate predicted values
                y_predicted = [linregress(x_segment, y_segment).slope * x_segment + linregress(x_segment, y_segment).intercept \
                               for x_segment, y_segment in zip(x_segments, y_segments)]
                y_predicted = np.concatenate(y_predicted)

                # Calculate the root mean square of the differences
                rms = np.sqrt(np.mean(np.square(y - y_predicted)))
                rms_values.append(rms)

            # Perform linear regression between log(segment_sizes) and rms_values
            dfa = linregress(np.log(segment_sizes), np.log(rms_values)).slope

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

        Parameters
        ----------
        per_day : bool, default False
            If True, returns the an array with the Sample Entropy for each day. If False, returns the Sample Entropy for
            all days combined.
        delay : int, default 1
            Time Delay in samples (often denoted *Tau* :math:`\\tau`, sometimes referred to as *lag*). If None, the optimal 
            delay will be estimated using neurokit2.complexity_delay().
        dimension : int, default 2
            Embedding Dimension (*m*, sometimes referred to as *d* or *order*). If None, the optimal dimension will be
            estimated using neurokit2.complexity_dimension().
        tolerance : float, default None
            Tolerance (often denoted as *r*), distance to consider two data points as similar. If "sd" (default), will be
            set to 0.2 * std. If None, the optimal tolerance  will be estimated using neurokit2.complexity_tolerance().
        Returns
        -------
        samp_en : float | pandas.Series
            Entropy Sample.
        '''
        # Save original input
        original_delay = delay
        original_dimension = dimension
        original_tolerance = tolerance

        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')

            samp_en = pd.Series(dtype=float)
            for day, day_data in day_groups:
                # Get glucose values
                signal = day_data['CGM'].values

                # Estimate optimal parameters for sample entropy
                if delay is None:
                    delay, _  = nk.complexity_delay(signal)
                if dimension is None:
                    dimension, _ = nk.complexity_dimension(signal, delay=delay)
                if tolerance is None:
                    tolerance, _ = nk.complexity_tolerance(signal, delay=delay, dimension=dimension)

                # Calculate sample entropy
                day_samp_en, _ = nk.entropy_sample(signal, delay=delay, dimension=dimension, tolerance=tolerance)
                samp_en[day] = day_samp_en   

                # reset delay, dimension and tolerance
                delay = original_delay
                dimension = original_dimension
                tolerance = original_tolerance         

        else:
            # Get glucose values
            signal = self.data['CGM'].values

            # Estimate optimal parameters for sample entropy
            if delay is None:
                delay, _  = nk.complexity_delay(signal)
            if dimension is None:
                dimension, _ = nk.complexity_dimension(signal,delay=delay)
            if tolerance is None:
                tolerance, _ = nk.complexity_tolerance(signal, delay=delay, dimension=dimension)

            # Calculate sample entropy
            samp_en, _ = nk.entropy_sample(signal, delay=delay, dimension=dimension, tolerance=tolerance)
        
        return samp_en
        
    # Multiscale Sample Entropy (MSE)
    def mse(self,
            per_day: bool = False,
            scale = 'default',
            dimension = 3,
            tolerance = 'sd',
            method = 'MSEn',
            **kwargs):
        '''
        Calculates the Multiscale Sample Entropy using neurokit2.entropy_multiscale()

        Parameters
        ----------
        per_day : bool, default False
            If True, returns the an array with the Multiscale Sample Entropy for each day. If False, returns the 
            Multiscale Sample Entropy for all days combined.
        scale : str, int or list, default 'default'
            A list of scale factors used for coarse graining the time series. If ‘default’, will use 
            range(len(signal) / (dimension + 10)). If ‘max’, will use all scales until half the 
            length of the signal. If an integer, will create a range until the specified int. For more information
            view the documentation for
            `neurokit2.entropy_multiscale() <https://neuropsychology.github.io/NeuroKit/functions/complexity.html#entropy-multiscale>`_.
        dimension : int, default 3
            Embedding Dimension (*m*, sometimes referred to as *d* or *order*). If None, the optimal dimension will be
            estimated using neurokit2.complexity_dimension().
        tolerance : float, default None
            Tolerance (often denoted as *r*), distance to consider two data points as similar. If "sd" (default), will be
            set to 0.2 * std. If None, the optimal tolerance  will be estimated using neurokit2.complexity_tolerance().
        method : str, default 'MSEn'
            Method to use. For more information view the documentation for neurokit2.entropy_multiscale() 
            `neurokit2.entropy_multiscale() <https://neuropsychology.github.io/NeuroKit/functions/complexity.html#entropy-multiscale>`_.
        **kwargs : dict
            Additional keyword arguments to be passed to neurokit2.entropy_multiscale() 
            
        Returns
        -------
        mse : float | pandas.Series
            Multiscale Sample Entropy.
        '''
        # Save original input
        original_dimension = dimension
        original_tolerance = tolerance

        if per_day:
            # Group data by day
            day_groups = self.data.groupby('Day')

            mse = pd.Series(dtype=float)
            for day, day_data in day_groups:
                # Get glucose values
                signal = day_data['CGM'].values

                # Estimate optimal parameters for sample entropy
                if dimension is None:
                    dimension, _ = nk.complexity_dimension(signal)
                if tolerance is None:
                    tolerance, _ = nk.complexity_tolerance(signal, dimension=dimension)

                # Calculate sample entropy
                day_mse, _ = nk.entropy_multiscale(signal, 
                                                   scale=scale, 
                                                   dimension=dimension, 
                                                   tolerance=tolerance, 
                                                   method=method, 
                                                   **kwargs)
                mse[day] = day_mse   

                # reset dimension and tolerance
                dimension = original_dimension
                tolerance = original_tolerance

        else:
            # Get glucose values
            signal = self.data['CGM'].values

            # Estimate optimal parameters for sample entropy
            if dimension is None:
                dimension, _ = nk.complexity_dimension(signal)
            if tolerance is None:
                tolerance, _ = nk.complexity_tolerance(signal, dimension=dimension)

            # Calculate sample entropy
            mse, _ = nk.entropy_multiscale(signal, 
                                           scale=scale, 
                                           dimension=dimension, 
                                           tolerance=tolerance, 
                                           method=method, 
                                           **kwargs)        
            
        return mse




    

    
