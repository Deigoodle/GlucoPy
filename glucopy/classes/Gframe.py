#3rd party
import numpy as np
import pandas as pd

# Built-in
from collections.abc import Sequence
import datetime
from typing import List
from itertools import islice

# Local
from glucopy.utils import disjoin_days_and_hours
from glucopy.utils import str_to_time

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
        
        self.unit = unit
        self.n_samples = self.data.shape[0]
        self.n_days = len(self.data['Day'].unique())


    # String representation
    def __str__(self):
        return str(self.data)
    
    # Metrics 
    # -------
    # 1. Joint data analysis metrics for glycaemia dynamics

    # Sample Mean
    def mean(self, 
             skipna:bool = True,
             numeric_only:bool = False,
             **kwargs) -> float:
        
        return self.data['CGM'].mean(skipna=skipna,numeric_only=numeric_only,**kwargs)
    
    # Standard Deviation, by default ddof=1, so its divided by n-1
    def std(self,
            skipna:bool = True,
            numeric_only:bool = False,
            ddof:int = 1,
            **kwargs) -> float:
        
        return self.data['CGM'].std(skipna=skipna,numeric_only=numeric_only,ddof=ddof,**kwargs)
    
    # Coefficient of Variation
    def cv(self,
           skipna:bool = True,
           numeric_only:bool = False,
           ddof:int = 1,
           **kwargs) -> float:
          
        return self.std(skipna=skipna,numeric_only=numeric_only,ddof=ddof,**kwargs)/self.mean(skipna=skipna,numeric_only=numeric_only,**kwargs)
            
    # % Coefficient of Variation
    def pcv(self,
            skipna:bool = True,
            numeric_only:bool = False,
            ddof:int = 1,
            **kwargs) -> float:
          
        return self.cv(skipna=skipna,numeric_only=numeric_only,ddof=ddof,**kwargs)*100
    
    # Quantiles
    def quantile(self,
                 q:float = 0.5,
                 interpolation:str = 'linear',
                 **kwargs) -> float:
        
        return self.data['CGM'].quantile(q=q, interpolation=interpolation, **kwargs)
    
    # Interquartile Range
    def iqr(self,
            interpolation:str = 'linear',
            **kwargs) -> float:
        
        q1 = self.quantile(q=0.25, interpolation=interpolation, **kwargs)
        q3 = self.quantile(q=0.75, interpolation=interpolation, **kwargs)
        
        return q3 - q1
    
    # Mean of Daily Differences
    def modd(self, 
             target_time: str | datetime.time, 
             error_range:int = 0, 
             ndays: int = 0) -> float:
        '''
        Calculates the Mean of Daily Differences (MODD) for a given time of day.

        Parameters
        ----------
        target_time : str | datetime.time
            Time of day to calculate the MODD for. If a string is given, it must be in the format 'HH:MM:SS','HH:MM'
            or 'HH'.
        error_range : int, default 0
            Maximum number of minutes that the given time can differ from the actual time in the data.
        ndays : int, default 0
            Number of days to use for the calculation. If 0, all days will be used. There will be used the first ndays

        Returns
        -------
        modd : float
        '''
        
        # Check input
        if error_range < 0:
            raise ValueError('error_range must be a positive number or 0')
    
        # Check input
        if ndays < 0 or ndays == 1:
            raise ValueError('ndays must be a greater than 1 or 0')
        
        if ndays == 0: # if ndays is 0, use all days
            ndays = self.n_days
    
        # convert time to same format as self.data['Time']
        if not isinstance(target_time, str) and not isinstance(target_time, datetime.time):
            raise TypeError('time must be a string or a datetime.time')
        elif isinstance(target_time, str):# String -> datetime.time
            target_str = target_time
            target_time = str_to_time(target_str)
        else: # datetime.time
            target_str = target_time.strftime('%H:%M:%S')
        
        # convert error_range to timedelta
        error_range = pd.to_timedelta(error_range, unit='m')
    
        cgm_values: List[float] = []

        # search given time in each day
        for day, day_data in islice(self.data.groupby('Day'), ndays):
            mask_exact = day_data['Time'] == target_time
            # if exact time is found, use it
            if mask_exact.any():
                cgm_values.append(day_data.loc[mask_exact, 'CGM'].values[0])
            # if not, search for closest time within error range
            elif error_range > pd.Timedelta('0 min'):
                # combine "day" and target_time to compare it with Timestamp
                target_date = str(day) + ' ' + target_str
                target_datetime = pd.to_datetime(target_date)
                # search for closest time within error range
                mask_range = ((day_data['Timestamp'] - target_datetime).abs() <= error_range)
                if mask_range.any():
                    closest_index = (day_data.loc[mask_range, 'Timestamp'] - target_datetime).abs().idxmin()
                    cgm_values.append(day_data.loc[closest_index, 'CGM'])
                else:
                    raise ValueError(f"No data found for date {day}")
    
        return np.sum(np.abs(np.diff(cgm_values))) / (ndays)

    # Time in Range
    def tir(self, 
            target_range:list= [70,140]):
        '''
        Calculates the Time in Range (TIR) for a given target range (CGM) for each day.

        Parameters
        ----------
        target_range : list of int|float, default [0,70,140,350]
            Target range in CGM unit for low, normal and high glycaemia.

        Returns
        -------
        tir : list 
            List of TIR for each day, in format [[low, normal, high], ...].
        '''
        # Check input, Ensure target_range is a list with 0 and the max value of the data
        if not isinstance(target_range, list) or not all(isinstance(i, (int, float)) for i in target_range):
            raise ValueError("target_range must be a list of numbers")
        if 0 not in target_range:
            target_range = [0] + target_range
        if max(self.data['CGM']) > target_range[-1]:
            target_range = target_range + [max(self.data['CGM'])]

        # Group data by day
        day_groups = self.data.groupby('Day')

        tir = []

        # Calculate TIR for each day
        for _, day_df in day_groups:
            day_df['ranges'] = pd.cut(day_df['CGM'], bins=target_range)
            result = (day_df.groupby([pd.Grouper(key='Timestamp',freq='D'),'ranges'],observed=False)['ranges'].count().unstack(0).T)
            summed_results = result.sum()
            tir.append(np.array(summed_results/summed_results.sum()*100))

        return tir

    
    # 2. Analysis of distribution in the plane for glycaemia dynamics.

    # Frecuency distribution : counts the amount of observations given certain intervals of CGM
    def fd(self,
           intervals: list = [0,70,140,350]):
        '''
        Calculates the frequency distribution for a given intervals of CGM for each day.

        Parameters
        ----------
        intervals : list of int|float, default [0,70,140,350]
            Intervals of Glucose concentration.
        
        Returns
        -------
        fd : list 
            List of frequency distribution for each day.
        '''
        # Check input
        if not isinstance(intervals, list) or not all(isinstance(i, (int, float)) for i in intervals):
            raise ValueError("intervals must be a list of numbers")
        
        # Group data by day
        day_groups = self.data.groupby('Day')

        fd = []

        # Calculate fd for each day
        for _, day_df in day_groups:
            day_df['intervals'] = pd.cut(day_df['CGM'], bins=intervals)
            result = (day_df.groupby([pd.Grouper(key='Timestamp',freq='D'),'intervals'],observed=False)['intervals'].count().unstack(0).T)
            summed_results = result.sum()
            fd.append(np.array(summed_results/summed_results.sum()))

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

        auc = []

        # Calculate AUC for each day
        for _, day_df in day_groups:
            # Convert timestamps to the specified time unit
            time_values = (day_df['Timestamp'] - day_df['Timestamp'].min()).dt.total_seconds() / factor
            auc.append(np.trapz(y=day_df['CGM'], x=time_values))

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

        mage = []
        lambd = max(self.data['CGM']) - self.mean()
        std = self.std()
        mean = self.mean()

        # Calculate MAGE for each day
        
        for _, day_df in day_groups:
            count = 0
            for cgm_value in day_df['CGM']:
                if (abs(cgm_value - mean) >  std):
                    count += 1

        return mage

    # Distance Travelled (DT)
    def dt(self):
        '''
        Calculates the Distance Travelled (DT) for each day.

        Parameters
        ----------
        None

        Returns
        -------
        dt : list 
            List of DT for each day.
        '''
        # Group data by day
        day_groups = self.data.groupby('Day')

        dt = []

        # Calculate DT for each day
        for _, day_data in day_groups:
            dt.append(np.sum(np.abs(np.diff(day_data['CGM']))))

        return dt
    
    # 4. Metrics for the analysis of glycaemic dynamics using scores of glucose values

    # Low Blood Glucose Index (LBGI) and High Blood Glucose Index (HBGI)
    def bgi(self,
            index_type:str = 'h',
            maximum: bool = False):
        '''
        Calculates the Low Blood Glucose Index (LBGI) for each day. Only works for CGM values in mg/dL.

        Parameters
        ----------
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

        # Group data by day
        day_groups = self.data.groupby('Day')

        lbgi = []
        for _, day_data in day_groups:
            values = day_data['CGM'].values
            f_values = np.vectorize(f)(values,index_type)
            risk = 22.77 * np.square(f_values)
            if maximum:
                lbgi.append(np.max(risk))
            else:
                lbgi.append(np.mean(risk))
        return lbgi
        
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
        # Group data by day
        day_groups = self.data.groupby('Day')   

        adrr = []
        for _, day_data in day_groups:
            adrr.append(self.bgi(index_type='h',maximum=True) + self.bgi(index_type='l',maximum=True))
        
        return np.mean(adrr)

    # Glycaemic Risk Assessment Diabetes Equation (GRADE)
    def grade(self,
              daily:bool = False):
        '''
        Calculates the Glycaemic Risk Assessment Diabetes Equation (GRADE) for each day. Only works for CGM values in mg/dL.

        Parameters
        ----------
        None

        Returns
        -------
        grade : list
            List of GRADE for each day.
        '''
        if daily:
            # Group data by day
            day_groups = self.data.groupby('Day')   

            grade = []
            for _, day_data in day_groups:
                values = day_data['CGM'].values
                grade.append(425 * np.square( np.log10( np.log10(values*18) ) + 0.16))

        else:
            values = self.data['CGM'].values
            grade = 425 * np.square( np.log10( np.log10(values*18) ) + 0.16)
        
        return grade

    # Q-Score Glucose=180.15588[g/mol] | 1 [mg/dL] -> 0.05551 [mmol/L] | 1 [mmol/L] -> 18.0182 [mg/dL]
    def qscore(self):
        pass

    # 5. Metrics for the analysis of glycaemic dynamics using variability estimation.

    # Continuous Overall Net Glycaemic Action (CONGA)
    '''
    def conga(self,
              separate_days: bool = True):
        
        Calculates the Continuous Overall Net Glycaemic Action (CONGA) for each day.

        Parameters
        ----------
        separate_days : bool, default True
            If True, returns the CONGA for each day separately. If False, returns the CONGA for all days combined.

        Returns
        -------
        conga : list
            List of CONGA for each day.
        
        conga = []
        if separate_days:
            # Group data by day
            day_groups = self.data.groupby('Day')
            for _, day_data in day_groups:
                # count number of times a measurement has a previous measurement 60 min ago
                k = 0
                for i in range(1, day_data.shape[0]):
                    if (day_data.iloc[i]['Timestamp'] - day_data.iloc[i-1]['Timestamp']) == pd.Timedelta('60 min'):
                        k += 1
    '''

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
            unit: str = 'm',
            per_day: bool = False):
        '''
        Calculates the Mean Absolute Glucose Change per unit of time (MAG).

        Parameters
        ----------
        unit : str, default 'm' (minutes)
            The time unit for the x-axis. Can be 's (seconds)', 'm (minutes)', or 'h (hours)'.
        per_day : bool, default False
            If True, returns the an array with the MAG for each day. If False, returns the MAG for all days combined.
        
        Returns
        -------
        mag : float
            Mean Absolute Glucose Change per unit of time.
        '''
        # Determine the factor to multiply the total seconds by
        if unit == 's':
            factor = 1
        elif unit == 'm':
            factor = 60
        elif unit == 'h':
            factor = 3600
        else:
            return "Error: Invalid time unit. Must be 's', 'm', or 'h'."
        
        if(per_day):
            # Group data by day
            day_groups = self.data.groupby('Day')
            mag = []
            for _, day_data in day_groups:
                # Calculate the difference between consecutive timestamps
                timeStamp_diff = pd.Series(np.diff(day_data['Timestamp']))
                # Calculate the difference between consecutive CGM values
                cgm_diff = pd.Series(np.abs(np.diff(day_data['CGM'])))
                # Calculate the MAG
                mag.append(np.sum(np.abs(cgm_diff)) / (timeStamp_diff.dt.total_seconds().sum()/factor))
            return mag
        else:
            # Calculate the difference between consecutive timestamps
            timeStamp_diff = pd.Series(np.diff(self.data['Timestamp']))
            # Calculate the difference between consecutive CGM values
            cgm_diff = pd.Series(np.abs(np.diff(self.data['CGM'])))
            # Calculate the MAG
            mag = np.sum(np.abs(cgm_diff)) / (timeStamp_diff.dt.total_seconds().sum()/factor)
            return mag






    

    
