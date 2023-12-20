#3rd party
import numpy as np
import pandas as pd

# Python
from collections.abc import Sequence
import datetime
from typing import List
from itertools import islice

# Local
from glucopy.utils.date_processing import disjoin_days_and_hours

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
    date_name : String or String array, default None
        The name or names of the column(s) containing the date information
        If it's a String, it will be the name of the single column containing the date information
        If it's a String array, it will be the 2 names of the columns containing the date information, eg. ['Date','Time']
        If it's None, it will be assumed that the date information is in the first column
    cgm_name : String, default None
        The name of the column containing the CGM signal information
        If it's None, it will be assumed that the CGM signal information is in the second column
    '''

    # Constructor
    def __init__(self, 
                 data=None, 
                 unit:str = 'mg/dL',
                 date_name = None,
                 cgm_name:str = None):
        
        self.unit = unit
        
        # Check data is a dataframe
        if isinstance(data, pd.DataFrame):
            # Check date_name
            if isinstance(date_name, str) or date_name is None:
                self.data = disjoin_days_and_hours(data, date_name, cgm_name)

            # if date_name is a list of 2 strings
            elif isinstance(date_name, Sequence) and len(date_name) == 2:
                self.data = pd.DataFrame(columns=['Timestamp','Day','Time','CGM'])
                combined_timestamp = pd.to_datetime(data[date_name[0]].astype(str) + ' ' + data[date_name[1]].astype(str))

                self.data['Timestamp'] = combined_timestamp
                self.data['Day'] = combined_timestamp.dt.date
                self.data['Time'] = combined_timestamp.dt.time
                self.data['CGM'] = data[cgm_name]

            else:
                raise ValueError('date_name must be a String or a sequence of 2 Strings')


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
             target_time: str | pd.Timestamp | datetime.time, 
             error_range, 
             ndays: int = 0) -> float:
        
        if error_range < 0:
            raise ValueError('error_range must be a positive number or 0')
    
        if ndays < 0 or ndays == 1:
            raise ValueError('ndays must be a greater than 1 or 0')
        
        if ndays == 0: # if ndays is 0, use all days
            ndays = len(self.data['Day'].unique())
    
        # convert time to same format as self.data['Time']
        if isinstance(target_time, str):# String -> datetime.time
            try:
                if target_time.count(':') == 0:  # Format is 'HH'
                    target_str = target_time + ':00:00'
                elif target_time.count(':') == 1:  # Format is 'HH:MM'
                    target_str = target_time + ':00'
                else:  # Format is 'HH:MM:SS'
                    target_str = target_time
                target_time = pd.to_datetime(target_str).time()
            except:
                raise ValueError('time must be a valid time format')
            
        elif isinstance(target_time, pd.Timestamp): # pandas Timestamp -> datetime.time
            target_str = target_time.strftime('%H:%M:%S')
            target_time = target_time.time()

        elif isinstance(target_time, datetime.time): # datetime.time
            target_str = target_time.strftime('%H:%M:%S')

        else:
            raise TypeError('time must be a string, a pandas Timestamp, or a datetime.time')
        
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
            else:
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

    def tir(self,
        total_time:int = 24,
        target_range:tuple = (70,140)):
    
        if total_time <= 0 or total_time > 24:
            raise ValueError('total_time must be an integer between 1 and 24 (inclusive)')
    
        # convert target_range to a list of floats
        try:
            target_range = [float(num) for num in target_range]
        except TypeError:
            raise ValueError("target_range must be an iterable of two numbers")
    
        if len(target_range) != 2:
            raise ValueError("target_range must contain exactly two numbers")
    
        if target_range[0] >= target_range[1]:
            target_range = sorted(target_range)
    
        # for each day calculate the time that the CGM signal was within the target range divided by total_time
        tir = []
        for _, day_data in self.data.groupby('Day'):
            day_tir = pd.Timedelta(0)
            prev_timestamp = None
            for timestamp, cgm in zip(day_data['Timestamp'], day_data['CGM']):
                if target_range[0] <= cgm <= target_range[1]:
                    if prev_timestamp is not None:
                        day_tir += timestamp - prev_timestamp
                    prev_timestamp = timestamp
            tir.append(day_tir.total_seconds() * 100 / (total_time * 3600))
    
        return tir

    
    # Mean of Daily differences
    
