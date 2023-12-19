#3rd party
import numpy as np
import pandas as pd

# Python
from collections.abc import Sequence

# Local
from glucopy.util.date_processing import disjoinDays

class Gframe:
    '''
    Description

    Parameters
    -----------
    data : pandas Dataframe 
        Dataframe containing the CGM signal information, it will be saved into a Dataframe with the columns ['Day','Time','CGM']
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
                self.data = disjoinDays(data, date_name, cgm_name)
            # if date_name is a list of 2 strings
            elif isinstance(date_name, Sequence) and len(date_name) == 2:
                self.data = pd.DataFrame(columns=['Day','Time','CGM'])
                self.data['Day'] = pd.to_datetime(data[date_name[0]]).dt.date
                self.data['Time'] = pd.to_datetime(data[date_name[1]]).dt.time
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
             time:str,
             ndays:int = 0) -> float:
        
        # convert time to same format as self.data['Time']
        try:
            time = pd.to_datetime(time).time()
        except:
            raise ValueError('time must be a valid time format')
        
        # get all the values with the time specified
        mask = self.data['Time'] == time
        values = self.data[mask]['CGM'].values

        # if ndays is 0, return the mean of the differences
        if ndays == 0:
            return np.mean(np.diff(values))
        # if ndays is not 0, return the mean of the differences of the first ndays
        else:
            return np.mean(np.abs(np.diff(values[:ndays])))


    
    # Mean of Daily differences
    
