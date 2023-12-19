#3rd party
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Signal:
    '''
    Description

    Parameters
    -----------
    data : Dataframe 
        It contais 2 Series, one for the time and one for the signal
    unit
    '''
    # Constructor
    def __init__(self, 
                 data=None, 
                 unit:str = 'mg/dL', 
                 copy:bool = True):
        
        self.unit = unit
        self.data = pd.DataFrame(columns=['Date','Glucose'])
        
        # Check data is a dataframe
        if isinstance(data, pd.DataFrame):
            if copy: # Make deep copy
                self.data["Date"] = data.iloc[:, 0]
                self.data["Glucose"] = data.iloc[:, 1]

            else:
                self.data = data.copy(deep=False)

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
             **kwargs):
        
        return self.data['Glucose'].mean(skipna=skipna,numeric_only=numeric_only,**kwargs)
    
    # Standard Deviation, by default ddof=1, so its divided by n-1
    def std(self,
            skipna:bool = True,
            numeric_only:bool = False,
            ddof:int = 1,
            **kwargs):
        
        return self.data['Glucose'].std(skipna=skipna,numeric_only=numeric_only,ddof=ddof,**kwargs)
    
    # Coefficient of Variation
    def cv(self,
           skipna:bool = True,
           numeric_only:bool = False,
           ddof:int = 1,
           **kwargs):
          
        return self.std(skipna=skipna,numeric_only=numeric_only,ddof=ddof,**kwargs)/self.mean(skipna=skipna,numeric_only=numeric_only,**kwargs)
            
    # % Coefficient of Variation
    def pcv(self,
            skipna:bool = True,
            numeric_only:bool = False,
            ddof:int = 1,
            **kwargs):
          
        return self.cv(skipna=skipna,numeric_only=numeric_only,ddof=ddof,**kwargs)*100
    
    # Quantiles
    def quantile(self,
                 q:float = 0.5,
                 numeric_only:bool = False,
                 interpolation:str = 'linear',
                 **kwargs):
        
        return self.data['Glucose'].quantile(q=q,numeric_only=numeric_only, interpolation=interpolation, **kwargs)
    
    # Interquartile Range
    def iqr(self,
            numeric_only:bool = False,
            interpolation:str = 'linear',
            **kwargs):
        
        q1 = self.quantile(q=0.25,numeric_only=numeric_only, interpolation=interpolation, **kwargs)
        q3 = self.quantile(q=0.75,numeric_only=numeric_only, interpolation=interpolation, **kwargs)
        
        return q3 - q1
    
    # Mean of Daily differences
    
