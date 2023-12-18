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
    def mean(self, 
             skipna:bool = True,
             numeric_only:bool = True,
             **kwargs):
        
        return self.data['Glucose'].mean(skipna=skipna,numeric_only=numeric_only,kwargs=kwargs)
