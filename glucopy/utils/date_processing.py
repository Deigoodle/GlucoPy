#3rd party
import pandas as pd

def disjoin_days_and_hours(df,
                           date_name = None, 
                           cgm_name = None) -> pd.DataFrame:
    
    if date_name is None:
        date_name = df.columns[0]
    if cgm_name is None:
        cgm_name = df.columns[1]

    disjoined_df = pd.DataFrame(columns=['Timestamp','Day','Time','CGM'])

    disjoined_df['Timestamp'] = df[date_name]
    disjoined_df['Day'] = df[date_name].dt.date
    disjoined_df['Time'] = df[date_name].dt.time
    disjoined_df['CGM'] = df[cgm_name]

    return disjoined_df