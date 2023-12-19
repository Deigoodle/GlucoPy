#3rd party
import pandas as pd

def disjoinDays(df, date_name = None, cgm_name = None):
    if date_name is None:
        date_name = df.columns[0]
    if cgm_name is None:
        cgm_name = df.columns[1]

    disjoined_df = pd.DataFrame(columns=['Day','Time','CGM'])
    disjoined_df['Day'] = df[date_name].dt.date
    disjoined_df['Time'] = df[date_name].dt.time
    disjoined_df['CGM'] = df[cgm_name]
    return disjoined_df