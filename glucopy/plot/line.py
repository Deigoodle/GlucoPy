# 3rd party
import plotly.express as px
import plotly.graph_objects as go

# Python
from datetime import datetime

# Local
from .. import Gframe

def line(gf: Gframe,
            separate_days: bool = True,
            height: float = None,
            width: float = None,
            ):
    '''
    Plots a line plot of the CGM values in the Gframe object

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    separate_days : bool, optional
        If True, the plot will be separated by days, by default False
    height : float, optional
        Height of the plot, by default None
    width : float, optional
        Width of the plot, by default None
    points : str, optional
        Show points in the plot, can be 'all', 'outliers', 'suspectedoutliers', False, by default None
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure object
    '''
    # Disclaimer
    # This is a bit of a hack, but i did it because some datasets have different time values for each day and that
    # broke the line plot. This is a temporary fix until I can figure out a better way to do this.

    if separate_days:
        day_groups = gf.data.groupby('Day')   
        fig = go.Figure()
        for day, day_data in day_groups:
            # Convert time to seconds past midnight
            day_data['Time'] = day_data['Time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
            fig.add_trace(go.Scatter(x=day_data['Time'], y=day_data['CGM'], name=str(day)))

    else:
        fig = px.line(gf.data, x='Timestamp', y='CGM')

    # Convert the x-axis labels back to time format
    fig.update_xaxes(tickvals=list(range(0, 24*3600, 1*3600)), ticktext=[f'{h}:00' for h in range(0, 24, 1)])
    fig.update_layout(xaxis_title='Time of day', yaxis_title='CGM (mg/dL)')

    return fig