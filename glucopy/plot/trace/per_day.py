# 3rd party
import plotly.graph_objects as go

# Local
from ...classes import Gframe

def per_day(gf: Gframe,
            num_days: int = None,
            height: float = None,
            width: float = None,
    ):
    '''
    Plots a line plot of the CGM values for each day in the Gframe object

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    num_days : int, optional
        Number of days to plot, by default None
    height : float, optional
        Height of the plot, by default None
    width : float, optional
        Width of the plot, by default None
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure object
    '''
    # Check input
    if not isinstance(gf, Gframe):
        raise TypeError('gf must be a Gframe object')
    
    fig = go.Figure()
    
    day_groups = gf.data.groupby('Day')
    
    trace_count = 0
    for day, day_data in day_groups:
        # Convert time to seconds past midnight
        seconds = day_data['Time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

        if trace_count == None or trace_count < num_days:
            visibility = True
        else:
            visibility = 'legendonly'
        
        fig.add_trace(go.Scatter(x=seconds, 
                                 y=day_data['CGM'], 
                                 name=str(day),
                                 text=day_data['Time'],
                                 mode='lines',
                                 visible = visibility))
        
        trace_count += 1

    # Convert the x-axis labels back to time format
    fig.update_xaxes(tickvals=list(range(0, 24*3600, 1*3600)), ticktext=[f'{h}' for h in range(0, 24, 1)])

    fig.update_layout(xaxis_title='Time of day [h]', 
                      yaxis_title='CGM (mg/dL)',
                      height=height,
                      width=width)

    return fig
