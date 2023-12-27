# 3rd party
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Built-in
from datetime import time

# Local
from .. import Gframe

def trace(gf: Gframe,
          separate_days: bool = True,
          show_interval: bool = True,
          interval: list[int] = [70, 140],
          height: float = None,
          width: float = None,
    ):
    '''
    Plots a line plot of the CGM values for each day in the Gframe object

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    separate_days : bool, optional
        If True, the plot will be separated by days, by default False
    show_interval : bool, optional
        If True, the values in the range will be highlighted, by default True
    interval : list[int], optional
        interval to highlight, by default [70, 140]
    height : float, optional
        Height of the plot, by default None
    width : float, optional
        Width of the plot, by default None
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure object
    '''
    # This is a bit of a hack, but i did it because some datasets have different time values for each day and that
    # broke the line plot. This is a temporary fix until I can figure out a better way to do this.
    fig = go.Figure()
    
    if separate_days:
        day_groups = gf.data.groupby('Day')   
        for day, day_data in day_groups:
            # Convert time to seconds past midnight
            seconds = day_data['Time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
            fig.add_trace(go.Scatter(x=seconds, y=day_data['CGM'], name=str(day),text=day_data['Time']))

    else:
        fig.add_trace(go.Scatter(x=gf.data['Timestamp'], y=gf.data['CGM'], name='CGM'))

    # Convert the x-axis labels back to time format
    fig.update_xaxes(tickvals=list(range(0, 24*3600, 1*3600)), ticktext=[f'{h}' for h in range(0, 24, 1)])
    fig.update_layout(xaxis_title='Time of day [h]', yaxis_title='CGM (mg/dL)')
    fig.update_layout(height=height, width=width)

    # Add shape for the range 70-140
    if show_interval:
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=0,
            y0=interval[0],
            x1=max(gf.data["Time"].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)),
            y1=interval[1],
            fillcolor="LightSkyBlue",
            opacity=0.5,
            layer="below",
            line_width=0,
        )

    return fig

def mean_trace(gf: Gframe,
            add_all_mean: bool = True,
            add_all_std: bool = True,
            add_std_peak: bool = True,
            add_quartiles: bool = False,
            height: float = None,
            width: float = None):
    '''
    Plots a line plot of the CGM values in the Gframe object

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    add_all_mean : bool, optional
        If True, the mean of all the data will be added to the plot, by default True
    add_all_std : bool, optional
        If True, the standard deviation of all the data will be added to the plot, by default True
    add_std_peak : bool, optional
        If True, the peaks of standard deviation will be added to the plot, by default True
    add_quartiles : bool, optional
        If True, the quartiles of the data will be added to the plot, by default False
    height : float, optional
        Height of the plot, by default None
    width : float, optional
        Width of the plot, by default None
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure object
    '''
    # Group the data by time
    time_groups = gf.data.groupby('Time')

    # Get the mean and std
    mean_list = time_groups['CGM'].mean()
    std_list = time_groups['CGM'].std()

    fig = go.Figure()

    # add traces
    fig.add_trace(go.Scatter(x=mean_list.index, y=mean_list, name='Mean CGM'))
    fig.add_trace(go.Scatter(x=mean_list.index, y=mean_list + std_list, fill=None, mode='lines', line_color='lightgray', name='+Std'))
    fig.add_trace(go.Scatter(x=mean_list.index, y=mean_list - std_list, fill='tonexty', mode='lines', line_color='lightgray', name='-Std'))
    
    # add extra traces
    if(add_all_mean):
        all_mean = np.array([gf.data['CGM'].mean()] * len(mean_list))
        fig.add_trace(go.Scatter(x=mean_list.index, y=all_mean, name='All mean', line_color='red', line=dict(dash='dot')))
    
    if(add_all_std):
        all_std = np.array([gf.data['CGM'].std()] * len(mean_list))
        fig.add_trace(go.Scatter(x=mean_list.index, y=all_mean+all_std, name='All mean+std', line_color='blue', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=mean_list.index, y=all_mean-all_std, name='All mean-std', line_color='blue', line=dict(dash='dot')))
    
    if(add_std_peak):
        std_max = [max(mean_list+std_list)] * len(mean_list)
        std_min = [min(mean_list-std_list)] * len(mean_list)
        fig.add_trace(go.Scatter(x=mean_list.index, y=std_max, name='Max std', line_color='green', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=mean_list.index, y=std_min, name='Min std', line_color='green', line=dict(dash='dot')))
    
    if(add_quartiles):
        q1 = time_groups['CGM'].quantile(0.25)
        q3 = time_groups['CGM'].quantile(0.75)
        fig.add_trace(go.Scatter(x=mean_list.index, y=q1, name='25%', line_color='orange', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=mean_list.index, y=q3, name='75%', line_color='orange', line=dict(dash='dot')))

    # update lauyout
    fig.update_layout(title_text='Mean and standard deviation for each time',
                      xaxis_title='Time of day [h]', 
                      yaxis_title='CGM (mg/dL)',
                      height=height,
                      width=width)

    return fig
        
def tir_trace(gf: Gframe,
              interval: list[int] = [70, 140],
              height: float = None,
              width: float = None
    ):
    '''
    Plots a line plot of the CGM values in the Gframe object separated by time in range for each day

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    interval : list[int], optional
        interval to highlight, by default [70, 140]
    height : float, optional
        Height of the plot, by default None
    width : float, optional
        Width of the plot, by default None

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure object
    '''
    # Group the data by day
    day_groups = gf.data.groupby('Day')

    # Create figure
    fig = go.Figure()

    # Add traces 
    show_first = True
    for day, day_data in day_groups:
        fig.add_trace(go.Scatter(x=day_data['Time'], y=day_data['CGM'], name=str(day), visible=show_first))

        # Add background color for the interval
        fig.add_shape(
            type="rect",
            x0=day_data['Time'].iloc[0],
            y0=interval[0],
            x1=day_data['Time'].iloc[-1],
            y1=interval[1],
            fillcolor="LightSkyBlue",
            opacity=0.5,
            layer="below",
            line_width=0,
        )
        if show_first:
            show_first = False
            
    # update layout
    fig.update_layout(
        xaxis_title='Time of day [h]',
        yaxis_title='CGM (mg/dL)',
        yaxis=dict(range=[0, gf.data['CGM'].max()+10]),  # Set y-axis to start at 0 and end at the max value
        height=height,
        width=width,
        title_text='CGM values for each day'
    )

    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[
                            {"visible": [True if i == j else False for j in range(len(day_groups))]},
                        ],
                        label=str(day),
                        method="update"
                    ) for i, day in enumerate(day_groups.groups.keys())
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1,
                xanchor="right",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    return fig

def mage_trace(gf: Gframe,
               height: float = None,
               width: float = None):
    '''
    Plots a line plot of the CGM values showing the MAGE for each day

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    height : float, optional
        Height of the plot, by default None
    width : float, optional
        Width of the plot, by default None
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure object
    '''
    # Group the data by day
    day_groups = gf.data.groupby('Day')

    # Create figure
    fig = go.Figure()

    # Add traces
    show_first = True
    for day, day_data in day_groups:
        # CGM
        fig.add_trace(go.Scatter(x=day_data['Time'], y=day_data['CGM'], name=str(day), visible=show_first))
        # mean and std
        mean = day_data['CGM'].mean()
        std = day_data['CGM'].std()
        fig.add_trace(go.Scatter(x=day_data['Time'], y=[mean] * len(day_data), name='Mean',line=dict(dash='dot'), line_color='red',visible=show_first))
        fig.add_trace(go.Scatter(x=day_data['Time'], y=[mean + std] * len(day_data), name='Mean + std',line=dict(dash='dot'),line_color='cyan', visible=show_first))
        fig.add_trace(go.Scatter(x=day_data['Time'], y=[mean - std] * len(day_data), name='Mean - std',line=dict(dash='dot'),line_color='cyan', visible=show_first))
        if show_first:
            show_first = False

    # update layout
    fig.update_layout(
        xaxis_title='Time of day [h]',
        yaxis_title='CGM (mg/dL)',
        yaxis=dict(range=[0, gf.data['CGM'].max()+10]),  # Set y-axis to start at 0 and end at the max value
        height=height,
        width=width,
        title_text='CGM values for each day'
    )

    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[
                            {"visible": [True if i//4 == j else False for i in range(len(day_groups)*4)]},
                        ],
                        label=str(day),
                        method="update"
                    ) for j, day in enumerate(day_groups.groups.keys())
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1,
                xanchor="right",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    return fig


        
