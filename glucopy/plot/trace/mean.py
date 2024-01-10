# 3rd party
import plotly.graph_objects as go
import numpy as np

# Local
from ...classes import Gframe

def mean(gf: Gframe,
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
    # Check input
    if not isinstance(gf, Gframe):
        raise TypeError('gf must be a Gframe object')
    
    # Group the data by time
    time_groups = gf.data.groupby('Time')

    # Get the mean and std
    mean_list = time_groups['CGM'].mean()
    std_list = time_groups['CGM'].std()

    fig = go.Figure()

    # add traces
    fig.add_trace(go.Scatter(x=mean_list.index, 
                             y=mean_list, 
                             name='Mean'
                            )
                 )
    fig.add_trace(go.Scatter(x=mean_list.index, 
                             y=mean_list + std_list, 
                             name='Std',
                             fill=None, 
                             mode='lines', 
                             line_color='lightgray',
                             legendgroup='std',
                            )
                 )
    fig.add_trace(go.Scatter(x=mean_list.index, 
                             y=mean_list - std_list, 
                             name='-Std',
                             fill='tonexty', 
                             mode='lines', 
                             line_color='lightgray',
                             legendgroup='std',
                             showlegend=False
                            )
                 )
    
    # add extra traces
    if(add_all_mean):
        all_mean = np.array([gf.data['CGM'].mean()] * len(mean_list))
        fig.add_trace(go.Scatter(x=mean_list.index, 
                                 y=all_mean, 
                                 name='All Mean', 
                                 line_color='red', 
                                 line=dict(dash='dot')
                                )
                     )
    
    if(add_all_std):
        all_std = np.array([gf.data['CGM'].std()] * len(mean_list))
        fig.add_trace(go.Scatter(x=mean_list.index, 
                                 y=all_mean+all_std, 
                                 name='All Std', 
                                 line_color='blue', 
                                 line=dict(dash='dot'),
                                 legendgroup='all std'
                                )
                     )
        fig.add_trace(go.Scatter(x=mean_list.index, 
                                 y=all_mean-all_std, 
                                 name='-All Std', 
                                 line_color='blue', 
                                 line=dict(dash='dot'),
                                 legendgroup='all std',
                                 showlegend=False
                                )
                      )
    
    if(add_std_peak):
        std_max = [max(mean_list+std_list)] * len(mean_list)
        std_min = [min(mean_list-std_list)] * len(mean_list)
        fig.add_trace(go.Scatter(x=mean_list.index, 
                                 y=std_max, 
                                 name='Std peak', 
                                 line_color='green', 
                                 line=dict(dash='dot'),
                                 legendgroup='std peak'
                                )
                     )
        fig.add_trace(go.Scatter(x=mean_list.index, 
                                 y=std_min, 
                                 name='Std nadir', 
                                 line_color='green', 
                                 line=dict(dash='dot'),
                                 legendgroup='std peak',
                                 showlegend=False
                                )
                     )
    
    if(add_quartiles):
        q1 = time_groups['CGM'].quantile(0.25)
        q3 = time_groups['CGM'].quantile(0.75)
        fig.add_trace(go.Scatter(x=mean_list.index,
                                 y=q1, 
                                 name='25% - 75%', 
                                 line_color='orange', 
                                 line=dict(dash='dot'),
                                 legendgroup='quartiles'
                                )
                     )
        fig.add_trace(go.Scatter(x=mean_list.index, 
                                 y=q3, 
                                 name='75%', 
                                 line_color='orange', 
                                 line=dict(dash='dot'),
                                 legendgroup='quartiles',
                                 showlegend=False
                                 )
                     )

    # update lauyout
    fig.update_layout(title_text='Mean and standard deviation for each time',
                      xaxis_title='Time of day [h]', 
                      yaxis_title='CGM (mg/dL)',
                      height=height,
                      width=width)

    return fig