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
    Plots a line plot of the mean and standard deviation of fixed intervals of 15 minutes

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    add_all_mean : bool, default True
        If True, the mean of all the data will be added to the plot
    add_all_std : bool, default True
        If True, the standard deviation of all the data will be added to the plot
    add_std_peak : bool, default True
        If True, the peaks of standard deviation will be added to the plot
    add_quartiles : bool, default False
        If True, the quartiles of the data will be added to the plot
    height : float, default None
        Height of the plot
    width : float, default None
        Width of the plot
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure object

    Examples
    --------
    Plot the mean and standard deviation of each interval of 15 minutes (default)

    .. ipython:: python

        import glucopy as gp
        gf = gp.data('prueba_1')
        gp.plot.mean(gf)

    .. image:: /../img/mean_plot_1.png
        :alt: Mean and std plot
        :align: center
    .. raw:: html

        <br>

    It is posible to add the quartiles of the intervals

    .. ipython:: python

        gp.plot.mean(gf, add_quartiles=True)

    .. image:: /../img/mean_plot_2.png
        :alt: Mean and std plot with quartiles
        :align: center
    '''
    # Check input
    if not isinstance(gf, Gframe):
        raise TypeError('gf must be a Gframe object')
    
    # Create a copy of the data
    data_copy = gf.data.copy()

    # Extract the time from the 'Timestamp' column and floor to 15-minute intervals
    data_copy['Interval'] = data_copy['Timestamp'].dt.floor('15Min').dt.time

    # Group the data by the 15-minute intervals
    time_groups = data_copy.groupby('Interval')

    # Check if time_groups is empty
    if len(time_groups) == 0:
        raise ValueError('All time groups have less than min_time_data data points')

    # Get the mean and std
    mean_list = time_groups['CGM'].mean()
    std_list = time_groups['CGM'].std()

    # Check if time_groups is empty
    if len(time_groups) == 0:
        raise ValueError('All time groups have less than min_time_data data points')

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