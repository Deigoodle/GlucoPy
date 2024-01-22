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
         min_time_data: int = 0,
         height: float = None,
         width: float = None):
    '''
    Plots a line plot of the CGM values in the Gframe object

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
    min_time_data : int, default 0
        Minimum number of data points for each time to be considered in the Gframe, if 0, all the data will be considered
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
    Plot the mean and standard deviation for each time of day, the problem with the example data is that has 112 days
    and data for every minute of the day, so the plot is not very useful, but after filtering the data, the plot is
    more useful

    .. ipython:: python

        import glucopy as gp
        gf = gp.data()
        gp.plot.mean(gf)

    .. image:: /../img/mean_plot_1.png
        :alt: Mean plot
        :align: center
    .. raw:: html
        
        <br>

    Plot the mean and standard deviation for each time of day, but filtering the data to only include times with at
    least 10 data points

    .. ipython:: python
    
        gp.plot.mean(gf, min_time_data=10)

    .. image:: /../img/mean_plot_2.png
        :alt: Mean plot with filtered data
        :align: center
    .. raw:: html
    
        <br>

    The resulting plot is more useful, but the problem is that the data is not evenly distributed, but it should work
    fine for Datasets with more evenly distributed data
    '''
    # Check input
    if not isinstance(gf, Gframe):
        raise TypeError('gf must be a Gframe object')
    if min_time_data is not None:
        if not isinstance(min_time_data, int):
            raise TypeError('min_time_data must be an integer')
        if min_time_data < 0:
            raise ValueError('min_time_data must be greater than 0')
    
    # Group the data by time
    time_groups = gf.data.groupby('Time')
    # print a list with the len of each group\

    # Filter the data
    if min_time_data != 0:
        time_groups = time_groups.filter(lambda x: len(x) >= min_time_data)
        time_groups = time_groups.groupby('Time')

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