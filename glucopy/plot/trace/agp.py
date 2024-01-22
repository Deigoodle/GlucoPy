# 3rd party
import plotly.graph_objects as go
import numpy as np
#from scipy.ndimage import median_filter
#from scipy.signal import savgol_filter

# Local
from ...classes import Gframe

def agp(gf: Gframe,
        add_quartiles: bool = True,
        add_deciles: bool = True,
        e: float = 1.0,
        height: float = None,
        width: float = None,):
    '''
    Plots an Ambulatory Glucose Profile plot of the CGM values in the Gframe object

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    add_quartiles : bool, default True
        If True, the quartiles (25%, 75%) of the data will be added to the plot
    add_deciles : bool, default True
        If True, the deciles (10%, 90%) of the data will be added to the plot
    e : non negative float, default 1.0
        Tolerance for negligible change
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
    Plot the Ambulatory Glucose Profile

    .. ipython:: python

        import glucopy as gp
        gf = gp.data()
        gp.plot.agp(gf)

    .. image:: /../img/agp_plot.png
        :alt: Ambulatory Glucose Profile
        :align: center
    '''
    # Check input
    if not isinstance(gf, Gframe):
        raise TypeError('gf must be a Gframe object')
    if e < 0:
        raise ValueError('e must be non negative')
    
    # group the data by hour
    time_groups = gf.data.groupby(gf.data['Timestamp'].dt.hour)

    # initial data
    median_series = time_groups['CGM'].median()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=median_series.index, 
                             y=smooth_sequence(median_series.values, e=e), 
                             name='50%',
                             mode='lines',
                             line_color='blue'
                             )
                 )
    
    if add_quartiles:
        q1_series = time_groups['CGM'].quantile(0.25)
        q3_series = time_groups['CGM'].quantile(0.75)
        fig.add_trace(go.Scatter(x=q3_series.index, 
                                 y=smooth_sequence(q3_series.values, e=e), 
                                 name='75%',
                                 fill=None,
                                 mode='lines',
                                 line_color='lightblue',
                                 legendgroup='25-75',
                                 showlegend=False
                                )
                     )
        fig.add_trace(go.Scatter(x=q1_series.index, 
                                 y=smooth_sequence(q1_series.values, e=e), 
                                 name='25% - 75%',
                                 fill='tonexty',
                                 mode='lines',
                                 line_color='lightblue',
                                 fillcolor='rgba(9,174,229,0.5)',
                                 legendgroup='25-75'
                                )
                     )
        
    if add_deciles:
        d1_series = time_groups['CGM'].quantile(0.1)
        d9_series = time_groups['CGM'].quantile(0.9)
        fig.add_trace(go.Scatter(x=d9_series.index, 
                                 y=smooth_sequence(d9_series.values, e=e), 
                                 name='90%',
                                 fill=None,
                                 line=dict(dash='dot'),
                                 line_color='lightblue',
                                 legendgroup='10-90',
                                 showlegend=False
                                )
                     )
        fig.add_trace(go.Scatter(x=d1_series.index, 
                                 y=smooth_sequence(d1_series.values, e=e), 
                                 name='10% - 90%',
                                 fill='tonexty',
                                 line=dict(dash='dot'),
                                 line_color='lightblue',
                                 fillcolor='rgba(9,174,229,0.2)',
                                 legendgroup='10-90',
                                )
                     )
        

    fig.update_layout(
        title='Ambulatory Glucose Profile',
        xaxis_title='Time of day [h]', 
        yaxis_title='CGM (mg/dL)',
        height=height,
        width=width,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(24))  # list of all hours
        )
    )
    
    return fig

def smooth_sequence(y: list,
                    e: float = 1):
    '''
    Smooths a sequence of values using the median filter

    Parameters
    ----------
    y : list
        Sequence of values to smooth
    e : float, default 
        Tolerance for negligible change, by default 1

    Returns
    -------
    smoothed_y : list
        Smoothed sequence
    '''

    n = len(y)
    smoothed_y = np.copy(y)
    # Perform smoothing until negligible change
    while True:
        previous_smoothed_y = np.copy(smoothed_y)

        for i in range(2,n-2):
            u = np.median([smoothed_y[i-1], (3 * smoothed_y[i-1] - smoothed_y[i-2]) / 2, smoothed_y[i]])
            v = np.median([smoothed_y[i-1], smoothed_y[i], smoothed_y[i+1]])
            w = np.median([smoothed_y[i+1], (3 * smoothed_y[i+1] - smoothed_y[i+2]) / 2, smoothed_y[i]])

            smoothed_y[i] = np.median([u, v, w])

        # calculate residuals
        residuals = y - smoothed_y

        # smooth residuals
        for i in range(2,n-2):
            u = np.median([residuals[i-1], (3 * residuals[i-1] - residuals[i-2]) / 2, residuals[i]])
            v = np.median([residuals[i-1], residuals[i], residuals[i+1]])
            w = np.median([residuals[i+1], (3 * residuals[i+1] - residuals[i+2]) / 2, residuals[i]])

            residuals[i] = np.median([u, v, w])
        
        # update y
        smoothed_y += residuals

        # Check for negligible change
        if np.allclose(smoothed_y, previous_smoothed_y, atol=e):
            break

    return smoothed_y