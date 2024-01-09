# 3rd party
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks

# Local
from ...classes import Gframe

def agp(gf: Gframe,
        add_quartiles: bool = True,
        add_deciles: bool = True,
        e: float = 1e-6,
        height: float = None,
        width: float = None,):
    '''
    Plots an Ambulatory Glucose Profile plot of the CGM values in the Gframe object

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    add_quartiles : bool, optional
        If True, the quartiles (25%, 75%) of the data will be added to the plot, by default True
    add_deciles : bool, optional
        If True, the deciles (10%, 90%) of the data will be added to the plot, by default True
    e : float, optional
        Tolerance for negligible change, by default 1e-6
    height : float, optional
        Height of the plot, by default None
    width : float, optional
        Width of the plot, by default None
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure object
    '''
    # group the data by time
    time_groups = gf.data.groupby('Time')

    # initial data
    median_series = time_groups['CGM'].median()
    if add_quartiles:
        q1_series = time_groups['CGM'].quantile(0.25)
        q3_series = time_groups['CGM'].quantile(0.75)
    if add_deciles:
        d1_series = time_groups['CGM'].quantile(0.1)
        d9_series = time_groups['CGM'].quantile(0.9)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=median_series.index, 
                             y=smooth_sequence(median_series.values, e=e), 
                             name='Smoothed Median',
                             mode='lines'))
    fig.add_trace(go.Scatter(x=median_series.index, 
                             y=median_series.values, 
                             name='Median',
                             mode='lines',
                             line=dict(color='rgba(255, 0, 0, 0.3)')))  # 0.5 is the opacity
    
    if add_quartiles:
        fig.add_trace(go.Scatter(x=q1_series.index, 
                                 y=smooth_sequence(q1_series.values, e=e), 
                                 name='25%',
                                 mode='lines'))
        fig.add_trace(go.Scatter(x=q3_series.index, 
                                 y=smooth_sequence(q3_series.values, e=e), 
                                 name='75%',
                                 mode='lines'))
    if add_deciles:
        fig.add_trace(go.Scatter(x=d1_series.index, 
                                 y=smooth_sequence(d1_series.values, e=e), 
                                 name='10%',
                                 mode='lines'))
        fig.add_trace(go.Scatter(x=d9_series.index, 
                                 y=smooth_sequence(d9_series.values, e=e), 
                                 name='90%',
                                 mode='lines'))

    fig.update_layout(xaxis_title='Time of day [h]', 
                      yaxis_title='CGM (mg/dL)',
                      height=height,
                      width=width)
    
    return fig

def smooth_sequence(y: list,
                    e: float = 1e-6):
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