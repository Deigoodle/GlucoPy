# 3rd party
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from astropy.timeseries import LombScargle
import numpy as np

# Local
from ...classes import Gframe

def periodogram(gf: Gframe,
                per_day: bool = True,
                height: float = None,
                width: float = None):
    '''
    Plots the best-fit curve obtained by a Lomb-Scargle periodogram using astropy.timeseries.LombScargle

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    per_day : bool, default True
        If True, the plot will be separated by days
    height : float, default None
        Height of the figure
    width : float, default None 
        Width of the figure

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure object
    '''
    # Check input
    if not isinstance(gf, Gframe):
        raise TypeError('gf must be a Gframe object')
    
    # Create subplots
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Original Data and Periodogram Fit', 'Frequency vs Power'))

    if per_day:
        day_groups = gf.data.groupby('Day')

        show_first = True
        for _, day_data in day_groups:
            # Get time values as numbers
            time_diff = day_data['Timestamp'].diff().dt.total_seconds().fillna(0) / 3600
            time_values = time_diff.cumsum().values

            # Compute periodogram
            ls = LombScargle(time_values, day_data['CGM'])
            frequency, power = ls.autopower()

            # Get best-fit curve
            best_frequency = frequency[np.argmax(power)]
            best_curve = ls.model(time_values, best_frequency)

            # Plot best-fit curve
            fig.add_trace(go.Scatter(x=day_data['Time'], 
                                     y=day_data['CGM'],
                                     name='Original Data', 
                                     visible=show_first),
                          row=2,
                          col=1)
            fig.add_trace(go.Scatter(x=day_data['Time'], 
                                     y=best_curve, 
                                     name='Periodogram Fit',
                                     visible=show_first),
                          row=2,
                          col=1)
            
            # Plot Power vs Frequency
            fig.add_trace(go.Scatter(x=frequency, y=power, name='Power', visible=show_first), row=1, col=1)

            if show_first:
                show_first = False
    else:
        # Get time values as numbers
        time_diff = gf.data['Timestamp'].diff().dt.total_seconds().fillna(0) / 3600
        time_values = time_diff.cumsum().values

        # Compute periodogram
        ls = LombScargle(time_values, gf.data['CGM'])
        frequency, power = ls.autopower()

        # Get best-fit curve
        best_frequency = frequency[np.argmax(power)]
        best_curve = ls.model(time_values, best_frequency)
        
        # Plot best-fit curve
        fig.add_trace(go.Scatter(x=gf.data['Timestamp'], 
                                 y=gf.data['CGM'],
                                 name='Original Data'),
                      row=2,
                      col=1)
        fig.add_trace(go.Scatter(x=gf.data['Timestamp'], 
                                 y=best_curve, 
                                 name='Periodogram Fit'),
                      row=2,
                      col=1)
        
        # Plot Power vs Frequency
        fig.add_trace(go.Scatter(x=frequency, y=power, name='Power'), row=1, col=1)
    
    # Set layout
    fig.update_layout(title='Lomb-Scargle periodogram',
                      height=height,
                      width=width)
    
    # Set axis titles
    fig.update_xaxes(title_text='Time [h]', row=1, col=1)
    fig.update_yaxes(title_text='Glucose [mg/dL]', row=1, col=1)
    fig.update_xaxes(title_text='Frequency [1/h]', row=2, col=1)
    fig.update_yaxes(title_text='Power', row=2, col=1)
    
    # Add Dropdown
    if per_day:
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=[{"visible": [i==j//3 for j in range(len(day_groups)*3)],
                                  "title": f'Periodogram {day}'}],
                            label=str(day),
                            method="update"
                        ) for i, day in enumerate(gf.data['Day'].unique())
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


