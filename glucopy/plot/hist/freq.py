# 3rd party
import plotly.graph_objects as go
import numpy as np

# Local
from ...classes import Gframe

def freq(gf: Gframe,
         per_day: bool = True,
         interval: list = [0,70,180],
         count: bool = False,
         height: float = None,
         width: float = None):
    '''
    Plots a histogram of the frequency of the glucose in the target range

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    per_day : bool, default True
        If True, the plot will be separated by days
    interval : list, default [0,70,180]
        Target range for the glucose
    count : bool, default False
        If True, the y axis will be the count of the glucose in the target range. If False, the y axis will be the
        frequency of the glucose in the target range
    height : float, default None
        Height of the figure
    width : float, default None
        Width of the figure

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure object

    Examples
    --------
    Plot the frequency of the glucose per day in the target range [0,70,180] (default), and clicking on the second day on
    the dropdown menu

    .. ipython:: python

        import glucopy as gp
        gf = gp.data()
        gp.plot.freq(gf)

    .. image:: /../img/freq_plot_1.png
        :alt: Frequency plot per day
        :align: center
    .. raw:: html
        
        <br>

    Plot the frequency of the glucose for the entire dataset and the target range [0,100,200,300]

    .. ipython:: python

        fig = gp.plot.freq(gf, per_day=False, interval=[0,100,200,300])

    .. image:: /../img/freq_plot_2.png
        :alt: Frequency plot
        :align: center
    '''
    # Check input
    if not isinstance(gf, Gframe):
        raise TypeError('gf must be a Gframe object')

    # Check input, Ensure interval is a list or numpy array of numbers
    if not isinstance(interval, (list, np.ndarray)):
        raise ValueError("interval must be a list or numpy array of numbers")
    
    # Convert interval to a list if it's a numpy array
    if isinstance(interval, np.ndarray):
        interval = interval.tolist()
    
    # Add 0 to the target range if it is not present to count the time below the target range
    if 0 not in interval:
        interval = [0] + interval

    # Add the max value of the data to the target range if it is not present to count the time above the target range
    max_value = max(gf.data['CGM'])
    if max_value <= interval[-1]:
        max_value = interval[-1] + 1
    if max_value > interval[-1]:
        interval = interval + [max_value]

    # Get frequencies
    frequencies = gf.fd(per_day = per_day, interval = interval, count = count)
    range_labels = [f'{interval[i]}-{interval[i+1]}' for i in range(len(interval)-1)]
    
    fig = go.Figure()
    first_day = ''

    if per_day:
        show_first = True
        for day, freq in frequencies.items():
            fig.add_trace(go.Bar(x=range_labels, y=freq, name=str(day), visible=show_first))
            if show_first:
                first_day = day
                show_first = False
    else:
        fig.add_trace(go.Bar(x=range_labels, y=frequencies))
        
    fig.update_layout(
        title=f'Glucose Frequency {first_day}',
        xaxis_title=f'Glucose [{gf.unit}]',
        yaxis_title='Number of readings' if count else 'Frequency',
        height=height,
        width=width
    )

    # Add Dropdown
    if per_day:
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=[{"visible": [i==j for j in range(len(frequencies))]},
                                  {"title": f'Glucose Frequency {day}'}],
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