# 3rd party
import plotly.graph_objects as go

# Local
from ...classes import Gframe

def freq(gf: Gframe,
              per_day: bool = True,
              target_range: list = [0,70,180],
              height: float = None,
              width: float = None):
    '''
    Plots a histogram of the Glucose rate of change

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    per_day : bool, optional
        If True, the plot will be separated by days, by default False
    target_range : list, optional
        Target range for the glucose, by default [0,70,180]
    height : float, optional
        Height of the figure, by default None
    width : float, optional
        Width of the figure, by default None

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure object
    '''
    # Check input
    if not isinstance(gf, Gframe):
        raise TypeError('gf must be a Gframe object')

    # Check input, Ensure target_range is a list with 0 and the max value of the data
    if not isinstance(target_range, list) or not all(isinstance(i, (int, float)) for i in target_range):
        raise ValueError("target_range must be a list of numbers")
    if 0 not in target_range:
        target_range = [0] + target_range
    if max(gf.data['CGM']) > target_range[-1]:
        target_range = target_range + [max(gf.data['CGM'])]

    # Get frequencies
    frequencies = gf.fd(per_day = per_day, target_range = target_range)
    range_labels = [f'{target_range[i]}-{target_range[i+1]}' for i in range(len(target_range)-1)]
    
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
        xaxis_title='Glucose (mg/dL)',
        yaxis_title='Frecuency',
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