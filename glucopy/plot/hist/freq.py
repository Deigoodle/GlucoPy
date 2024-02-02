# 3rd party
import plotly.graph_objects as go

# Local
from ...classes import Gframe

def freq(gf: Gframe,
         per_day: bool = True,
         target_range: list = [0,70,180],
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
    target_range : list, default [0,70,180]
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

        fig = gp.plot.freq(gf, per_day=False, target_range=[0,100,200,300])

    .. image:: /../img/freq_plot_2.png
        :alt: Frequency plot
        :align: center
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
    frequencies = gf.fd(per_day = per_day, target_range = target_range, count = count)
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