# 3rd party
import plotly.graph_objects as go

# Local
from ...classes import Gframe
from ...utils import time_factor

def roc(gf: Gframe,
        per_day: bool = True,
        height: float = None,
        width: float = None):
    '''
    Plots a histogram of the Glucose rate of change

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

    Examples
    --------
    Plot the glucose rate of change per day (default), and clicking on the second day on the dropdown menu

    .. ipython:: python

        import glucopy as gp
        gf = gp.data()
        gp.plot.roc(gf)

    .. image:: /../img/roc_plot_1.png
        :alt: Rate of change histogram per day
        :align: center
    .. raw:: html
        
        <br>

    Plot the glucose rate of change for the entire dataset

    .. ipython:: python
    
        gp.plot.roc(gf, per_day=False)

    .. image:: /../img/roc_plot_2.png
        :alt: Rate of change histogram
        :align: center
    '''
    # Check input
    if not isinstance(gf, Gframe):
        raise TypeError('gf must be a Gframe object')
    
    fig = go.Figure()
    
    if per_day:
        day_groups = gf.data.groupby('Day')
        mean = []
        std = []
        show_first = True
        for day, day_data in day_groups:
            x = day_data['CGM'].diff().abs() / (day_data['Timestamp'].diff().dt.total_seconds() / 60)
            mean.append(x.mean())
            std.append(x.std())
            fig.add_trace(go.Histogram(x=x, name=str(day), visible=show_first, xbins=dict(size=0.1),
                                       marker=dict(line=dict(color='black', width=1))))
            if show_first:
                fig.update_layout(title=f'Glucose Rate of Change {day}. Mean: {mean[0]:.2f} Std: {std[0]:.2f}')
                show_first = False
    else:
        x = gf.data['CGM'].diff().abs() / (gf.data['Timestamp'].diff().dt.total_seconds() / 60)
        fig.add_trace(go.Histogram(x=x, xbins=dict(size=0.1), 
                                   marker=dict(line=dict(color='black', width=1))))
        fig.update_layout(title=f'Glucose Rate of Change. Mean: {x.mean():.2f} Std: {x.std():.2f}')

    fig.update_layout(
        xaxis_title=f'Glucose Rate of Change (mg/dL/m)',
        yaxis_title='Measurements',
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
                            args=[{"visible": [i==j for j in range(len(day_groups))]},
                                  {"title": f'Glucose Rate of Change {day}. Mean: {mean[i]:.2f} Std: {std[i]:.2f}'}],
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

    

    
