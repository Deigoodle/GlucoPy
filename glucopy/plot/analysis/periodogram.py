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
                width: float = None
                ):
    '''
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
    Plot the best-fit curve obtained by a Lomb-Scargle periodogram per day

    .. ipython:: python

        import glucopy as gp
        gf = gp.data()
        gp.plot.periodogram(gf)

    .. image:: /../img/periodogram_plot_1.png
        :alt: Periodogram plot per day
        :align: center
    .. raw:: html

        <br>

    Plot the best-fit curve obtained by a Lomb-Scargle periodogram for the entire dataset

    .. ipython:: python

        fig = gp.plot.periodogram(gf, per_day=False)

    .. image:: /../img/periodogram_plot_2.png
        :alt: Periodogram plot
        :align: center
    '''
    # Check input
    if not isinstance(gf, Gframe):
        raise TypeError('gf must be a Gframe object')

    # Create subplots
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Power vs Frequency', 'Original Data and Periodogram Fit'))

    if per_day:
        day_groups = gf.data.groupby('Day')

        show_first = True
        for _, day_data in day_groups:
            compute_periodogram(df=day_data, 
                                fig=fig, 
                                show_trace=show_first, 
                                per_day=per_day)

            if show_first:
                show_first = False

    else:
        compute_periodogram(df=gf.data,
                            fig=fig,
                            show_trace=True,
                            per_day=per_day)

    # Set layout
    fig.update_layout(title='Lomb-Scargle periodogram',
                      height=height,
                      width=width)

    # Set axis titles
    fig.update_xaxes(title_text='Time [h]', row=2, col=1)
    fig.update_yaxes(title_text=f'Glucose [{gf.unit}]', row=2, col=1)
    fig.update_xaxes(title_text='Frequency [1/h]', row=1, col=1)
    fig.update_yaxes(title_text='Power', row=1, col=1)

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

def compute_periodogram(df, fig, show_trace, per_day):
    '''
    Calculates the periodogram and add the traces to the figure.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    fig : plotly.graph_objects.Figure
        Figure to add the traces.
    show_trace : bool
        If True, the traces will be visible.
    per_day : bool
        If True, the traces will use the 'Time' column as x-axis. If False, the traces will use the 'Timestamp' column as x-axis.
    '''
    # Get time values as numbers
    time_diff = df['Timestamp'].diff().dt.total_seconds().fillna(0) / 3600
    time_values = time_diff.cumsum().values

    # Compute periodogram
    ls = LombScargle(time_values, df['CGM'])
    frequency, power = ls.autopower()

    # Get best-fit curve
    best_frequency = frequency[np.argmax(power)]
    best_curve = ls.model(time_values, best_frequency)

    # Get X values
    if per_day:
        x_values = df['Time']
    else:
        x_values = df['Timestamp']

    fig.add_trace(go.Scatter(x=x_values, 
                             y=df['CGM'],
                             name='Original Data', 
                             visible=show_trace),
                  row=2,
                  col=1)
    fig.add_trace(go.Scatter(x=x_values, 
                             y=best_curve, 
                             name='Periodogram Fit',
                             visible=show_trace),
                  row=2,
                  col=1)

    # Plot Power vs Frequency
    fig.add_trace(go.Scatter(x=frequency, 
                             y=power, 
                             name='Power', 
                             visible=show_trace), 
                  row=1, 
                  col=1)

