# 3rd party
import plotly.graph_objects as go

# Local
from ...classes import Gframe

def per_day(gf: Gframe,
            num_days: int = 0,
            height: float = None,
            width: float = None,
    ):
    '''
    Plots a line plot of the CGM values for each day in the Gframe object

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    num_days : int, default 0
        Number of days to plot, if 0 all days are plotted. The days that are not plotted can still be shown by clicking on
        the legend
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
    Plot all days in the Gframe object

    .. ipython:: python

        import glucopy as gp
        gf = gp.data()
        gp.plot.per_day(gf)

    .. image:: /../img/per_day_plot_1.png
        :alt: Per day plot
        :align: center
    .. raw:: html

        <br>

    Plot only the first 10 days in the Gframe object (the rest of the days still can be shown by clicking on the legend)

    .. ipython:: python

        gp.plot.per_day(gf, num_days=10)

    .. image:: /../img/per_day_plot_2.png
        :alt: Per day plot 10 days
        :align: center
    '''
    # Check input
    if not isinstance(gf, Gframe):
        raise TypeError('gf must be a Gframe object')
    
    fig = go.Figure()
    
    day_groups = gf.data.groupby('Day')
    
    day_count = 0
    for day, day_data in day_groups:
        # Convert time to seconds past midnight
        seconds = day_data['Time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

        if num_days == 0 or day_count < num_days:
            visibility = True
        else:
            visibility = 'legendonly'
        
        fig.add_trace(go.Scatter(x=seconds, 
                                 y=day_data['CGM'], 
                                 name=str(day),
                                 text=day_data['Time'],
                                 mode='lines',
                                 visible = visibility))
        
        day_count += 1

    # Convert the x-axis labels back to time format
    fig.update_xaxes(tickvals=list(range(0, 24*3600, 1*3600)), ticktext=[f'{h}' for h in range(0, 24, 1)])

    fig.update_layout(xaxis_title='Time of day [h]', 
                      yaxis_title=f'Glucose [{gf.unit}]',
                      height=height,
                      width=width)

    return fig
