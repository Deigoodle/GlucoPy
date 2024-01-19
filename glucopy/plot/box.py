# 3rd party
import plotly.express as px

# Local
from ..classes import Gframe

def box(gf: Gframe,
        per_day: bool = True,
        group_by_week: bool = False,
        height: float = None,
        width: float = None,
        points: str = None
        ):
    '''
    Plots a box plot of the CGM values in the Gframe object

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    per_day : bool, optional
        If True, the plot will be separated by days, by default False. If the Gframe contains a lot of days, it is
        and per_day is True, it is recommended to set group_by_week to True
    group_by_week : bool, optional
        If True, the plot will be grouped by week, by default False. Only works if per_day is True
    height : float, optional
        Height of the plot, by default None
    width : float, optional
        Width of the plot, by default None
    points : str, optional
        Show points in the plot, can be 'all', 'outliers', 'suspectedoutliers', False, by default None
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure object

    Examples
    --------
    Plot a box plot of the entire dataset

    .. ipython:: python
    
        import glucopy as gp
        gf = gp.data('prueba_1')
        gp.plot.box(gf, per_day=False)

    .. image:: /../img/box_plot_1.png
        :alt: Box plot
        :align: center
    .. raw:: html
        
        <br>

    Plot a box plot per day

    .. ipython:: python

        gp.plot.box(gf, per_day=True)

    .. image:: /../img/box_plot_2.png
        :alt: Box plot per day
        :align: center
    .. raw:: html
        
        <br>

    Plot a box plot per day, grouped by week

    .. ipython:: python
    
        gp.plot.box(gf, per_day=True, group_by_week=True)

    .. image:: /../img/box_plot_3.png
        :alt: Box plot per week
        :align: center
    '''
    # Check input
    if not isinstance(gf, Gframe):
        raise TypeError('gf must be a Gframe object')

    if per_day:
        if group_by_week:
            # Create a copy of the data DataFrame
            data_copy = gf.data.copy()

            # Aggregate data by week
            data_copy['Week'] = data_copy['Timestamp'].dt.to_period('W-Mon').dt.to_timestamp()
            fig = px.box(data_copy, x='Week', y='CGM', points=points, color='Week')
        else:
            fig = px.box(gf.data, x='Day', y='CGM', points=points, color='Day')

    else:
        fig = px.box(gf.data, x='CGM', points=points)
        fig.update_xaxes(title='Glucose')

    fig.update_layout(height=height, width=width)

    return fig

