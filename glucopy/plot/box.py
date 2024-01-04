# 3rd party
import plotly.express as px

# Local
from ..classes import Gframe

def box(gf: Gframe,
        per_day: bool = True,
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
        If True, the plot will be separated by days, by default False
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
    '''
    if per_day:
        fig = px.box(gf.data, x='Day', y='CGM', points=points, color='Day')
    else:
        fig = px.box(gf.data, x='CGM', points=points)

    fig.update_layout(height=height, width=width)

    return fig

