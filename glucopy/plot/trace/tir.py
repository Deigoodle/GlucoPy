# 3rd party
import plotly.graph_objects as go

# Local
from ...classes import Gframe

def tir(gf: Gframe,
        interval: list[int] = [70, 180],
        height: float = None,
        width: float = None
    ):
    '''
    Plots a line plot of the CGM values in the Gframe object separated by time in range for each day

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    interval : list[int], default [70, 180]
        interval to highlight
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
    Plot the CGM values for each day in the Gframe object highlighting the values in the interval [70, 180] (default), and
    clicking on the second day on the dropdown menu

    .. ipython:: python

        import glucopy as gp
        gf = gp.data()
        gp.plot.tir(gf)

    .. image:: /../img/tir_plot_1.png
        :alt: TIR plot
        :align: center
    .. raw:: html

        <br>

    Plot the CGM values for each day in the Gframe object highlighting the values in the interval [100, 200], and
    clicking on the second day on the dropdown menu

    .. ipython:: python

        gp.plot.tir(gf, interval=[100, 200])

    .. image:: /../img/tir_plot_2.png
        :alt: TIR plot 100-200
        :align: center
    '''
    # Check input
    if not isinstance(gf, Gframe):
        raise TypeError('gf must be a Gframe object')
    
    # Group the data by day
    day_groups = gf.data.groupby('Day')

    # Create figure
    fig = go.Figure()

    # Add traces 
    show_first = True
    shapes = {day: [] for day in day_groups.groups.keys()}
    for day, day_data in day_groups:
        out_range = ~day_data['CGM'].between(interval[0], interval[1]) 

        # Add a single trace with all the data
        fig.add_trace(go.Scatter(x=day_data['Time'], 
                                 y=day_data['CGM'], 
                                 name='In range', 
                                 visible=show_first, 
                                 mode='markers+lines',
            )
        )

        # Get continuous intervals out_range
        out_starts_ends = []
        start = None
        for i, value in enumerate(out_range):
            if start is None and value:
                start = i
            elif start is not None and not value:
                out_starts_ends.append((start, i))
                start = None
        if start is not None:
            out_starts_ends.append((start, len(out_range)))
        
        # Add background color for the out of range intervals
        previous_end = None
        for start, end in out_starts_ends:
            if day_data['CGM'].iloc[start] >= interval[1]:
                y0 = interval[1]
                y1 = day_data['CGM'].max()
            elif day_data['CGM'].iloc[start] <= interval[0]:
                y0 = day_data['CGM'].min()
                y1 = interval[0]

            # Add red rectangle
            out_shape = dict(
                    type="rect",
                    x0=day_data['Time'].iloc[start],
                    y0=y0,
                    x1=day_data['Time'].iloc[end-1],
                    y1=y1,
                    fillcolor="rgba(255, 0, 0, 0.5)",
                    line=dict(width=0),
                )
            shapes[day].append(out_shape)

            # Add green rectangle
            if previous_end is not None:
                in_shape = dict(
                    type="rect",
                    x0=day_data['Time'].iloc[previous_end-1],
                    y0=interval[0],
                    x1=day_data['Time'].iloc[start],
                    y1=interval[1],
                    fillcolor="rgba(0, 255, 0, 0.5)",
                    line=dict(width=0),
                )
                shapes[day].append(in_shape)
            
            previous_end = end
        
        # Add green rectangle for the last interval
        if previous_end != len(out_range) and previous_end is not None:
            in_shape = dict(
                type="rect",
                x0=day_data['Time'].iloc[previous_end-1],
                y0=interval[0],
                x1=day_data['Time'].iloc[-1],
                y1=interval[1],
                fillcolor="rgba(0, 255, 0, 0.5)",
                line=dict(width=0),
            )
            shapes[day].append(in_shape)

        # Add green rectangle for the first interval
        if out_starts_ends and out_starts_ends[0][0] != 0:
            in_shape = dict(
                type="rect",
                x0=day_data['Time'].iloc[0],
                y0=interval[0],
                x1=day_data['Time'].iloc[out_starts_ends[0][0]],
                y1=interval[1],
                fillcolor="rgba(0, 255, 0, 0.5)",
                line=dict(width=0),
            )
            shapes[day].append(in_shape)

        # Add green rectangle for the entire day if all values are in range
        if not out_starts_ends:
            in_shape = dict(
                type="rect",
                x0=day_data['Time'].iloc[0],
                y0=interval[0],
                x1=day_data['Time'].iloc[-1],
                y1=interval[1],
                fillcolor="rgba(0, 255, 0, 0.5)",
                line=dict(width=0),
            )
            shapes[day].append(in_shape)

        if show_first:
            show_first = False
            first_day = day
            
    # update layout
    fig.update_layout(
        xaxis_title='Time of day [h]',
        yaxis_title=f'CGM [{gf.unit}]',
        yaxis=dict(range=[0, gf.data['CGM'].max()+10]),  # Set y-axis to start at 0 and end at the max value
        height=height,
        width=width,
        title_text='CGM values for each day',
        shapes=shapes[first_day]
    )

    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[
                            {"visible": [True if j == i else False for j in range(len(day_groups))]},
                            {"shapes": shapes[day]}
                        ],
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