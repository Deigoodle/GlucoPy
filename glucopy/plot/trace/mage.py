# 3rd party
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks

# Local
from ...classes import Gframe 

def mage(gf: Gframe,
         height: float = None,
         width: float = None,):
    '''
    Plots a line plot of the CGM values in the Gframe object separated by time in range for each day

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
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
    Plot the MAGE, and clicking on the second day on the dropdown menu

    .. ipython:: python

        import glucopy as gp
        gf = gp.data()
        gp.plot.mage(gf)

    .. image:: /../img/mage_plot.png
        :alt: MAGE
        :align: center
    '''
    # Check input
    if not isinstance(gf, Gframe):
        raise TypeError('gf must be a Gframe object')
    
    # Group the data by day
    day_groups = gf.data.groupby('Day')
    day_means = []
    day_stds = []

    # Create figure
    fig = go.Figure()

    # Add traces
    trace_indices = {}
    shapes = {day: [] for day in day_groups.groups.keys()}
    show_first = True
    for day, day_data in day_groups:
        # Calculate the mean and std for each day
        day_mean = day_data['CGM'].mean()
        day_std = day_data['CGM'].std()

        # Save the mean and std for each day
        day_means.append(day_mean)
        day_stds.append(day_std)

        # Initialize trace_indices[day] as an empty list
        trace_indices[day] = []

        fig.add_trace(go.Scatter(x=day_data['Time'], y=day_data['CGM'], name=str(day), visible=show_first))
        trace_indices[day].append(len(fig.data) - 1)

        fig.add_trace(go.Scatter(x=day_data['Time'], y=[day_mean] * len(day_data), name='Mean',line=dict(dash='dot'), line_color='red',visible=show_first))
        trace_indices[day].append(len(fig.data) - 1)

        fig.add_trace(go.Scatter(x=day_data['Time'], y=[day_mean + day_std] * len(day_data), name='Mean + std',line=dict(dash='dot'),line_color='cyan', visible=show_first))
        trace_indices[day].append(len(fig.data) - 1)

        fig.add_trace(go.Scatter(x=day_data['Time'], y=[day_mean - day_std] * len(day_data), name='Mean - std',line=dict(dash='dot'),line_color='cyan', visible=show_first))
        trace_indices[day].append(len(fig.data) - 1)

        # find peaks and nadirs
        peaks, _ = find_peaks(day_data['CGM'].values)
        nadirs, _ = find_peaks(-day_data['CGM'].values)

        # make sure that the peaks and nadirs have the same size
        if peaks.size > nadirs.size:
            nadirs = np.append(nadirs, day_data['CGM'].size - 1)
        elif peaks.size < nadirs.size:
            peaks = np.append(peaks, day_data['CGM'].size - 1)
        
        # calculate the difference between the peaks and the nadirs
        differences = np.abs(day_data['CGM'].iloc[peaks].values - day_data['CGM'].iloc[nadirs].values)
        
        peak_nadir_pairs = np.array(list(zip(peaks, nadirs)))
        peak_nadir_pairs = peak_nadir_pairs[differences > day_std]

        # add shapes
        for peak, nadir in peak_nadir_pairs:
            shape = dict(
                type="rect",
                xref="x",
                yref="y",
                x0=day_data['Time'].iloc[peak],
                y0=day_data['CGM'].iloc[peak],
                x1=day_data['Time'].iloc[nadir],
                y1=day_data['CGM'].iloc[nadir],
                fillcolor="rgba(255, 204, 0, 0.5)",
                line=dict(width=0),
            )
            shapes[day].append(shape)

            # Add annotations for the peak and nadir as scatter traces
            fig.add_trace(
                go.Scatter(
                    x=[day_data['Time'].iloc[peak]],
                    y=[day_data['CGM'].iloc[peak]],
                    mode='text',
                    text=[day_data['CGM'].iloc[peak]],
                    textposition="top right",
                    showlegend=False,
                    visible=show_first
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[day_data['Time'].iloc[nadir]],
                    y=[day_data['CGM'].iloc[nadir]],
                    mode='text',
                    text=[day_data['CGM'].iloc[nadir]],
                    textposition="bottom left",
                    showlegend=False,
                    visible=show_first
                )
            )
            trace_indices[day].append(len(fig.data) - 1)
            trace_indices[day].append(len(fig.data) - 2)

        if show_first:
            show_first = False
            first_day = day

    # update layout
    fig.update_layout(
        xaxis_title='Time of day [h]',
        yaxis_title=f'Glucose [{gf.unit}]',
        yaxis=dict(range=[0, gf.data['CGM'].max()+10]),  # Set y-axis to start at 0 and end at the max value
        height=height,
        width=width,
        title=f'MAGE for each day.   Mean = {day_means[0]:.2f}   Std = {day_stds[0]:.2f}',
        shapes=shapes[first_day]
    )

    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[
                            {"visible": [i in trace_indices[day] for i in range(len(fig.data))]
                             },  
                            {"shapes": shapes[day], 
                             "title": f'MAGE for each day.   Mean = {day_means[i]:.2f}   Std = {day_stds[i]:.2f}'
                             }  
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