# 3rd party
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks

# Local
from ..classes import Gframe

def trace(gf: Gframe,
          per_day: bool = True,
          show_interval: bool = True,
          interval: list[int] = [70, 180],
          height: float = None,
          width: float = None,
    ):
    '''
    Plots a line plot of the CGM values for each day in the Gframe object

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    per_day : bool, optional
        If True, the plot will be separated by days, by default True
    show_interval : bool, optional
        If True, the values in the range will be highlighted, by default True
    interval : list[int], optional
        interval to highlight, by default [70, 180]
    height : float, optional
        Height of the plot, by default None
    width : float, optional
        Width of the plot, by default None
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure object
    '''
    # This is a bit of a hack, but i did it because some datasets have different time values for each day and that
    # broke the line plot. This is a temporary fix until I can figure out a better way to do this.
    fig = go.Figure()
    
    if per_day:
        day_groups = gf.data.groupby('Day')   
        for day, day_data in day_groups:
            # Convert time to seconds past midnight
            seconds = day_data['Time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
            fig.add_trace(go.Scatter(x=seconds, y=day_data['CGM'], name=str(day),text=day_data['Time']))

    else:
        fig.add_trace(go.Scatter(x=gf.data['Timestamp'], y=gf.data['CGM'], name='CGM'))

    # Convert the x-axis labels back to time format
    fig.update_xaxes(tickvals=list(range(0, 24*3600, 1*3600)), ticktext=[f'{h}' for h in range(0, 24, 1)])
    fig.update_layout(xaxis_title='Time of day [h]', yaxis_title='CGM (mg/dL)')
    fig.update_layout(height=height, width=width)

    # Add shape for the range 70-180
    if show_interval:
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=0,
            y0=interval[0],
            x1=max(gf.data["Time"].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)),
            y1=interval[1],
            fillcolor="LightSkyBlue",
            opacity=0.5,
            layer="below",
            line_width=0,
        )

    return fig

def mean_trace(gf: Gframe,
            add_all_mean: bool = True,
            add_all_std: bool = True,
            add_std_peak: bool = True,
            add_quartiles: bool = False,
            height: float = None,
            width: float = None):
    '''
    Plots a line plot of the CGM values in the Gframe object

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    add_all_mean : bool, optional
        If True, the mean of all the data will be added to the plot, by default True
    add_all_std : bool, optional
        If True, the standard deviation of all the data will be added to the plot, by default True
    add_std_peak : bool, optional
        If True, the peaks of standard deviation will be added to the plot, by default True
    add_quartiles : bool, optional
        If True, the quartiles of the data will be added to the plot, by default False
    height : float, optional
        Height of the plot, by default None
    width : float, optional
        Width of the plot, by default None
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure object
    '''
    # Group the data by time
    time_groups = gf.data.groupby('Time')

    # Get the mean and std
    mean_list = time_groups['CGM'].mean()
    std_list = time_groups['CGM'].std()

    fig = go.Figure()

    # add traces
    fig.add_trace(go.Scatter(x=mean_list.index, y=mean_list, name='Mean CGM'))
    fig.add_trace(go.Scatter(x=mean_list.index, y=mean_list + std_list, fill=None, mode='lines', line_color='lightgray', name='+Std'))
    fig.add_trace(go.Scatter(x=mean_list.index, y=mean_list - std_list, fill='tonexty', mode='lines', line_color='lightgray', name='-Std'))
    
    # add extra traces
    if(add_all_mean):
        all_mean = np.array([gf.data['CGM'].mean()] * len(mean_list))
        fig.add_trace(go.Scatter(x=mean_list.index, y=all_mean, name='All mean', line_color='red', line=dict(dash='dot')))
    
    if(add_all_std):
        all_std = np.array([gf.data['CGM'].std()] * len(mean_list))
        fig.add_trace(go.Scatter(x=mean_list.index, y=all_mean+all_std, name='All mean+std', line_color='blue', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=mean_list.index, y=all_mean-all_std, name='All mean-std', line_color='blue', line=dict(dash='dot')))
    
    if(add_std_peak):
        std_max = [max(mean_list+std_list)] * len(mean_list)
        std_min = [min(mean_list-std_list)] * len(mean_list)
        fig.add_trace(go.Scatter(x=mean_list.index, y=std_max, name='Max std', line_color='green', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=mean_list.index, y=std_min, name='Min std', line_color='green', line=dict(dash='dot')))
    
    if(add_quartiles):
        q1 = time_groups['CGM'].quantile(0.25)
        q3 = time_groups['CGM'].quantile(0.75)
        fig.add_trace(go.Scatter(x=mean_list.index, y=q1, name='25%', line_color='orange', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=mean_list.index, y=q3, name='75%', line_color='orange', line=dict(dash='dot')))

    # update lauyout
    fig.update_layout(title_text='Mean and standard deviation for each time',
                      xaxis_title='Time of day [h]', 
                      yaxis_title='CGM (mg/dL)',
                      height=height,
                      width=width)

    return fig
        
def tir_trace(gf: Gframe,
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
    interval : list[int], optional
        interval to highlight, by default [70, 180]
    height : float, optional
        Height of the plot, by default None
    width : float, optional
        Width of the plot, by default None

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure object
    '''
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
        if (out_starts_ends and out_starts_ends[0][0] != 0) or not out_starts_ends:
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

        if show_first:
            show_first = False
            first_day = day
            
    # update layout
    fig.update_layout(
        xaxis_title='Time of day [h]',
        yaxis_title='CGM (mg/dL)',
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

def mage_trace(gf: Gframe,
               height: float = None,
               width: float = None,
               smooth: bool = True):
    '''
    Plots a line plot of the CGM values in the Gframe object separated by time in range for each day

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    height : float, optional
        Height of the plot, by default None
    width : float, optional
        Width of the plot, by default None
    smooth : bool, optional
        If True, the CGM values will be smoothed, by default True

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure object
    '''
    
    # Group the data by day
    day_groups = gf.data.groupby('Day')

    # Create figure
    fig = go.Figure()

    # Add traces
    trace_indices = {}
    shapes = {day: [] for day in day_groups.groups.keys()}
    show_first = True
    for day, day_data in day_groups:
        day_mean = day_data['CGM'].mean()
        day_std = day_data['CGM'].std()

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

        values = day_data['CGM'].values
        smoothed_values = np.copy(values)
        if smooth:
            # First 4 elements are replaced with their arithmetic mean
            smoothed_values[0:4] = smoothed_values[0:4].mean()
            # Apply weights
            for i in range(4, len(smoothed_values)-4):
                smoothed_values[i] = (values[i-4] + 2*values[i-3] + 4*values[i-2] + 8*values[i-1] + 16*values[i] + \
                                      8*values[i+1] + 4*values[i+2] + 2*values[i+3] + values[i+4]) / 46
            # Last 4 elements are replaced with their arithmetic mean
            smoothed_values[-4:] = smoothed_values[-4:].mean()

        # find peaks and nadirs
        peaks, _ = find_peaks(smoothed_values)
        nadirs, _ = find_peaks(-smoothed_values)

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
        yaxis_title='CGM (mg/dL)',
        yaxis=dict(range=[0, gf.data['CGM'].max()+10]),  # Set y-axis to start at 0 and end at the max value
        height=height,
        width=width,
        title=f'MAGE for each day',
        shapes=shapes[first_day]
    )

    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[
                            {"visible": [i in trace_indices[day] for i in range(len(fig.data))]},
                            {"shapes": shapes[day]}
                        ],
                        label=str(day),
                        method="update"
                    ) for day in day_groups.groups.keys()
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

        
