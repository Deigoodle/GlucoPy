# 3rd party
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks

# Local
from .. import Gframe

def trace(gf: Gframe,
          separate_days: bool = True,
          show_interval: bool = True,
          interval: list[int] = [70, 140],
          height: float = None,
          width: float = None,
    ):
    '''
    Plots a line plot of the CGM values for each day in the Gframe object

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    separate_days : bool, optional
        If True, the plot will be separated by days, by default True
    show_interval : bool, optional
        If True, the values in the range will be highlighted, by default True
    interval : list[int], optional
        interval to highlight, by default [70, 140]
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
    
    if separate_days:
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

    # Add shape for the range 70-140
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
              interval: list[int] = [70, 140],
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
        interval to highlight, by default [70, 140]
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
    for day, day_data in day_groups:
        out_range = ~day_data['CGM'].between(interval[0], interval[1]) 

        # Add a single trace with all the data
        fig.add_trace(go.Scatter(x=day_data['Time'], y=day_data['CGM'], name='In range', visible=show_first, mode='lines', line=dict(color='green')))

        # Add a scatter trace with mode='markers' for the out-of-range values
        fig.add_trace(go.Scatter(x=day_data[out_range]['Time'], y=day_data[out_range]['CGM'], name='Out of range', visible=show_first, mode='markers', marker=dict(color='red', size=6)))

        # Add background color for the interval
        fig.add_shape(
            type="rect",
            x0=day_data['Time'].iloc[0],
            y0=interval[0],
            x1=day_data['Time'].iloc[-1],
            y1=interval[1],
            fillcolor="LightSkyBlue",
            opacity=0.5,
            layer="below",
            line_width=0,
        )

        if show_first:
            show_first = False
            
    # update layout
    fig.update_layout(
        xaxis_title='Time of day [h]',
        yaxis_title='CGM (mg/dL)',
        yaxis=dict(range=[0, gf.data['CGM'].max()+10]),  # Set y-axis to start at 0 and end at the max value
        height=height,
        width=width,
        title_text='CGM values for each day'
    )

    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[
                            {"visible": [True if j in [2*i, 2*i+1] else False for j in range(2*len(day_groups))]},
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
               width: float = None):
    '''
    Plots a line plot of the CGM values showing the MAGE for each day

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
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
    trace_indices = {}
    shapes = {day: [] for day in day_groups.groups.keys()}
    show_first = True
    for day, day_data in day_groups:
        # get axes values
        x = np.array(day_data['Time'])
        y = np.array(day_data['CGM'])

        # Initialize the list of traces for this day
        trace_indices[day] = []

        # CGM
        fig.add_trace(go.Scatter(x=x, y=y, name=str(day), visible=show_first))
        trace_indices[day].append(len(fig.data) - 1)

        # mean and std
        mean = y.mean()
        std = y.std()

        fig.add_trace(go.Scatter(x=x, y=[mean] * len(day_data), name='Mean',line=dict(dash='dot'), line_color='red',visible=show_first))
        trace_indices[day].append(len(fig.data) - 1)

        fig.add_trace(go.Scatter(x=x, y=[mean + std] * len(day_data), name='Mean + std',line=dict(dash='dot'),line_color='cyan', visible=show_first))
        trace_indices[day].append(len(fig.data) - 1)

        fig.add_trace(go.Scatter(x=x, y=[mean - std] * len(day_data), name='Mean - std',line=dict(dash='dot'),line_color='cyan', visible=show_first))
        trace_indices[day].append(len(fig.data) - 1)

        # Find peaks and nadirs using scipy's find_peaks
        peaks, _ = find_peaks(y)
        nadirs, _ = find_peaks(-y)

        # Determine whether to start with a peak or a nadir
        first_peak = next((i for i in peaks if y[i] > mean + std), None)
        first_nadir = next((i for i in nadirs if y[i] < mean - std), None)
        if first_peak is None:
            start_with_peak = False
        elif first_nadir is None:
            start_with_peak = True
        else:
            start_with_peak = first_peak < first_nadir

        # Initialize variables to keep track of the maximum peak and minimum nadir
        max_peak = None
        min_nadir = None

        # Iterate over the peaks and nadirs
        for peak, nadir in zip(peaks, nadirs):
            if start_with_peak:
                # If the current peak is above mean + std and is higher than the current maximum peak, update max_peak
                if (y[peak] > mean + std) and (max_peak is None or y[peak] > y[max_peak]):
                    max_peak = peak
                    min_nadir = None  # Reset min_nadir whenever a new max_peak is found
            else:
                # If the current nadir is below mean - std and is lower than the current minimum nadir, update min_nadir
                if (y[nadir] < mean - std) and (min_nadir is None or y[nadir] < y[min_nadir]):
                    min_nadir = nadir
                    max_peak = None  # Reset max_peak whenever a new min_nadir is found

            # If max_peak is not None, look for a nadir that is below mean - std and is lower than the current minimum nadir
            if (max_peak is not None) and (y[nadir] < mean - std) and (min_nadir is None or y[nadir] < y[min_nadir]):
                min_nadir = nadir

            # If min_nadir is not None, look for a peak that is above mean + std and is higher than the current maximum peak
            if (min_nadir is not None) and (y[peak] > mean + std) and (max_peak is None or y[peak] > y[max_peak]):
                max_peak = peak

            # If both max_peak and min_nadir have been found, add a rectangle shape between them and reset them to None
            if max_peak is not None and min_nadir is not None:
                shape = dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=x[min(max_peak, min_nadir)],
                    y0=y[min(max_peak, min_nadir)],
                    x1=x[max(max_peak, min_nadir)],
                    y1=y[max(max_peak, min_nadir)],
                    fillcolor="rgba(255, 0, 0, 0.3)",
                    line=dict(width=0),
                )
                shapes[day].append(shape)
                
                # Add annotations for the peak and nadir as scatter traces
                fig.add_trace(
                    go.Scatter(
                        x=[x[max_peak]],
                        y=[y[max_peak]],
                        mode='text',
                        text=[y[max_peak]],
                        textposition="top right",
                        showlegend=False,
                        visible=show_first
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[x[min_nadir]],
                        y=[y[min_nadir]],
                        mode='text',
                        text=[y[min_nadir]],
                        textposition="bottom right",
                        showlegend=False,
                        visible=show_first
                    )
                )
                
                max_peak = None
                min_nadir = None
                trace_indices[day].append(len(fig.data) - 1)
                trace_indices[day].append(len(fig.data) - 2)  # Add the indices of the annotation traces
                
        if show_first:
            show_first = False

    # update layout
    fig.update_layout(
        xaxis_title='Time of day [h]',
        yaxis_title='CGM (mg/dL)',
        yaxis=dict(range=[0, gf.data['CGM'].max()+10]),  # Set y-axis to start at 0 and end at the max value
        height=height,
        width=width,
        title_text='CGM values for each day'
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


        
