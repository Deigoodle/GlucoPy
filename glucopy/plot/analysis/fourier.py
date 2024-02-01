# 3rd party
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import curve_fit

# Local
from ...classes import Gframe

def fourier(gf: Gframe,
            n: int = 10,
            amplitude_guess: [float] = None,
            phase_guess: [float] = None,
            num_days: int = 0,
            height: float = None,
            width: float = None
            ):
    '''
    Plots the best-fit curve obtained by a Fourier series using scipy.optimize.curve_fit. The Fourier series is given by:

    .. math::

        f(t) = m + \sum_{i=1}^{N} A_i \cos(\\frac{2\pi i (t-PHS_i)}{hourRange})

    where:
    
    - :math:`m` is :math:`\\frac{AUC(h)}{hourRange}`.
    - :math:`AUC` is the area under the curve.
    - :math:`A_i` is the amplitude of the :math:`i`-th harmonic.
    - :math:`PHS_i` is the phase shift of the :math:`i`-th harmonic.

    Parameters
    ----------
    gf : Gframe
        Gframe object to plot
    n : int, default 10
        (:math:`N`) Number of harmonics to fit
    amplitude_guess : [float], default None
        Initial guess for the amplitudes, if None, all amplitudes are set to 1
    phase_guess : [float], default None
        Initial guess for the phase shifts, if None, all phase shifts are set to 1
    num_days : int, default 0
        Number of days to plot, if 0 all days are plotted.
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
    Plot the Fourier Transformation with 5 harmonics for the first 5 days.

    .. ipython:: python

        import glucopy as gp
        gf = gp.data()
        gp.plot.fourier(gf, n=5, num_days=5)

    .. image:: /../img/fourier_plot_1.png
        :alt: Fourier 5n
        :align: center
    .. raw:: html

        <br>

    Plot the Fourier Transformation with 10 harmonics for the first 5 days.

    .. ipython:: python

        fig = gp.plot.fourier(gf, n=10, num_days=5)

    .. image:: /../img/fourier_plot_2.png
        :alt: Fourier 10n
        :align: center

    Note that in both cases, the day 2020-11-27 fails to fit, that is because the number of glucose values is too low
    to fit the Fourier series. To solve this issue, we can decrease the number of harmonics, but this will decrease the
    quality of the fit for the rest of the days.

    '''
    # Check input
    if not isinstance(gf, Gframe):
        raise TypeError('gf must be a Gframe object')
    
    if not isinstance(n, int):
        raise TypeError('n must be an integer')
    
    if amplitude_guess:
        if not isinstance(amplitude_guess, list):
            raise TypeError('amplitude_guess must be a list')
        if len(amplitude_guess) != n:
            raise ValueError('amplitude_guess must have the same length as n')
    else:
        amplitude_guess = [1] * n

    if phase_guess:
        if not isinstance(phase_guess, list):
            raise TypeError('phase_guess must be a list')
        if len(phase_guess) != n:
            raise ValueError('phase_guess must have the same length as n')
    else:
        phase_guess = [1] * n
        
    # Group the data by day
    day_groups = gf.data.groupby('Day')

    # Calculate AUC for each day
    auc = gf.auc(per_day=True, time_unit='h')

    # Create figure
    fig = go.Figure()

    # Create list of days in case one day fails to fit
    list_of_days = list(gf.data['Day'].unique())

    # Loop over days
    show_first = True
    day_count = 0
    for day, day_data in day_groups:
        # Break if num_days is reached
        if day_count > num_days:
            list_of_days = list_of_days[:num_days]
            break

        # Get time as numbers
        num_time = (day_data['Timestamp'] - day_data['Timestamp'].min()).dt.total_seconds() / 3600

        hour_range = num_time.max() - num_time.min()

        # Initial guess for the parameters of the Fourier series (m, amplitude, phase)
        initial_guess = [auc[day]/hour_range] + [hour_range] + amplitude_guess + phase_guess

        # Get best-fit parameters
        try:
            popt, _ = curve_fit(f=fourier_series, 
                                xdata=num_time, 
                                ydata=day_data['CGM'], 
                                p0=initial_guess)
        except Exception as e:
            print(f'Error fitting day {day}: {e}')
            list_of_days.remove(day)
            continue

        # Create the fitted curve
        F = fourier_series(num_time, 
                           popt[0],   # m
                           popt[1],   # hour_range
                           *popt[2:]) # amplitudes and phases

        # Calculate the residuals
        residuals = day_data['CGM'] - F

        # Add the curves
        fig.add_trace(go.Scatter(x=day_data['Time'], 
                                 y=day_data['CGM'],
                                 name='Original Data',
                                 visible=show_first))
        fig.add_trace(go.Scatter(x=day_data['Time'],
                                 y=F,
                                 name='Fourier Fit',
                                 visible=show_first))
        fig.add_trace(go.Scatter(x=day_data['Time'],
                                 y=residuals,
                                 name='Residuals',
                                 visible=show_first))
        
        if show_first:
            show_first = False
        
        # Update day count
        day_count += 1

    # Update layout
    fig.update_layout(title='Fourier Transformation',
                      height=height, 
                      width=width,
                      xaxis_title='Time',
                      yaxis_title='Glucose',
                      )
    
    # Update axes
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
    
    # Add Dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[{"visible": [i==j//3 for j in range(len(day_groups)*3)],
                                "title": f'Fourier Transformation {day}'}],
                        label=str(day),
                        method="update"
                    ) for i, day in enumerate(list_of_days)
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

def fourier_series(t, m, hour_range, *args):
    N = len(args) // 2
    amplitudes = args[:N]
    phases = args[N:]
    result = m
    for i in range(N):
        A_i = amplitudes[i]
        phase_i = phases[i]
        result += A_i * np.cos(2 * np.pi * (i + 1) * (t - phase_i) / hour_range)
    return result


    
    