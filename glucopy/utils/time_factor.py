def time_factor(t: str) -> int:
    '''
    Return the time factor of a given time string.

    Parameters
    ----------
    t : str
        The time string to be parsed, can be one of the following:
        - 's': second
        - 'm': minute
        - 'h': hour

    Returns
    -------
    int
        The time factor of the given time string.

    Examples
    --------
    Convert a time string to time factor

    .. ipython:: python

        import glucopy as gp
        gp.time_factor('s')
        gp.time_factor('m')
        gp.time_factor('h')
    '''
    t = t.lower()

    if t == 's':
        return 1
    elif t == 'm':
        return 60
    elif t == 'h':
        return 3600
    else:
        raise ValueError('Invalid time string: {}'.format(t))