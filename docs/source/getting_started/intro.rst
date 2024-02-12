====================
Intro to ``GlucoPy``
====================
.. currentmodule:: glucopy

Using ``GlucoPy`` is simple. The first step is to import the package.

.. code-block:: python

    import glucopy as gp

Then, you need to import `CGM` data.

Importing data
~~~~~~~~~~~~~~

- Using the :func:`read_csv` function:

.. code-block:: python

    gf = gp.read_csv('path_to_file.csv')

- Using the :func:`read_excel` function:

.. code-block:: python

    gf = gp.read_excel('path_to_file.xlsx')

- Using a :py:class:`pandas.DataFrame`:

.. code-block:: python

    import pandas as pd
    df = pd.read_csv('path_to_file.csv')
    gf = gp.Gframe(df)

- If you have no `CGM` dataset, you can still try this toolbox using the sample data provided with the package in
  the :func:`data` function

.. code-block:: python

    gf = gp.data()

All the previous functions return an instance of :class:`Gframe` which is the main object of the package. This object
contains the `CGM` data and provides a set of methods to manipulate and analyze it.

Calculating metrics
~~~~~~~~~~~~~~~~~~~

Basic metrics
-------------
The :class:`Gframe` object provides a set of methods to calculate metrics from the `CGM` data. For example, simple metrics like
the mean, standard deviation, and coefficient of variation can be calculated using like this:

.. ipython:: python

    import glucopy as gp
    gf = gp.data()
    mean = gf.mean()
    std = gf.std()
    cv = gf.cv()
    mean, std, cv

Calculating metrics for each day
--------------------------------
Most of the methods provided by :class:`Gframe` can be calculated for each day of the dataset. For example, to calculate the
mean for each day:

.. ipython:: python

    day_means = gf.mean(per_day=True)
    day_means

And since the returned value when `per_day` is set to `True` is a :py:class:`pandas.Series`, you can use any method provided
by the :py:class:`pandas.Series` object. For example, to calculate the max of the means:

.. ipython:: python

    day_means.max()

You can alos access the mean of a specific day using the index of the :py:class:`pandas.Series`:

.. ipython:: python

    day_means["2020-12-01"]

Glycaemia specific metrics
--------------------------
:class:`Gframe` also provides methods to calculate metrics specific to glycaemia. For example, to calculate the
Time in Range (TIR) for a given range:

.. ipython:: python

    tir = gf.tir() # default range is [70, 180]
    tir

Calculating the Mean of Daily Differences (MODD)

.. ipython:: python

    modd = gf.modd()
    modd

Calculating the Glucose Variability Percentage (GVP)

.. ipython:: python

    gvp = gf.gvp()
    gvp

You can check the documentation of :doc:`../glucopy.Gframe` to see the full list of methods provided.

Summary
-------
You can also calculate a summary of the dataset using the :meth:`Gframe.summary` method:

.. ipython:: python

    summary = gf.summary()
    summary

Plotting
~~~~~~~~
``GlucoPy`` uses `Plotly <https://plotly.com/python/>`_ to create interactive plots. The module :doc:`../glucopy.plot` provides a set of functions to create different
types of plots. For example, to create a plot of the `CGM` data:

.. ipython:: python

    gp.plot.per_day(gf,num_days=7)

.. image:: ../../img/per_day_plot_2.png
    :alt: per_day_plot
    :align: center

Or a plot that highlighs the `TIR`:

.. ipython:: python

    gp.plot.tir(gf)

.. image:: ../../img/tir_plot_1.png
    :alt: tir_plot
    :align: center

