=======
Metrics
=======
.. currentmodule:: glucopy

This module contains functions for calculating metrics of a glucose time series. The functions are meant to be used
by the :class:`glucopy.Gframe` class, but can be used independently as well as long as the input has the next columns:

- 'Timestamp' : datetime64[ns]  # pandas datetime
- 'Day' : datetime.date 
- 'Time' : datetime.time
- 'CGM' : number 

1. Joint data analysis metrics for glycaemia dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   metrics.mean
   metrics.std
   metrics.cv
   metrics.quantile
   metrics.iqr
   metrics.modd
   metrics.tir

2. Analysis of distribution in the plane for glycaemia dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   metrics.fd
   metrics.auc

3. Amplitude and distribution of frequencies metrics for glycaemia dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   metrics.mage
   metrics.dt

4. Metrics for the analysis of glycaemic dynamics using scores of glucose values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   metrics.bgi
   metrics.grade

5. Metrics for the analysis of glycaemic dynamics using variability estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   metrics.conga
   metrics.gvp
   metrics.mag

6. Computational methods for the analysis of glycemic dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   metrics.dfa
   metrics.samp_en
   metrics.mse