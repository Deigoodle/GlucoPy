=======
Metrics
=======
.. currentmodule:: glucopy

This module contains functions for calculating metrics of a glucose time series. The functions are meant to be used
by the :class:`glucopy.Gframe` class, but can be used independently as well as long as the input has the next columns:

- 'Timestamp' : datetime 
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


