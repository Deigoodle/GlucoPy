======
Gframe
======
.. currentmodule:: glucopy

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Gframe

Utilities
~~~~~~~~~
.. autosummary::
   :toctree: api/

   Gframe.convert_unit

Summary
~~~~~~~
.. autosummary::
   :toctree: api/

   Gframe.summary

Metrics
~~~~~~~

1. Joint data analysis metrics for glycaemia dynamics
-----------------------------------------------------

.. autosummary::
   :toctree: api/

   Gframe.mean
   Gframe.std
   Gframe.cv
   Gframe.pcv
   Gframe.quantile
   Gframe.iqr
   Gframe.modd
   Gframe.tir

2. Analysis of distribution in the plane for glycaemia dynamics
---------------------------------------------------------------

.. autosummary::
   :toctree: api/

   Gframe.fd
   Gframe.auc

3. Amplitude and distribution of frequencies metrics for glycaemia dynamics
---------------------------------------------------------------------------

.. autosummary::
   :toctree: api/

   Gframe.mage
   Gframe.dt

4. Metrics for the analysis of glycaemic dynamics using scores of glucose values
--------------------------------------------------------------------------------

.. autosummary::
   :toctree: api/

   Gframe.bgi
   Gframe.hbgi
   Gframe.lbgi
   Gframe.adrr
   Gframe.grade
   Gframe.qscore

5. Metrics for the analysis of glycaemic dynamics using variability estimation
------------------------------------------------------------------------------

.. autosummary::
   :toctree: api/

   Gframe.mard
   Gframe.conga
   Gframe.gvp
   Gframe.mag

6. Computational methods for the analysis of glycemic dynamics
--------------------------------------------------------------

.. autosummary::
   :toctree: api/

   Gframe.dfa
   Gframe.samp_en
   Gframe.mse