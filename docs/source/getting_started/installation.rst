============
Installation
============
Currently ``glucopy`` is only available from source 

Dependencies
~~~~~~~~~~~~

Required dependencies
---------------------

- `Python`_ 3.11 or later
- `NumPy`_ 1.26.2 or later
- `Pandas`_ 2.1.4 or later
- `Plotly`_ 5.18.0 or later
- `Scipy`_ 1.11.4 or later
- `Astropy`_ 6.0.0 or later
- `NeuroKit2`_ 0.2.7 or later
- `Request`_ 2.31.0 or later

Optional dependencies
---------------------

Excel files
***********

- `openpyxl`_ 3.1.2 or later
- `xlrd`_ 2.0.1 or later


Installing from source
~~~~~~~~~~~~~~~~~~~~~~

Obtaining the source package
----------------------------

The latest development version of ``glucopy`` can be cloned from GitHub using the following command:

.. code-block:: shell

    git clone https://github.com/Deigoodle/GlucoPy.git

Building and installing
-----------------------

First, navigate to the directory containing the source code.

.. code-block:: shell

    cd GlucoPy

Then, run the following command to install the package:

.. code-block:: shell

    pip install .

If you want to make changes to the code, you can install the package in editable mode:

.. code-block:: shell

    pip install -e .

.. Dependencies
.. _Python: https://www.python.org/
.. _NumPy: https://numpy.org/
.. _Pandas: https://pandas.pydata.org/
.. _Plotly: https://plotly.com/python/
.. _Scipy: https://www.scipy.org/
.. _Astropy: https://www.astropy.org/
.. _NeuroKit2: https://neuropsychology.github.io/NeuroKit/
.. _Request: https://docs.python-requests.org/en/master/

.. Optional dependencies
.. _openpyxl: https://openpyxl.readthedocs.io/en/stable/
.. _xlrd: https://xlrd.readthedocs.io/en/latest/