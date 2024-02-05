# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Add project root to path
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))


project = 'Glucopy'
copyright = '2024, Diego Soto Castillo'
author = 'Diego Soto Castillo'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',  # Enables support for NumPy-style docstrings
              'sphinx.ext.viewcode',  # Add links to highlighted source code
              'sphinx.ext.autosummary',
              'IPython.sphinxext.ipython_console_highlighting',
              'IPython.sphinxext.ipython_directive',
              'matplotlib.sphinxext.plot_directive' # Enables plotting in sphinx
              ]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

autosummary_generate = True
navigation_with_keys=False

# Plot configuration
plot_html_show_source_link = False
plot_html_show_formats = False

# HTML Theme
html_theme_options = {
    'navigation_depth': 4,
    'github_url': 'https://github.com/Deigoodle/GlucoPy',
}

