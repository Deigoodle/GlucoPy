# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Add project root to path
import os
import sys
import inspect

sys.path.insert(0, os.path.abspath('../..'))

project = 'GlucoPy'
copyright = '2024, Diego Soto Castillo'
author = 'Diego Soto Castillo'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',  # Enables support for NumPy-style docstrings
              #'sphinx.ext.viewcode',  # Add links to highlighted source code
              'sphinx.ext.linkcode', # Add links to GitHub source code
              'sphinx.ext.autosummary',
              'IPython.sphinxext.ipython_console_highlighting',
              'IPython.sphinxext.ipython_directive',
              'matplotlib.sphinxext.plot_directive', # Enables plotting in sphinx
              'sphinx_copybutton',  # Add copy button to code blocks
              'sphinx.ext.intersphinx',  # Link to other projects' documentation
              ]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

autosummary_generate = True
navigation_with_keys = False

# Plot configuration
plot_html_show_source_link = False
plot_html_show_formats = False

# HTML Theme
html_theme_options = {
    'navigation_depth': 4,
    'github_url': 'https://github.com/Deigoodle/GlucoPy',
}
html_css_files = [
    'custom.css',
]

# This will remove the 'In [1]:' like prompts from copied text
copybutton_prompt_text = r"In \[\d*\]: |\.\.\.: |\$ "
copybutton_prompt_is_regexp = True

# Inter-sphinx configuration
intersphinx_mapping = {
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'plotly': ('https://plotly.com/python-api-reference', None),
    'neurokit2': ('https://neuropsychology.github.io/NeuroKit/', None),
}

# Link to the source code
import inspect
import os
import sys

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    module = sys.modules[info['module']]
    fullname = info['fullname']
    obj = module
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
    relpath = os.path.relpath(inspect.getsourcefile(obj))
    lineno = inspect.getsourcelines(obj)[1]
    return f"https://github.com/deigoodle/GlucoPy/tree/main/glucopy/{relpath}#L{lineno}"
    


