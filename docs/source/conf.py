# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
autodoc_mock_imports = ["pycuda", "pycuda.autoinit.device.get_attribute", "dolphin.cuda_base"]

sys.path.insert(0, os.path.abspath('../../'))

project = 'dolphin'
copyright = '2023, Maxime Debarbat'
author = 'Maxime Debarbat'
about = {}
exec(open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..", "dolphin", 'version.py')).read(), about)
release = about['__version__']
version = release

print(f"Building documentation for Dolphin {release}")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.todo',
              'sphinx.ext.viewcode',
              'sphinx.ext.autodoc',
              'sphinx.ext.intersphinx',
              'sphinx.ext.autosectionlabel']

templates_path = ['_templates']
exclude_patterns = []
add_module_names = True
autodoc_member_order = 'bysource'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_sidebars = {'**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html']}
html_show_sourcelink = True
html_show_copyright = True
html_logo = '_static/banner.png'
html_theme_options = {
    'logo_only': False,
    'display_version': True
}
