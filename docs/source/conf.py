# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MOFTransformer"
copyright = "2022, Yeonghun Kang, Hyunsoo Park"
author = "Yeonghun Kang, Hyunsoo Park"
release = "2.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # read markdown
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv"]  # setting from pycharm

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
# html_static_path = ["_static"]

pygments_style = "sphinx"
pygments_dark_style = "monokai"

html_theme_options = {
    # announcement
    # "announcement": "<em>Important</em> announcement!",
    # Adding an edit button
    "source_repository": "https://github.com/hspark1212/MOFTransformer/",
    "source_branch": "main",
    "source_directory": "docs/",
    # color
    "light_css_variables": {
        "color-brand-primary": "#4970E0",
        "color-brand-content": "#4970E0",
    },
}

html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}
