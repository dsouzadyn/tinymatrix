"""Sphinx configuration for TinyMatrix documentation."""

project = "TinyMatrix"
copyright = "2026, Dylan Dsouza"
author = "Dylan Dsouza"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "TinyMatrix Documentation"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}
