import importlib.metadata

project = "numpygrad"
copyright = "2026, Nick Richardson"
author = "Nick Richardson"
release = importlib.metadata.version("numpygrad")

extensions: list[str] = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
