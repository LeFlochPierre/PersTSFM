"""Reusable, leakage-controlled time-series foundation-model benchmark core.

The package imports only numpy/pandas in its data and evaluation modules. FEV
and GPU/model dependencies are lazy imports, used only by their dedicated commands.
"""
__version__ = "0.1.0"
