.. PyMVG documentation master file, created by
   sphinx-quickstart on Sat Nov  2 17:43:20 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyMVG's documentation!
=================================

PyMVG file format
=================

Perhaps the best starting point for PyMVG is its file format. Such a
file specifies a camera system completely. The file is valid
JSON. Here is an example:

.. literalinclude:: pymvg_camsystem_example.json

Plotting utilities
==================

Given the above example, we can plot the camera system.

.. plot:: pyplots/plot_camsystem_example.py
   :include-source:

Contents:

.. toctree::
   :maxdepth: 2

.. figure:: images/camera_model.png

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

