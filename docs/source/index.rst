PyMVG documentation
===================

`PyMVG <http://pymvg.readthedocs.org/>`_ is a Python implementation of
various computational camera geometry operations.

Features:

- triangulate 2D features from multiple calibrated cameras into a
  single 3D point (using algorithm from `the classic textbook by
  Hartley & Zisserman
  <http://www.amazon.com/Multiple-View-Geometry-Computer-Vision/dp/0521540518>`_).
- load camera calibrations from `ROS <http://ros.org>`_ (which uses
  `OpenCV <http://opencv.org>`_)
- load camera system calibrations from `MultiCamSelfCal
  <https://github.com/strawlab/MultiCamSelfCal>`_
- complete implementation of OpenCV camera model in pure Python in a
  single file for easy understanding
- complete implementation of DLT camera calibration procedure
- completely vectorized code for rapid operation on many points using
  `numpy <http://numpy.org>`_
- completely written in `Python <http://python.org>`_

It contains a complete re-implementation of the OpenCV camera model
and can thus use calibrations made by or for OpenCV. PyMVG is entirely
written in Python, and thus -- depending on your preferences -- it may
be significantly easier to understand than the equivalent OpenCV
implementation. PyMVG makes extensive use of `numpy
<http://numpy.org>`_, and thus when called on large batches of points,
is no slower than native code.

Ecosystem
---------

PyMVG is designed to interoperate with `OpenCV <http://opencv.org>`_,
`ROS <http://ros.org>`_, and `MultiCamSelfCal
<https://github.com/strawlab/MultiCamSelfCal>`_. Unit tests ensure
exact compatibility the relevant parts of these packages.

See also `opengl-hz <https://github.com/strawlab/opengl-hz>`_.

Development
-----------

All development is done on `our github repository
<https://github.com/strawlab/pymvg>`_.

.. toctree::

  pymvg_file_format
  plotting_utilities
  camera_model
  api_reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

