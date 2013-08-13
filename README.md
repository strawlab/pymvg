# pymvg - Python Multi-View Geometry

[PyMVG](https://github.com/strawlab/pymvg) is a Python implementation
of various computational camera geometry operations. In particular, it
allows triangulation of 2D features from multiple calibrated cameras
into a single 3D point.

It contains a complete re-implementation of the OpenCV camera model
and can thus use calibrations made by or for OpenCV. PyMVG is entirely
written in Python, and thus -- depending on your preferences -- it may
be significantly easier to understand than the equivalent OpenCV
implementation. PyMVG makes extensive use of
[numpy](http://numpy.org), and thus when called on large batches of
points, is no slower than native code.

## Ecosystem

PyMVG is designed to interoperate with [OpenCV](http://opencv.org),
[ROS](http://ros.org), and
[MultiCamSelfCal](https://github.com/strawlab/MultiCamSelfCal). Unit
tests ensure exact compatibility the relevant parts of these packages.

See also [opengl-hz](https://github.com/strawlab/opengl-hz).

## tests

PyMVG has a large collection of unit tests which ensure correctness
and fulfilled expectationss for use other software (see 'Ecosystem',
above).

This package is well tested. To run the tests:

    cd test
    nosetests
