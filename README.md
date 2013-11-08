# pymvg - Python Multi-View Geometry

[PyMVG](https://github.com/strawlab/pymvg) is a Python implementation
of various computational camera geometry operations.

Features:

- triangulate 2D features from multiple calibrated cameras into a single 3D point
- load camera calibrations from [ROS](http://ros.org) (which uses [OpenCV](http://opencv.org))
- load camera system calibrations from [MultiCamSelfCal](https://github.com/strawlab/MultiCamSelfCal)
- complete implementation of OpenCV camera model in pure Python in a single file for easy understanding
- complete implementation of DLT camera calibration procedure
- completely vectorized code for rapid operation on many points using [numpy](http://numpy.org)
- completely written in [Python](http://python.org)

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

    nosetests -w pymvg/test

## TODO

- Implement extrinsic camera calibration to find camera pose when intrinsic parameters are known and image coordinates of known 3D points are given
- Fix known failing test pymvg.test.external.mcsc.test_mcsc.test_mcsc_roundtrip
- Implement OpenGL 3D -> 2d transform for augmented reality applications
