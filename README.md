# pymvg - Python Multi-View Geometry

PyMVG is a Python implementation of various computational camera
geometry operations.

## Documentation

Documentation is available [here](http://pymvg.readthedocs.org/).

## tests

[![Build Status](https://travis-ci.org/strawlab/pymvg.png)](https://travis-ci.org/strawlab/pymvg)

PyMVG has a large collection of unit tests which ensure correctness
and fulfilled expectationss for use other software (see 'Ecosystem',
above). To run the tests:

    nosetests -w pymvg/test

To run the tests with external dependencies:

    nosetests -w pymvg/test/external/opencv
    nosetests -w pymvg/test/external/ros
    nosetests -w pymvg/test/external/mcsc

## TODO

- Implement extrinsic camera calibration to find camera pose when intrinsic parameters are known and image coordinates of known 3D points are given
- Fix known failing test pymvg.test.external.mcsc.test_mcsc.test_mcsc_roundtrip
- Implement OpenGL 3D -> 2d transform for augmented reality applications
