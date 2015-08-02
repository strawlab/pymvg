# pymvg - Python Multi-View Geometry

PyMVG is a Python implementation of various computational camera
geometry operations.

## Documentation

Documentation is available [here](http://pymvg.readthedocs.org/).

## tests

[![Build Status](https://travis-ci.org/strawlab/pymvg.png?branch=master)](https://travis-ci.org/strawlab/pymvg)

[![Coverage Status](https://coveralls.io/repos/strawlab/pymvg/badge.png?branch=master)](https://coveralls.io/r/strawlab/pymvg?branch=master)

PyMVG has a large collection of unit tests which ensure correctness
and fulfilled expectations for use other software (see 'Ecosystem' in
the documentation). To run the tests:

    nosetests -w pymvg/test

To run the tests with external dependencies:

    nosetests -w pymvg/test/external/opencv
    nosetests -w pymvg/test/external/ros
    nosetests -w pymvg/test/external/mcsc

## TODO

- Implement extrinsic camera calibration to find camera pose when intrinsic parameters are known and image coordinates of known 3D points are given
- Fix known failing test pymvg.test.external.mcsc.test_mcsc.test_mcsc_roundtrip
- Implement OpenGL 3D -> 2d transform for augmented reality applications
