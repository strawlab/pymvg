# camera_model

`camera_model` is a ROS-centric, object-oriented, OpenCV-compatible
model of a calibrated 3D camera. It contains an implementation of a
calibrated camera that produces the same results as OpenCV, yet the
implementation is in Python. This may make it suitable for easier
debugging and learning than delving into the source code of OpenCV.

See also [opengl-hz](https://github.com/strawlab/opengl-hz).

## tests

This package is well tested. To run the tests:

    cd tests
    nosetests
