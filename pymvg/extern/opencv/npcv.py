import numpy as np
import cv2 as cv # ubuntu: sudo apt-get install python3-opencv

# helper functions ---------------

def numpy2opencv_image(arr):
    arr = np.array(arr)
    if arr.ndim==1:
        arr = arr[:,np.newaxis]
    assert arr.ndim==2
    if arr.dtype in [np.float32]:
        result = cv.CreateMat( arr.shape[0], arr.shape[1], cv.CV_32FC1)
    elif arr.dtype in [np.float64, float]:
        result = cv.CreateMat( arr.shape[0], arr.shape[1], cv.CV_64FC1)
    elif arr.dtype in [np.uint8]:
        result = cv.CreateMat( arr.shape[0], arr.shape[1], cv.CV_8UC1)
    else:
        raise ValueError('unknown numpy dtype "%s"'%arr.dtype)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            result[i,j] = arr[i,j]
    return result

def opencv_image2numpy( cvimage ):
    pyobj = np.asarray(cvimage)
    if pyobj.ndim == 2:
        # new OpenCV version
        result = pyobj
    else:
        # old OpenCV, so hack this
        width = cvimage.width
        height = cvimage.height
        assert cvimage.channels == 1
        assert cvimage.nChannels == 1
        assert cvimage.depth == 32
        assert cvimage.origin == 0
        result = np.empty( (height,width), dtype=float )
        for i in range(height):
            for j in range(width):
                result[i,j] = cvimage[i,j]
    return result

def numpy2opencv_pointmat(npdistorted):
    src = cv.CreateMat( npdistorted.shape[0], 1, cv.CV_64FC2)
    for i in range(npdistorted.shape[0]):
        src[i,0] = npdistorted[i,0], npdistorted[i,1]
    return src

def opencv_pointmat2numpy(dst):
    assert dst.width==1
    assert dst.channels == 2
    r = np.empty( (dst.height,2) )
    for i in range(dst.height):
        a,b = dst[i,0]
        r[i,:] = a,b
    return r

# --------------------- internal tests -----------------------------

def test_roundtrip_image():
    orig = np.array( [[100.0,100],
                      [100,200],
                      [100,300],
                      [100,400]] )
    testcv = numpy2opencv_image(orig)
    test = opencv_image2numpy( testcv )
    assert orig.shape==test.shape
    assert np.allclose( orig, test )

def test_roundtrip_pointmat():
    orig = np.array( [[100.0,100],
                      [100,200],
                      [100,300],
                      [100,400]] )
    testcv = numpy2opencv_pointmat(orig)
    test = opencv_pointmat2numpy( testcv )
    assert orig.shape==test.shape
    assert np.allclose( orig, test )

