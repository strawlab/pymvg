import pymvg
import numpy as np

def build_Bc(X3d,x2d):
    """build B and c matrices for DLT method"""
    B = []
    c = []

    assert len(X3d)==len(x2d)
    if len(X3d) < 6:
        raise ValueError('2 equations and 11 unknowns means we need 6 points!')
    for i in range(len(X3d)):
        X = X3d[i,0]
        Y = X3d[i,1]
        Z = X3d[i,2]
        x = x2d[i,0]
        y = x2d[i,1]

        B.append( [X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z] )
        B.append( [0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z] )

        c.append( x )
        c.append( y )
    return np.array(B), np.array(c)

def DLT(X3d, x2d, width=640, height=480):
    """Given 3D coordinates X3d and 2d coordinates x2d, find camera model"""

    # There are improvements that could be made. For further reading, see:
    # http://kwon3d.com/theory/calib.html
    # http://users.rsise.anu.edu.au/~hartley/Papers/algebraic/ICCV/final/algebraic.pdf

    B,c = build_Bc(X3d,x2d)
    DLT_avec_results = np.linalg.lstsq(B,c)
    a_vec,residuals = DLT_avec_results[:2]
    Mhat = np.array(list(a_vec)+[1])
    Mhat.shape=(3,4)
    cam = pymvg.CameraModel.load_camera_from_M(Mhat,width=width,height=height)
    results = {'cam':cam,
               'residuals':residuals,
               }
    return results
