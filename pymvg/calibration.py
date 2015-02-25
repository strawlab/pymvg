import numpy
import numpy.linalg
import scipy.optimize
import pymvg.camera_model

def create_matrix_A(hom_points_3d, hom_points_2d):
    """build matrix A for DLT method"""

    assert hom_points_3d.ndim == 2 and hom_points_3d.shape[1] == 4
    assert hom_points_2d.ndim == 2 and hom_points_2d.shape[1] == 3
    assert hom_points_3d.shape[0] == hom_points_2d.shape[0]

    N = hom_points_3d.shape[0]

    _A = []
    for i in range(N):
        X, Y, Z, S = hom_points_3d[i]
        u, v, w = hom_points_2d[i]

        _A.append([  0,   0,   0,   0, -w*X, -w*Y, -w*Z, -w*S,  v*X,  v*Y,  v*Z,  v*S])
        _A.append([w*X, w*Y, w*Z, w*S,    0,    0,    0,    0, -u*X, -u*Y, -u*Z, -u*S])

    A = numpy.array(_A, dtype=numpy.float64)
    assert A.shape == (2*N, 12)
    return A


def get_normalize_2d_matrix(points_2d):
    """normalize 2d points to mean 0 and rms sqrt(2)"""

    pts_mean = points_2d.mean(axis=0)
    centered_pts_2d = points_2d - pts_mean
    s = 1 / numpy.linalg.norm(centered_pts_2d, axis=1).mean()
    xm, ym = pts_mean
    T = numpy.array([[s, 0, -s*xm],
                     [0, s, -s*ym],
                     [0, 0,  1 ]], dtype=numpy.float64)
    return T


def get_normalize_3d_matrix(points_3d):
    """normalize 3d points to mean 0 and rms sqrt(2)"""

    pts_mean = points_3d.mean(axis=0)
    centered_pts_3d = points_3d - pts_mean
    s = 1 / numpy.linalg.norm(centered_pts_3d, axis=1).mean()
    xm, ym, zm = pts_mean
    U = numpy.array([[s, 0, 0, -s*xm],
                     [0, s, 0, -s*ym],
                     [0, 0, s, -s*zm],
                     [0, 0, 0,  1 ]], dtype=numpy.float64)
    return U


def get_homogeneous_coordinates(points):
    """return points in homogeneous coordinates"""

    assert points.ndim == 2
    assert points.shape[1] in [2, 3]
    if points.shape[1] == 3:
        assert not numpy.allclose(points[:,2], 1.)
    return numpy.hstack((points, numpy.ones((points.shape[0], 1))))


def DLT(X3d, x2d, width=640, height=480):
    """Given 3D coordinates X3d and 2d coordinates x2d, find camera model"""

    # Implementation of DLT algorithm from 
    # "Hartley & Zisserman - Multiple View Geometry in computer vision - 2nd Edition"

    # Normalize 2d points and keep transformation matrix
    T = get_normalize_2d_matrix(x2d)
    Tinv = numpy.linalg.inv(T)
    hom_points_2d = get_homogeneous_coordinates(x2d)
    normalized_points_2d = numpy.empty(hom_points_2d.shape)
    for i, x in enumerate(hom_points_2d):
        normalized_points_2d[i,:] = numpy.dot(T, x)

    # Normalize 3d points and keep transformation matrix
    U = get_normalize_3d_matrix(X3d)
    hom_points_3d = get_homogeneous_coordinates(X3d)
    normalized_points_3d = numpy.empty(hom_points_3d.shape)
    for i, x in enumerate(hom_points_3d):
        normalized_points_3d[i,:] = numpy.dot(U, x)

    # get matrix A
    A = create_matrix_A(normalized_points_3d, normalized_points_2d)

    # solve via singular value decomposition
    # XXX: We do not require U, maybe there is a faster way to compute s and V
    # XXX: in numpy the returned V is already transposed!
    _, singular_values, VT = numpy.linalg.svd(A, full_matrices=False)
    sol_idx = numpy.argmin(singular_values)
    assert sol_idx == 11
    Pvec_n = VT.T[:,sol_idx]  # that's why we need to pick the rows here...

    P_n = Pvec_n.reshape((3, 4))

    # Denormalize
    P = numpy.dot(Tinv, numpy.dot(P_n, U))

    print P
    cam = pymvg.camera_model.CameraModel.load_camera_from_M(P,width=width,height=height)
    x2d_reproj = cam.project_3d_to_pixel(X3d)
    dx = x2d_reproj - numpy.array(x2d)
    reproj_error = numpy.sqrt(numpy.sum(dx**2,axis=1))
    mean_reproj_error = numpy.mean(reproj_error)
    results = {'cam':cam,
               'mean_reproj_error':mean_reproj_error,
               }
    return results
