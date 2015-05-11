import pymvg.camera_model
import numpy as np

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

    A = np.array(_A, dtype=np.float64)
    assert A.shape == (2*N, 12)
    return A


def get_normalize_2d_matrix(points_2d):
    """normalize 2d points to mean 0 and rms sqrt(2)"""

    pts_mean = points_2d.mean(axis=0)
    centered_pts_2d = points_2d - pts_mean
    s = 1 / np.linalg.norm(centered_pts_2d, axis=1).mean()
    xm, ym = pts_mean
    T = np.array([[s, 0, -s*xm],
                     [0, s, -s*ym],
                     [0, 0,  1 ]], dtype=np.float64)
    return T


def get_normalize_3d_matrix(points_3d):
    """normalize 3d points to mean 0 and rms sqrt(2)"""

    pts_mean = points_3d.mean(axis=0)
    centered_pts_3d = points_3d - pts_mean
    s = 1 / np.linalg.norm(centered_pts_3d, axis=1).mean()
    xm, ym, zm = pts_mean
    U = np.array([[s, 0, 0, -s*xm],
                     [0, s, 0, -s*ym],
                     [0, 0, s, -s*zm],
                     [0, 0, 0,  1 ]], dtype=np.float64)
    return U


def get_homogeneous_coordinates(points):
    """return points in homogeneous coordinates"""

    assert points.ndim == 2
    assert points.shape[1] in [2, 3]
    if points.shape[1] == 3:
        assert not np.allclose(points[:,2], 1.)
    return np.hstack((points, np.ones((points.shape[0], 1))))


def DLT(X3d, x2d, width=640, height=480):
    """Given 3D coordinates X3d and 2d coordinates x2d, find camera model"""

    # Note: may want to implement Hatze (1988). This returns
    # orthogonal rotation matrix. See notes and implementation in
    # https://github.com/NatPRoach/HawkMothCode/blob/master/ManualExtractionCode/DLTcal5.m

    # Implementation of DLT algorithm from
    # "Hartley & Zisserman - Multiple View Geometry in computer vision - 2nd Edition"

    # Normalize 2d points and keep transformation matrix
    T = get_normalize_2d_matrix(x2d)
    Tinv = np.linalg.inv(T)
    hom_points_2d = get_homogeneous_coordinates(x2d)
    normalized_points_2d = np.empty(hom_points_2d.shape)
    for i, x in enumerate(hom_points_2d):
        normalized_points_2d[i,:] = np.dot(T, x)

    # Normalize 3d points and keep transformation matrix
    U = get_normalize_3d_matrix(X3d)
    hom_points_3d = get_homogeneous_coordinates(X3d)
    normalized_points_3d = np.empty(hom_points_3d.shape)
    for i, x in enumerate(hom_points_3d):
        normalized_points_3d[i,:] = np.dot(U, x)

    # get matrix A
    A = create_matrix_A(normalized_points_3d, normalized_points_2d)

    # solve via singular value decomposition
    _, singular_values, VT = np.linalg.svd(A, full_matrices=False)
    sol_idx = np.argmin(singular_values)
    assert sol_idx == 11
    Pvec_n = VT.T[:,sol_idx]  # that's why we need to pick the rows here...

    P_n = Pvec_n.reshape((3, 4))

    # Denormalize
    P = np.dot(Tinv, np.dot(P_n, U))

    cam = pymvg.camera_model.CameraModel.load_camera_from_M(P,width=width,height=height)
    x2d_reproj = cam.project_3d_to_pixel(X3d)
    dx = x2d_reproj - np.array(x2d)
    reproj_error = np.sqrt(np.sum(dx**2,axis=1))
    mean_reproj_error = np.mean(reproj_error)
    results = {'cam':cam,
               'mean_reproj_error':mean_reproj_error,
               }
    return results
