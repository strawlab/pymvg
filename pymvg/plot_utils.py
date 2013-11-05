import numpy as np

import matplotlib.pyplot as plt

def plot_camera( ax, cam, scale=0.2, show_upper_left=False, axes_size=0.2):
    C = cam.get_camcenter()
    C.shape=(3,)
    ax.plot( [C[0]], [C[1]], [C[2]], 'ko', ms=5 )

    world_coords = cam.project_camera_frame_to_3d( [[axes_size,0,0],
                                                    [0,axes_size,0],
                                                    [0,0,axes_size]])

    for i in range(3):
        c = 'rgb'[i]
        vv = world_coords[i]
        v = np.vstack( ([C],[vv]) )
        ax.plot( v[:,0], v[:,1], v[:,2], c+'-' )

    if cam.width is None or cam.height is None:
        raise ValueError('Camera width/height must be defined to plot.')

    uv_raw = np.array([[0,0],
                       [0,cam.height],
                       [cam.width, cam.height],
                       [cam.width, 0],
                       [0,0]])
    pts3d_near = cam.project_pixel_to_3d_ray( uv_raw, distorted=True, distance=0.1*scale)
    pts3d_far = cam.project_pixel_to_3d_ray( uv_raw, distorted=True, distance=scale)
    # ring at near depth
    ax.plot( pts3d_near[:,0], pts3d_near[:,1], pts3d_near[:,2], 'k-' )
    # ring at far depth
    ax.plot( pts3d_far[:,0], pts3d_far[:,1], pts3d_far[:,2], 'k-' )
    # connectors
    for i in range(len(pts3d_near)-1):
        pts3d = np.vstack((pts3d_near[i,:],pts3d_far[i,:]))
        ax.plot( pts3d[:,0], pts3d[:,1], pts3d[:,2], 'k-' )

    ax.text( C[0], C[1], C[2], cam.name ) # fails unless using mplot3d
    if show_upper_left:
        ax.text( pts3d_far[0,0], pts3d_far[0,1], pts3d_far[0,2], 'UL' )

def plot_system( ax, system, **kwargs):
    for name, cam in system.get_camera_dict().iteritems():
        plot_camera( ax, cam, **kwargs)
