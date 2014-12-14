import numpy as np

import roslib
roslib.load_manifest('visualization_msgs')
roslib.load_manifest('geometry_msgs')

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

def v3(arr):
    assert arr.ndim==1
    assert arr.shape[0]==3
    return Point(x=arr[0],y=arr[1],z=arr[2])

def get_frustum_markers(cam, id_base=0, scale=1.0,
                        frame_id = '/map', stamp=None):
    uv_raw = np.array([[0,0],
                       [0,cam.height],
                       [cam.width, cam.height],
                       [cam.width, 0],
                       [0,0]])
    pts3d_near = cam.project_pixel_to_3d_ray( uv_raw, distorted=True, distance=0.1*scale)
    pts3d_far = cam.project_pixel_to_3d_ray( uv_raw, distorted=True, distance=scale)

    markers = []

    # ring at near depth
    NEAR_RING=0+id_base
    near_ring = marker = Marker()
    marker.header.frame_id = frame_id
    if stamp is not None:
        marker.header.stamp = stamp
    marker.id = NEAR_RING
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.01*scale; #line width
    marker.color.a = 1.0;
    marker.color.r = 0.6;
    marker.color.g = 1.0;
    marker.color.b = 0.6;
    marker.points = [v3(pts3d_near[i]) for i in range( pts3d_near.shape[0] ) ]
    markers.append(marker)

    # ring at far depth
    FAR_RING = 1
    far_ring = marker = Marker()
    marker.header.frame_id = frame_id
    if stamp is not None:
        marker.header.stamp = stamp
    marker.id = FAR_RING+id_base
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.01*scale; #line width
    marker.color.a = 1.0;
    marker.color.r = 0.6;
    marker.color.g = 1.0;
    marker.color.b = 0.6;
    marker.points = [v3(pts3d_far[i]) for i in range( pts3d_far.shape[0] ) ]
    markers.append(marker)

    # connectors
    for i in range(len(pts3d_near)-1):
        ID = 2+i
        pts3d = np.vstack((pts3d_near[i,:],pts3d_far[i,:]))

        marker = Marker()
        marker.header.frame_id = frame_id
        if stamp is not None:
            marker.header.stamp = stamp
        marker.id = ID+id_base
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.01*scale; #line width
        marker.color.a = 1.0;
        marker.color.r = 0.6;
        marker.color.g = 1.0;
        marker.color.b = 0.6;
        marker.points = [v3(pts3d[i]) for i in range( pts3d.shape[0] ) ]
        markers.append(marker)

    n_ids = marker.id - id_base

    marray = MarkerArray(markers)
    return {'markers':marray, 'n_ids':n_ids}

