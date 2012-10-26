#!/usr/bin/env python

import numpy as np

# ROS imports
import roslib; roslib.load_manifest('camera_model')
import geometry_msgs
import sensor_msgs

from camera_model import CameraModel
from camera_model.camera_model import point_msg_to_tuple, parse_rotation_msg
import camera_model

def _build_test_camera(from_pmat=False,at_origin=False,ROS_test_data=False,get_input_data=False,flipped=False):
    if from_pmat:
        d = {'width': 848,
             'name': 'camera',
             'height': 480}
        pmat =  np.array([[ -1.70677031e+03,  -4.10373295e+03,  -3.88568028e+02, 6.89034515e+02],
                          [ -6.19019195e+02,  -1.01292091e+03,  -2.67534989e+03, 4.51847857e+02],
                          [ -4.52548832e+00,  -3.78900498e+00,  -7.35860226e-01, 1.00000000e+00]])
        cam = camera_model.load_camera_from_pmat( pmat, **d)
        if flipped:
            cam = cam.get_flipped_camera()
        if get_input_data:
            return dict(cam=cam)
        return cam
    if not at_origin:
        translation = geometry_msgs.msg.Point()
        translation.x = 0.273485679077
        translation.y = 0.0707310128808
        translation.z = 0.0877802104531

        rotation = geometry_msgs.msg.Quaternion()
        rotation.x = 0.309377331102
        rotation.y = 0.600893485738
        rotation.z = 0.644637681813
        rotation.w = 0.357288321925
    else:
        translation = geometry_msgs.msg.Point()
        translation.x = 0.0
        translation.y = 0.0
        translation.z = 0.0

        rotation = geometry_msgs.msg.Quaternion()
        rotation.x = 0.0
        rotation.y = 0.0
        rotation.z = 0.0
        rotation.w = 1.0

    if ROS_test_data:
        i = sensor_msgs.msg.CameraInfo()
        # these are from image_geometry ROS package in the utest.cpp file
        i.height = 480
        i.width = 640
        i.distortion_model = 'plumb_bob'
        i.D = [-0.363528858080088, 0.16117037733986861, -8.1109585007538829e-05, -0.00044776712298447841, 0.0]
        i.K = [430.15433020105519,                0.0, 311.71339830549732,
                                     0.0, 430.60920415473657, 221.06824942698509,
                                     0.0,                0.0,                1.0]
        i.R = [0.99806560714807102, 0.0068562422224214027, 0.061790256276695904,
                      -0.0067522959054715113, 0.99997541519165112, -0.0018909025066874664,
                      -0.061801701660692349, 0.0014700186639396652, 0.99808736527268516]
        i.P = [295.53402059708782, 0.0, 285.55760765075684, 0.0,
                      0.0, 295.53402059708782, 223.29617881774902, 0.0,
                      0.0, 0.0, 1.0, 0.0]
    else:
        i = sensor_msgs.msg.CameraInfo()
        i.height = 494
        i.width = 659
        i.distortion_model = 'plumb_bob'
        i.D = [-0.34146457767225, 0.196070795764995, 0.000548988393912233, 0.000657058395082583, -0.0828776806503243]
        i.K = [516.881868241566, 0.0, 333.090936517613, 0.0, 517.201263180996, 231.526036849886, 0.0, 0.0, 1.0]
        i.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        i.P = [442.17529296875, 0.0, 334.589001099812, 0.0, 0.0, 474.757141113281, 228.646131377705, 0.0, 0.0, 0.0, 1.0, 0.0]


    cam = CameraModel(translation=point_msg_to_tuple(translation),
                      rotation=parse_rotation_msg(rotation),
                      intrinsics=i,
                      name='cam',
                      )
    if flipped:
        cam = cam.get_flipped_camera()
    if get_input_data:
        return dict(cam=cam,
                    translation=translation,
                    rotation=rotation,
                    )
    return cam

def _build_points_3d():
    n_pts = 100
    x,y,z = 10.001, 20.001, 30.001
    theta = np.linspace(0,10,n_pts)%(2*np.pi)
    h = np.linspace(0,10,n_pts)
    r = 0.05
    pts3D = np.vstack( (r*np.cos(theta)+x, r*np.sin(theta)+y, h+z )).T
    return pts3D

def get_default_options():
    result = []
    for at_origin in (True,False):
        for ROS_test_data in (True,False):
            #for flipped in (True,False):
            for flipped in (False,):
                opts = dict(at_origin=at_origin,ROS_test_data=ROS_test_data,flipped=flipped)
                result.append(opts)
    #result.append( dict(from_pmat=True) )
    return result
