#!/usr/bin/env python
import numpy as np
import argparse
import json
import threading

import roslib; roslib.load_manifest('flyvr')
roslib.load_manifest('visualization_msgs')
import rospy

import tf.transformations
from sensor_msgs.msg import CameraInfo
import sensor_msgs.msg
import std_msgs.msg
from geometry_msgs.msg import Point, Pose, Transform
import geometry_msgs.msg
import tf.broadcaster
import tf.msg
from visualization_msgs.msg import Marker, MarkerArray

import flyvr.simple_geom as simple_geom
import flyvr.display_client as display_client
from pymvg.camera_model import CameraModel
import pymvg.extern.ros.rviz_utils as rviz_utils

class MyApp:
    def __init__(self,name,scale=1.0):
        self.name = name
        self.scale = scale
        self.intrinsics = None
        self.translation = None
        self.rotation = None
        self._lock = threading.Lock()
        self.tl = tf.TransformListener()
        self.cam = None

        ci_name = self.get_frame_id()+'/camera_info'
        rospy.loginfo('now listening for CameraInfo message on topic %r'%ci_name)

        rospy.Subscriber(ci_name,
                         CameraInfo, self.on_camera_info)

        self.topic_name = self.get_frame_id()+'/frustum'
        rospy.loginfo('publishing frustum (scale %s) at %r'%(self.scale,
                                                             self.topic_name))
        self.publisher = rospy.Publisher(self.topic_name, MarkerArray)

        rospy.loginfo('now listening for transform at %r'%self.get_frame_id())
        rospy.Timer(rospy.Duration(1.0/20.0), self.on_timer) # 20 fps

    def get_frame_id(self):
        return '/'+self.name

    def on_camera_info(self, msg):
        with self._lock:
            self.intrinsics = msg

    def on_timer(self, _):
        now = rospy.Time.now()
        try:
            translation,rotation = self.tl.lookupTransform('/map',
                                                           self.get_frame_id(),
                                                           now)
        except (tf.LookupException, tf.ExtrapolationException) as err:
            return

        with self._lock:
            self.translation = translation
            self.rotation = rotation

        self.new_data()

    def new_data(self):
        with self._lock:
            if (self.translation is None or
                self.rotation is None or
                self.intrinsics is None):
                return
            newcam = CameraModel.load_camera_from_ROS_tf( translation=self.translation,
                                                          rotation=self.rotation,
                                                          intrinsics=self.intrinsics,
                                                          name=self.get_frame_id(),
                                                          )
        self.cam = newcam

        self.draw()

    def draw(self):
        r = rviz_utils.get_frustum_markers( self.cam, scale=self.scale )
        self.publisher.publish(r['markers'])

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('draw_rviz_frustum',anonymous=True)

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='cam_0')
    parser.add_argument('--scale', type=float, default=1.0)

    argv = rospy.myargv()
    args = parser.parse_args(argv[1:])

    app = MyApp(args.name,scale=args.scale)
    app.run()
