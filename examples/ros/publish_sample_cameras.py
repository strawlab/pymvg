import numpy as np

from pymvg.multi_camera_system import build_example_system
from pymvg.extern.ros.ros_publisher import ROSPublisher

import roslib # ROS is required for this example. See http://ros.org
roslib.load_manifest('rospy')
import rospy

cam_pubs = []

rospy.init_node("publish_sample_cameras")

n=6
z=5.0
system = build_example_system(n=n,z=z)

for name in system.get_names():
    cam_pubs.append(ROSPublisher(system.get_camera(name)))

rospy.spin()
