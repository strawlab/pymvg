from pymvg.camera_model import CameraModel
import numpy as np

import roslib # ROS is required for this example. See http://ros.org
roslib.load_manifest('tf')

import rospy
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import CameraInfo
import tf.broadcaster

UPDATES_PER_SECOND=4

class CamPub:
    def __init__(self,cam):
        self.cam = cam
        self.tf_b = tf.broadcaster.TransformBroadcaster()

        self.frame_id = '/'+cam.name
        rospy.loginfo('sending example data for camera %r'%self.frame_id)

        self.intr_pub = rospy.Publisher(self.frame_id+'/camera_info',
                                        CameraInfo, latch=True)

        rospy.Timer(rospy.Duration(1.0/UPDATES_PER_SECOND), self.on_timer) # 20 fps

    def on_timer(self,_):
        now = rospy.Time.now()
        future = now + rospy.Duration(1.0/UPDATES_PER_SECOND)

        # publish camera intrinsics
        intrinsics = self.cam.get_intrinsics_as_bunch()
        intrinsic_msg = CameraInfo(**intrinsics.__dict__)
        intrinsic_msg.header.stamp = now
        intrinsic_msg.header.frame_id = self.frame_id
        self.intr_pub.publish( intrinsic_msg )

        # publish camera transform
        translation, rotation = self.cam.get_ROS_tf()
        self.tf_b.sendTransform( translation,
                                 rotation,
                                 future,
                                 self.frame_id,
                                 '/map',
                                 )

base = CameraModel.load_camera_default()

n=6
x = np.linspace(0, 2*n, n)
theta = np.linspace(0, 2*np.pi, n)
dim = 5.0
cam_pubs = []

rospy.init_node("show_cameras_in_ros_rviz")

for i in range(n):
    center = np.array( (x[i], 0.0, dim) )
    lookat = center + np.array( (0,1,0))
    up = -np.sin(theta[i]), 0, np.cos(theta[i])

    cam = base.get_view_camera(center,lookat,up)
    cam.name = 'cam_%d'%i
    cam_pubs.append(CamPub(cam))

rospy.spin()
