"""ros_publisher - publish camera information to ROS"""
import roslib # ROS is required for this module. See http://ros.org
roslib.load_manifest('tf')

import rospy
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import CameraInfo
import tf.broadcaster

class ROSPublisher:
    def __init__(self,cam,updates_per_second=4.0):
        self.cam = cam
        self.updates_per_second=updates_per_second
        self.tf_b = tf.broadcaster.TransformBroadcaster()

        self.frame_id = '/'+cam.name.replace(' ','_').replace(':','_')
        rospy.loginfo('sending example data for camera %r'%self.frame_id)

        self.intr_pub = rospy.Publisher(self.frame_id+'/camera_info',
                                        CameraInfo, latch=True)

        rospy.Timer(rospy.Duration(1.0/self.updates_per_second), self.on_timer) # 20 fps

    def on_timer(self,_):
        now = rospy.Time.now()
        future = now + rospy.Duration(1.0/self.updates_per_second)

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
