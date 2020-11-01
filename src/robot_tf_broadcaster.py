#!/usr/bin/env python
import rospy

import tf
from geometry_msgs.msg import PoseStamped
import numpy as np
import sys

class ArucoTFBroadcastNode:
    """
    This node publish the TF between the aruco frame  and odom by using the pose detected by the overhead camera using
    aruco
    """
    def __init__(self):
        self.tf_broadcast = tf.TransformBroadcaster()
        self.tf_listen = tf.TransformListener()
        odom_frame = rospy.get_param("~odom_frame","odom")
        aruco_frame = rospy.get_param("~aruco_frame","aruco_link")
        self.base_link_name = rospy.get_namespace()[1:]+aruco_frame
        self. odom_name = rospy.get_namespace()[1:]+odom_frame
        aruco_id = rospy.get_param('krabby_aruco_id', 5)
        rospy.Subscriber("/pose_robots/%s" % aruco_id,
                         PoseStamped,
                         self.callback_pose)

    def callback_pose(self, msg):
        self.tf_listen.waitForTransform(self.base_link_name, self.odom_name, rospy.Time.now(), rospy.Duration(1.0))
        position, quaternion = self.tf_listen.lookupTransform(self.base_link_name, self.odom_name, rospy.Time(0))

        odom_to_baselink = tf.transformations.concatenate_matrices(tf.transformations.translation_matrix(position), tf.transformations.quaternion_matrix(quaternion))
        baselink_in_aruco_p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        baselink_in_aruco_q = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        baselink_to_aruco = tf.transformations.concatenate_matrices(tf.transformations.translation_matrix(baselink_in_aruco_p), tf.transformations.quaternion_matrix(baselink_in_aruco_q))

        odom_to_aruco = tf.transformations.concatenate_matrices(odom_to_baselink, baselink_to_aruco)

        quat = tf.transformations.quaternion_from_matrix(odom_to_aruco)
        quat = quat / np.linalg.norm(quat)

        self.tf_broadcast.sendTransform(odom_to_aruco[0:3, 3],
                         quat,
                         rospy.Time.now(),
                         self.odom_name,
                         msg.header.frame_id)


def main(args):
    rospy.init_node('aruco_tf_broadcaster')
    ArucoTFBroadcastNode()
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)
