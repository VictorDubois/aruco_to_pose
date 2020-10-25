#!/usr/bin/env python3

import cv2
import rospy
from detection import Detector
import sys
import numpy as np
import math

from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry


class WeathercockDetectorNode:
    """
    This node check the orientation of the weathercock using the camera
    The orientation is checked only when the robot is located inside an ROE
    The result is outputted as a ROS param on isWeathercockSouth
    """
    def __init__(self):
        cv2.setNumThreads(4)
        self.weathercock_id = 17
        roe = rospy.get_param("~roe")
        self.roe_min = [roe["min"]["x"], roe["min"]["x"], roe["min"]["t"]]
        self.roe_max = [roe["max"]["x"], roe["max"]["y"], roe["max"]["t"]]
        self.d = Detector.Detector()
        rospy.set_param('isWeathercockSouth', False)
        self.is_set = False
        # subscribed Topic
        self.img_topic = "/robot_camera/image_raw/compressed"
        self.pose_subscriber = rospy.Subscriber("odom", Odometry, callback=self.callback, queue_size=1)

    def detect_weathercock_orientation(self, img):
        corners, ids = self.d.detect(img)
        if ids is not None:
            for aruco_id, corner in zip(ids, corners):
                if aruco_id[0] == self.weathercock_id:
                    is_south = corner[0,0,1] > corner[0,3,1]
                    self.is_set = True
                    rospy.set_param('isWeathercockSouth', bool(is_south))
                    break

    def is_in_roe(self, pose_msg):
        q = pose_msg.orientation
        yaw = math.atan2(2.0 * (q.y * q.z + q.w * q.x), q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z)
        ego = [pose_msg.position.x, pose_msg.position.x, yaw]
        for i in range(0, 3):
            if not (self.roe_min[i] <= ego[i] <= self.roe_max[i]):
                return False
        return True

    def callback(self, ros_data):
        if not self.is_set and self.is_in_roe(ros_data.pose.pose):
            img_msg = rospy.wait_for_message(self.img_topic, CompressedImage)
            np_arr = np.fromstring(img_msg.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.detect_weathercock_orientation(image_np)


def main(args):
    """ Initializes and cleanup ros node """
    rospy.init_node('aruco_pose_detector', anonymous=True)
    WeathercockDetectorNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
