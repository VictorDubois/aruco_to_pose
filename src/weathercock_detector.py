#!/usr/bin/env python3

import cv2
import rospy
from detection import Detector
import sys
import numpy as np
import math
import contextlib
import threading

from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from std_msgs.msg import Duration


@contextlib.contextmanager
def processing_lock(rospy, lock=threading.Lock()):
    if not lock.acquire(blocking=False):
        raise RuntimeError("locked")

    #rospy.loginfo("Got it!")
    try:
        yield lock
    finally:
        lock.release()


class WeathercockDetectorNode:
    """
    This node check the orientation of the weathercock using the camera
    The orientation is checked only when the robot is located inside an ROE
    The result is outputted as a ROS param on isWeathercockSouth
    """
    def __init__(self):
        cv2.setNumThreads(4)
        self.weathercock_id = 17
        roe = rospy.get_param("~roe", {'min': {"x":-4,"y":-4,"t":-3.14}, 'max': {"x":4,"y":4,"t":6.292}})
        self.weathercock_stabilisation_time = rospy.get_param("~weathercock_stabilisation_time", 30)
        self.roe_min = [roe["min"]["x"], roe["min"]["x"], roe["min"]["t"]]
        self.roe_max = [roe["max"]["x"], roe["max"]["y"], roe["max"]["t"]]
        self.d = Detector.Detector()
        rospy.set_param('isWeathercockSouth', False)
        self.is_set = False
        # subscribed Topic
        self.img_topic = "camera/image_raw/compressed"
        self.remaining_time_topic = "/remaining_time"
        self.pose_subscriber = rospy.Subscriber("odom", Odometry, callback=self.update_pose, queue_size=1)
        self.time_subscriber = rospy.Subscriber(self.remaining_time_topic, Duration, callback=self.update_rem_time, queue_size=1)
        self.image_subscriber = rospy.Subscriber(self.img_topic, CompressedImage, callback=self.callback, queue_size=1)
        self.remaining_time = rospy.Time.from_sec(100)
        self.pose = Pose()

    def update_pose(self, ros_data):
        self.pose = ros_data.pose.pose

    def update_rem_time(self, ros_data):
        self.remaining_time = ros_data.data

    def detect_weathercock_orientation(self, img):
        corners, ids = self.d.detect(img)
        if ids is not None:
            for aruco_id, corner in zip(ids, corners):
                if aruco_id[0] == self.weathercock_id:
                    is_south = corner[0, 0, 1] > corner[0, 3, 1]
                    self.is_set = True
                    rospy.set_param('isWeathercockSouth', bool(is_south))
                    rospy.loginfo(f"weathercock detected {'South' if is_south else 'North'}")
                    cv2.imwrite('/tmp/log/weathercock_picture.png', img)
                    rospy.signal_shutdown("Process done")
                    break

    def is_in_roe(self, pose_msg):
        q = pose_msg.orientation
        yaw = math.atan2(2.0 * (q.y * q.z + q.w * q.x), q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z)
        ego = [pose_msg.position.x, pose_msg.position.x, yaw]
        for i in range(0, 3):
            if not (self.roe_min[i] <= ego[i] <= self.roe_max[i]):
                return False
        return True

    def is_time_to_check(self):
        return self.remaining_time.to_sec() < 100 - self.weathercock_stabilisation_time

    def callback(self, img_msg):
        rospy.logdebug(self.is_time_to_check())
        rospy.logdebug(self.is_set)
        rospy.logdebug(self.is_in_roe(self.pose))

        if not self.is_set and self.is_in_roe(self.pose) and self.is_time_to_check():
            rospy.logdebug("It's time to check")
            thread = threading.Thread(target=self.process_image, args=[img_msg])
            thread.start()

    def process_image(self, img_msg):
        try:
            with processing_lock(rospy):
                rospy.logdebug("start frame")
                np_arr = np.fromstring(img_msg.data, np.uint8)
                image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                self.detect_weathercock_orientation(image_np)
                rospy.logdebug("end frame")
        except Exception:
            # Probably because a frame is already being processed
            rospy.loginfo("locked")

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
