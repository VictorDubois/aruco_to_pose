#!/usr/bin/env python3

from frame import FisheyeFrame
from detection import Detector
from coordinate import Transforms
import sys

import cv2
import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation


class ArucoPublisherNode:
    def __init__(self):
        path_calibration_camera = rospy.get_param("~calib_file", 'resources/parameters_fisheye_pi.txt')
        rospy.loginfo(f"loading camera calibration from {path_calibration_camera}")
        self.f = FisheyeFrame.FisheyeFrame(id_=0, parameters=path_calibration_camera, balance=1)
        self.d = Detector.Detector()
        self.r_origin, self.t_origin = [[0,0,0]], [[0,0,0]]
        self.origin_id = 42
        self.initialized = False
        self.marker_sizes = {0: 0.07, 1: 0.07, 2: 0.07, 3: 0.07, 4: 0.07, 5: 0.07, 6: 0.07, 7: 0.07, 8: 0.07, 9: 0.07,
                             10: 0.07, 42: 0.10}
        self.image_pub = rospy.Publisher("/output/image_raw/compressed",
                                         CompressedImage, queue_size=1)
        self.robots_pose_pub = []
        for i in range(0, 10):
            self.robots_pose_pub.append(rospy.Publisher(f"/pose_robots/{i}", PoseStamped, queue_size=1))

        # subscribed Topic
        self.img_subscriber = rospy.Subscriber("/raspicam_node/image/compressed",
                                               CompressedImage, self.callback, queue_size=1)

    def core(self, img):
        self.f.grabbed_frame = img
        cv2.setNumThreads(4)
        img = self.f.undistorted_frame()
        corners, ids = self.d.detect(img)
        poses = {}

        if ids is not None:
            for aruco_id, corner in zip(ids, corners):
                poses[aruco_id[0]] = self.d.find_marker_pose(corner, self.f.K, self.f.D, self.marker_sizes[aruco_id[0]])

        if self.image_pub.get_num_connections() > 0:
            img = cv2.aruco.drawDetectedMarkers(img, corners, ids)
            #### Create CompressedIamge ####
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', img)[1]).tostring()
            # Publish new image
            self.image_pub.publish(msg)

        if poses.get(self.origin_id):
            self.r_origin, self.t_origin = poses[self.origin_id]

        for aruco_id in poses:
            rotation, translation = poses[aruco_id]
            rotation, translation = Transforms.marker_to_marker(self.r_origin, self.t_origin, rotation, translation)
            r = Rotation.from_matrix(rotation)

            if aruco_id < 10:
                pose_msg = PoseStamped()
                pose_msg.header.frame_id = "world"
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.pose.position.x = translation[0][0]
                pose_msg.pose.position.y = translation[1][0]
                pose_msg.pose.position.z = translation[2][0]
                quat = r.as_quat()
                pose_msg.pose.orientation.x = quat[0]
                pose_msg.pose.orientation.y = quat[1]
                pose_msg.pose.orientation.z = quat[2]
                pose_msg.pose.orientation.w = quat[3]
                self.robots_pose_pub[aruco_id].publish(pose_msg)

    def stop(self):
        self.f.close()

    def callback(self, ros_data):
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.core(image_np)


def main(args):
    """ Initializes and cleanup ros node """
    rospy.init_node('aruco_pose_detector', anonymous=True)
    ArucoPublisherNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
