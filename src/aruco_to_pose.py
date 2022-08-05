#!/usr/bin/env python
import rospy
from frame import FisheyeFrame
from detection import Detector
from coordinate import Transforms
import sys

import cv2
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation
import threading
import contextlib

@contextlib.contextmanager
def processing_lock(rospy, lock=threading.Lock()):
    if not lock.acquire(blocking=False):
        raise RuntimeError("locked")

    #rospy.loginfo("Got it!")
    try:
        yield lock
    finally:
        lock.release()

class ArucoPublisherNode:
    def __init__(self):
        path_calibration_camera = rospy.get_param("~calib_file", 'resources/calib.out')
        rospy.loginfo(f"loading camera calibration from {path_calibration_camera}")
        self.f = FisheyeFrame.FisheyeFrame(id_=0, parameters=path_calibration_camera, balance=0.5)
        self.d = Detector.Detector()
        self.r_origin, self.t_origin = [[0,0,0]], [[0,0,0]]
        self.origin_id = 42
        self.initialized = False
        self.marker_sizes = {0: 0.07, 1: 0.07, 2: 0.07, 3: 0.07, 4: 0.07, 5: 0.07, 6: 0.07, 7: 0.07, 8: 0.07, 9: 0.07,
                             10: 0.07, 42: 0.10}
        self.image_pub_undistort = rospy.Publisher("aruco_to_pose/debug/undistort/compressed",
                                         CompressedImage, queue_size=1)
        self.robots_pose_pub = []
        for i in range(0, 11):
            self.robots_pose_pub.append(rospy.Publisher(f"/pose_robots/{i}", PoseStamped, queue_size=1))

        # subscribed Topic
        self.img_subscriber = rospy.Subscriber("camera/image/compressed",
                                               CompressedImage, self.callback, queue_size=1)
        #self.img_subscriber = rospy.Subscriber("camera/image",
        #                                       Image, self.callback, queue_size=1)

    def core(self, distort, timestamp):
        rospy.logdebug("in core")
        self.f.grabbed_frame = distort
        cv2.setNumThreads(4)
        rospy.loginfo("before undistort")
        rospy.loginfo("before undistort")
        img = self.f.undistorted_scaled_frame()
        rospy.loginfo("after undistort")
        rospy.loginfo("after undistort")
        corners, ids = self.d.detect(img)
        rospy.loginfo("after detect makers")
        rospy.loginfo("after detect makers")
        poses = {}

        if ids is not None:
            rospy.logdebug("aruco tag found!")
            rospy.logdebug("aruco tag found!")
            rospy.logdebug(ids)
            for aruco_id, corner in zip(ids, corners):
                if aruco_id[0] in self.marker_sizes:
                    poses[aruco_id[0]] = self.d.find_marker_pose(corner, self.f.new_K, np.array([]), self.marker_sizes[aruco_id[0]])
                    rospy.loginfo("after find_maker_pose of " + str(aruco_id[0]))
                    rospy.loginfo("after find_maker_pose of " + str(aruco_id[0]))
            rospy.loginfo("after find_maker_pose")
            rospy.loginfo("after find_maker_pose")

        if self.image_pub_undistort.get_num_connections() > 0:
            rospy.logdebug("Who wants my undistort?")
            img = cv2.aruco.drawDetectedMarkers(img, corners, ids)
            for pose in poses.values():
                img = cv2.aruco.drawAxis(img, self.f.new_K, np.array([]), pose[0], pose[1], 0.4)
            #### Create CompressedIamge ####
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', img)[1]).tostring()
            # Publish new image
            self.image_pub_undistort.publish(msg)

        if poses.get(self.origin_id):
            self.r_origin, self.t_origin = poses[self.origin_id]

        for aruco_id in poses:
            rotation, translation = poses[aruco_id]
            rotation, translation = Transforms.marker_to_marker(self.r_origin, self.t_origin, rotation, translation)
            r = Rotation.from_matrix(rotation)

            if aruco_id < 11:
                pose_msg = PoseStamped()
                pose_msg.header.frame_id = "aruco"
                pose_msg.header.stamp = timestamp
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
        if rospy.Time.now() - ros_data.header.stamp > rospy.Duration.from_sec(0.2):
            # In case we are lagging a lot
            rospy.loginfo("Old image, dropped")
            #return

        thread = threading.Thread(target=self.process_image, args=[ros_data])
        thread.start()

    def process_image(self, ros_data):
        try:
            with processing_lock(rospy):
                #rospy.loginfo("start frame")
                #rospy.logdebug("Inside callback")
                rospy.loginfo("start frame")
                np_arr = np.fromstring(ros_data.data, np.uint8)
                rospy.loginfo("after fromstring")
                image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                rospy.loginfo("after imdecode")
                self.core(image_np, ros_data.header.stamp)
                #rospy.loginfo("end frame")
        except Exception:
            # Probably because a frame is already being processed
            rospy.loginfo("locked")

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
