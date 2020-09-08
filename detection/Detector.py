
import cv2


class Detector:
    def __init__(self):
        self.PARAM = cv2.aruco.DetectorParameters_create()
        self.DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

    def detect(self, frame, K, D):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.DICT, parameters=self.PARAM)
        return corners, ids

    def find_marker_pose(self, corners, K, D, size):
        rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, size, K, D)
        return rvec[0], tvec[0]


