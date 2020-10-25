
import cv2


class Detector:
    def __init__(self):
        self.PARAM = cv2.aruco.DetectorParameters_create()
        self.DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

    def detect(self, frame):
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.DICT, parameters=self.PARAM)
        return corners, ids

    @staticmethod
    def find_marker_pose(corners, K, D, size):
        if cv2.__version__.startswith('4'):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, size, K, D)
        else:
            rvec, tvec = cv2.aruco.estimatePoseSingleMarkers(corners, size, K, D)

        return rvec[0], tvec[0]



