
from .Frame import Frame
import cv2
import numpy as np


class FisheyeFrame(Frame):
    def __init__(self, balance=1.0, id_=0, parameters=None):
        Frame.__init__(self, id_, parameters)
        self.balance = balance
        self.map1, self.map2, self.new_K = None, None, None
        self.init_map()


    def init_map(self):
        if self.K is not None and self.D is not None:
            self.new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D, self.frame_size, self.balance,
                                                          centerPrincipalPoint=True)
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.new_K, self.frame_size,
                                                             cv2.CV_16SC2)

    def undistorted_frame(self):
        frame = self.frame()
        if self.map1 is not None and self.map2 is not None:
            frame = cv2.remap(frame, self.map1, self.map2,
                              interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
        return frame

    def undistorted_scaled_frame(self):
        frame = self.frame()
        undistorted_img = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img

    def update_balance(self, balance):
        self.balance = balance
        self.init_map()

    def check_balance(self):
        def nothing(x):
            pass

        cv2.namedWindow('Display')
        cv2.createTrackbar('balance', 'Display', 0, 100, nothing)

        while True:
            frame = self.frame()
            bal = cv2.getTrackbarPos('balance', 'Display') / 100
            self.update_balance(bal)
            img_und = self.undistorted_scaled_frame()
            cv2.imshow('Display', img_und)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()
