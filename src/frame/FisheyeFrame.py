
from .Frame import Frame
import cv2
import numpy as np


class FisheyeFrame(Frame):
    def __init__(self, balance=1.0, id_=0, parameters=None):
        super().__init__(id_, parameters)
        self.map1, self.map2 = self.init_map()
        self.balance = balance

    def init_map(self):
        map1, map2 = None, None
        if self.K is not None and self.D is not None:
            w, h = int(self.frame_size[0]), int(self.frame_size[1])
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3),
                                                             self.K, (w, h), cv2.CV_16SC2)
        return map1, map2

    def undistorted_frame(self):
        frame = self.frame()
        if self.map1 is not None and self.map2 is not None:
            frame = cv2.remap(frame, self.map1, self.map2,
                              interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
        return frame

    def undistorted_scaled_frame(self, dim2=None, dim3=None):
        frame = self.frame()
        dim1 = frame.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
        #assert dim1[0] / dim1[1] == DIM[0] / DIM[
        #    1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
        if not dim2:
            dim2 = dim1
        if not dim3:
            dim3 = dim1
        scaled_K = self.K # * dim1[0] / dim1[0] #DIM[0]  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

        # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, self.D, dim2, np.eye(3), balance=self.balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, self.D, np.eye(3), self.new_K, dim3, cv2.CV_16SC2)
        undistorted_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img

    def update_balance(self, balance):
        self.balance = balance

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
