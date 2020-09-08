
from frame.Frame import Frame

import os
import cv2
import numpy as np


class Calibration:
    def __init__(self, checkerboard):
        self.CHECKERBOARD = checkerboard
        self.CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def record_calibration(self, f, folder):
        cv2.namedWindow('display')
        cv2.setNumThreads(4)
        i = 0
        while True:
            img = f.frame()
            disp = img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD, None)
            if ret:
                cv2.drawChessboardCorners(disp, self.CHECKERBOARD, corners, ret)
            cv2.imshow('display', disp)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
            elif key == ord('s'):
                if ret:
                    cv2.imwrite(os.path.join(folder, str(i) + '.png'), gray)
                    i += 1

    def detect_corners_checkerboard(self, _dir):
        objp = np.zeros((self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = self.load_calibration_img(_dir)

        img = cv2.imread(images[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        for name in images:
            img = cv2.imread(name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD, None)

            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.CRITERIA)
                objpoints.append(objp)
                imgpoints.append(corners2)

        return objpoints, imgpoints, img_shape

    def compute_parameters(self, objpoints, imgpoints, img_shape, file):
        pass

    @staticmethod
    def load_coefficients(path):
        K, D = None, None
        if path is not None:
            cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
            K = cv_file.getNode("K").mat()
            D = cv_file.getNode("D").mat()
            cv_file.release()
        return K, D

    @staticmethod
    def save_coefficients(K, D, path):
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        cv_file.write("K", K)
        cv_file.write("D", D)
        cv_file.release()

    @staticmethod
    def load_calibration_img(_dir):
        assert os.path.isdir(_dir), 'The image folder does not exist'
        images = []
        for root, _, files in os.walk(_dir, topdown=False):
            for name in files:
                if name.endswith('.png'):
                    images.append(os.path.join(root, name))
        return images
