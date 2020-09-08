
from calibration.Calibration import Calibration
import cv2
import numpy as np


class FisheyeCalibration(Calibration):
    def __init__(self, checkerboard):
        super().__init__(checkerboard)
        self.CALIB_FLAGS = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

    def compute_parameters(self, objpoints, imgpoints, img_shape, file):
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
        rms, _, _, _, _ = cv2.fisheye.calibrate(objpoints, imgpoints, img_shape,
                                                K, D, rvecs, tvecs, self.CALIB_FLAGS,
                                                super().CRITERIA)

        self.save_coefficients(K, D, file)
