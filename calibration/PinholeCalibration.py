
from calibration.Calibration import Calibration
import cv2


class PinholeCalibration(Calibration):
    def __init__(self, checkerboard):
        super().__init__(checkerboard)

    def compute_parameters(self, objpoints, imgpoints, img_shape, file):
        ret, K, D, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
        if ret:
            self.save_coefficients(K, D, file)
        else:
            print('Calibration failed')

