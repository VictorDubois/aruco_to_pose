
import cv2
import os


class Frame:
    def __init__(self, id_=0, parameters=None):
        self.K, self.D, self.frame_size = self.load_coefficients(parameters)
        self.grabbed_frame = None

    @staticmethod
    def load_coefficients(path):
        K, D, frame_size = None, None, None
        if path is not None:
            cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
            K = cv_file.getNode("K").mat()
            D = cv_file.getNode("D").mat()
            frame_size = cv_file.getNode("shape").mat()
            cv_file.release()
        return K, D, frame_size

    @staticmethod
    def open(id_):
        return None

    def close(self):
        pass

    def frame(self):
        return self.grabbed_frame

    def undistorted_frame(self):
        pass
