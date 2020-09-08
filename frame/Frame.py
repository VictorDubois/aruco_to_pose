
import cv2
import os


class Frame:
    def __init__(self, id_=0, parameters=None):
        self.cap = self.open(id_)
        self.K, self.D = self.load_coefficients(parameters)

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
    def open(id_):
        cap = cv2.VideoCapture(id_)
        if not cap.isOpened():
            print('Unable to read camera feed')
        return cap

    def close(self):
        if self.cap.isOpened():
            self.cap.release()

    def frame(self):
        _, frame = self.cap.read()
        return frame

    def undistorted_frame(self):
        pass
