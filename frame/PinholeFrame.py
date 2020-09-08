
from .Frame import Frame
import cv2


class PinholeFrame(Frame):
    def __init__(self, id_=0, parameters=None):
        super().__init__(id_, parameters)

    def undistorted_frame(self):
        frame = self.frame()
        if self.K is not None and self.D is not None:
            frame = cv2.undistort(frame, self.K, self.D, None, None)
        return frame
