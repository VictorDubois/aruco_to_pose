
from plateau import Plateau
from frame import PinholeFrame
from detection import Detector
from coordinate.transforms import *

import cv2


class System:
    def __init__(self):
        self.p = Plateau.Plateau((600, 400), 'resources/img', 200, 42, np.array([[300, 250, 0]]))
        self.f = PinholeFrame.PinholeFrame(id_=0, parameters='resources/parameters_webcam_victor.txt')
        self.d = Detector.Detector()
        self.Rref, self.Tref = None, None
        self.initialized = False

    def initialize(self):
        cv2.namedWindow('display')
        cv2.setNumThreads(4)
        count_try = 50
        while not self.initialized and count_try > 0:
            count_try -= 1
            im = self.f.undistorted_frame()
            cv2.imshow('display', im.copy())
            cor, ids = self.d.detect(im, self.f.K, self.f.D)

            poses = {}
            if len(cor) > 0:
                for i in range(len(ids)):
                    poses[ids[i][0]] = self.d.find_marker_pose(cor[i], self.f.K, self.f.D, self.p.markers[ids[i][0]].size)

            if poses.get(self.p.origin_id):
                self.Rref, self.Tref = poses[self.p.origin_id]
                self.initialized = True
                print('Initialized!')
                break
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.stop()
                cv2.destroyAllWindows()
                break
        print('Could not initialize system')

    def core(self):
        if self.initialized:
            cv2.namedWindow('display')
            cv2.setNumThreads(4)
            self.p.update()
            while True:
                im = self.f.undistorted_frame()
                cor, ids = self.d.detect(im, self.f.K, self.f.D)

                poses = {}
                if len(cor) > 0:
                    for i in range(len(ids)):
                        poses[ids[i][0]] = self.d.find_marker_pose(cor[i], self.f.K, self.f.D,
                                                                   self.p.markers[ids[i][0]].size)

                disp = im.copy()
                disp = cv2.aruco.drawDetectedMarkers(disp, cor, ids)
                cv2.imshow('display', disp.copy())

                if poses.get(self.p.origin_id):
                    self.Rref, self.Tref = poses[self.p.origin_id]
                posrot = {}
                ids__ = []
                for id in poses:
                    if poses.get(id) and id != self.p.origin_id:
                        ids__.append(id)
                        R, T = poses[id]
                        Ror, Tor = marker_to_marker(self.Rref, self.Tref, R, T)
                        posrot[id] = plateau_3D_to_plateau_2D(Ror, Tor, self.p.center, 43)
                self.p.update(ids__, posrot)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    self.stop()
                    cv2.destroyAllWindows()
                    break

    def stop(self):
        self.f.close()
        cv2.destroyAllWindows()
