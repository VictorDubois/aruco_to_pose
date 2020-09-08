
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def marker_to_marker(Rref, Tref, R, T):
    Rref_r, _ = cv2.Rodrigues(Rref)
    R_r, _ = cv2.Rodrigues(R)
    R_f = np.linalg.inv(Rref_r).dot(R_r)
    T_f = np.linalg.inv(Rref_r).dot(np.transpose(T - Tref))
    return R_f, T_f


def marker_to_plateau_3D(R, T, scale, center):
    Rref = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    Tref = center
    Rp = Rref.dot(R)
    Tp = Rref.dot(scale * T) + np.transpose(Tref)
    return Rp, Tp


def plateau_3D_to_plateau_2D(R, T, center, z):
    _, Tp = marker_to_plateau_3D(R, T, 200, center)
    n = np.array([0, 0, 1])
    tp = np.array([Tp[0][0], Tp[1][0], Tp[2][0]])
    position = (tp - (np.dot(tp, n) * n))[:2].astype(int)
    angle = Rot.from_matrix(R).as_euler('xyz', degrees=True)[2]
    return position, int(angle)
