
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def marker_to_marker(Rref, Tref, R, T):
    Rref_r, _ = cv2.Rodrigues(Rref)
    R_r, _ = cv2.Rodrigues(R)
    R_f = np.linalg.inv(Rref_r).dot(R_r)
    T_f = np.linalg.inv(Rref_r).dot(np.transpose(T - Tref))
    return R_f, T_f