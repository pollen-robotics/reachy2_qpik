import numpy as np
from scipy.spatial.transform import Rotation as R


def l2_error(p_des, p):
    return np.linalg.norm(p_des - p)


def euler_error(R_des, R_curr):
    return np.linalg.norm(R.from_matrix(R_des).as_euler("xyz") - R.from_matrix(R_curr).as_euler("xyz"))


def rodrigues_error(R_des, R_curr):
    R_err = R_des @ R_curr.T
    err = R.from_matrix(R_err).as_rotvec()
    return np.linalg.norm(err)


def quat_error(q_des, q):
    c = np.clip(np.abs(np.dot(q_des, q)), -1.0, 1.0)
    return np.arccos(c)


def combined_error(ep, etheta, _lambda=1e-6 / np.deg2rad(1) ** 2):
    return np.sqrt(ep**2 + _lambda * etheta**2)


def velocity_error(p_des_prev, p_prev, p_des, p, dt, Kp=5):
    v_des = (p_des_prev - p_prev) / dt
    v = (p_prev - p) / dt

    ep = l2_error(p_des, p)

    v_cmd = v_des + Kp * ep

    edq = np.linalg.norm(v_des - v)

    return edq, v_cmd
