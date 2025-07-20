"""Metrics utilities for IK error computation."""

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R


def l2_error(p_des: npt.NDArray[np.float64], p: npt.NDArray[np.float64]) -> np.float64:
    """Compute the Euclidean distance between the desired and current position."""
    return np.linalg.norm(p_des - p)


def euler_error(R_des: npt.NDArray[np.float64], R_curr: npt.NDArray[np.float64]) -> np.float64:
    """Compute the angular error between two rotation matrices using Euler angles."""
    euler_des = R.from_matrix(R_des).as_euler("xyz")
    euler_curr = R.from_matrix(R_curr).as_euler("xyz")
    return np.linalg.norm(euler_des - euler_curr)


def rot_error(P, Q):
    """Compute the angle between two rotation matrices P and Q.

    Think of this as an angular distance in the axis-angle representation.
    """
    # https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R = np.dot(P, Q.T)
    tr = (np.trace(R) - 1) / 2
    if tr > 1.0:
        tr = 1.0
    elif tr < -1.0:
        tr = -1.0
    return np.arccos(tr)


def rodrigues_error(R_des: npt.NDArray[np.float64], R_curr: npt.NDArray[np.float64]) -> np.float64:
    """Compute the angular error between two rotation matrices using the axis-angle formula."""
    R_err = R_des @ R_curr.T
    err = R.from_matrix(R_err).as_rotvec()
    return np.linalg.norm(err)


def quat_error(q_des: npt.NDArray[np.float64], q: npt.NDArray[np.float64]) -> np.float64:
    """Compute the angular difference between two quaternions."""
    c = np.clip(np.abs(np.dot(q_des, q)), -1.0, 1.0)
    return np.arccos(c)


def combined_error(ep: float, etheta: float, lambda_theta: float = 1e-6 / np.deg2rad(1) ** 2) -> np.float64:
    """Compute a weighted combination of position and orientation errors."""
    return np.sqrt(ep**2 + lambda_theta * etheta**2)


def unhinged_distance_between_poses(pose1, pose2) -> float:
    """Compute the distance between two poses in 6D space.

    Units be dammned, it is a well known fact that 1°==1mm and I'm tired of
    of pretending otherwise.
    """
    distance_translation = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
    distance_angle = rot_error(pose1[:3, :3], pose2[:3, :3])

    unhinged_distance = distance_translation * 1000 + np.rad2deg(distance_angle)

    return unhinged_distance
