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
