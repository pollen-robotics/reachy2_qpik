"""Utilitaries functions for Control Loop IK."""

import copy
import math

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R


def limit_orbita3d_joints(joints: list[float], orbita3D_max_angle: float) -> list[float]:
    """Casts the 3 orientations to ensure the orientation is reachable by an Orbita3D. i.e. casting into Orbita's cone.

    Args:
        joints (list[float]): List of joint values [roll, pitch, yaw] in radians.
        orbita3D_max_angle (float): Maximum allowed polar angle for Orbita3D [rad].

    Returns:
        list[float]: Adjusted [roll, pitch, yaw] values within the valid cone.
    """
    joints = copy.deepcopy(joints)
    rotation = R.from_euler("XYZ", [joints[0], joints[1], joints[2]], degrees=False)
    new_joints = rotation.as_euler("ZYZ", degrees=False)
    new_joints[1] = min(orbita3D_max_angle, max(-orbita3D_max_angle, new_joints[1]))
    rotation = R.from_euler("ZYZ", new_joints, degrees=False)
    [roll, pitch, yaw] = rotation.as_euler("XYZ", degrees=False)
    joints = [float(roll), float(pitch), float(yaw)]
    return joints


def limit_orbita3d_joints_wrist(joints: list[float], orbita3D_max_angle: float) -> list[float]:
    """Casts the 3 orientations to ensure the orientation is reachable by an Orbita3D using the wrist conventions.
    i.e. casting into Orbita's cone.

    Args:
        joints (list[float]): Full list of joint values for the arm (7 values).
        orbita3D_max_angle (float): Maximum allowed polar angle for Orbita3D [rad].

    Returns:
        list[float]: Adjusted joint values with wrist within the valid cone.
    """
    joints = copy.deepcopy(joints)
    wrist_joints = joints[4:7]

    wrist_joints = limit_orbita3d_joints(wrist_joints, orbita3D_max_angle)

    joints[4:7] = wrist_joints

    return joints


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) a signal with a Savitzky-Golay filter.

    This filter removes high-frequency noise while preserving the shape
    and features of the signal. Optionally, it can compute the nth derivative.

    Args:
        y (array_like): Input signal, shape (N,).
        window_size (int): Length of the filter window (odd number).
        order (int): Order of the polynomial used in the filtering (must be < `window_size` - 1).
        deriv (int, optional): Order of the derivative to compute. Defaults to 0 (only smoothing).
        rate (int, optional): Sample rate multiplier. Defaults to 1.

    Returns:
        numpy.ndarray: Smoothed signal (or its nth derivative).
    """
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order + 1)
    half_window = (window_size - 1) // 2

    # Precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * math.factorial(deriv)

    # Pad the signal extremes with values taken from the signal itself
    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode="valid")


def multiturn_safety_check(
    joints: npt.NDArray[np.float64],
    shoulder_pitch_limit: float,
    elbow_yaw_limit: float,
    wrist_yaw_limit: float,
    emergency_state: str,
) -> tuple[npt.NDArray[np.float64], bool, str]:
    """Limit the number of turns allowed on the joints and detect violations.

    Args:
        joints (numpy.ndarray): Current joint positions [rad].
        shoulder_pitch_limit (float): Maximum absolute allowed shoulder pitch angle [rad].
        elbow_yaw_limit (float): Maximum absolute allowed elbow yaw angle [rad].
        wrist_yaw_limit (float): Maximum absolute allowed wrist yaw angle [rad].
        emergency_state (str): Previous emergency state message (if any).

    Returns:
        tuple:
            - numpy.ndarray: Adjusted joint positions (clamped to limits).
            - bool: True if an emergency stop condition was triggered.
            - str: Updated emergency state message.
    """
    # print(f"[{joints[1]:.2f},{joints[2]:.2f},{joints[6]:.2f}]")
    joints = copy.deepcopy(joints)
    emergency_stop = False
    # Shoulder pitch
    if joints[0] > shoulder_pitch_limit:
        joints[0] = shoulder_pitch_limit
        emergency_state += "\n" + "EMERGENCY STOP: shoulder pitch limit reached"
        emergency_stop = True
    if joints[0] < -shoulder_pitch_limit:
        joints[0] = -shoulder_pitch_limit
        emergency_state += "\n" + "EMERGENCY STOP: shoulder pitch limit reached"
        emergency_stop = True
    # Elbow yaw
    if joints[2] > elbow_yaw_limit:
        joints[2] = elbow_yaw_limit
        emergency_state += "\n" + "EMERGENCY STOP: elbow yaw limit reached"
        emergency_stop = True
    if joints[2] < -elbow_yaw_limit:
        joints[2] = -elbow_yaw_limit
        emergency_state += "\n" + "EMERGENCY STOP: elbow yaw limit reached"
        emergency_stop = True
    # Wrist yaw
    if joints[6] > wrist_yaw_limit:
        joints[6] = wrist_yaw_limit
        emergency_state += "\n" + "EMERGENCY STOP: wrist yaw limit reached"
        emergency_stop = True
    if joints[6] < -wrist_yaw_limit:
        joints[6] = -wrist_yaw_limit
        emergency_state += "\n" + "EMERGENCY STOP: wrist yaw limit reached"
        emergency_stop = True
    return np.array(joints), emergency_stop, emergency_state
