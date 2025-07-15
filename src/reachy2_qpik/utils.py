"""Utilitaries functions for Control Loop IK."""

import copy
import math

import numpy as np


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    rate: int
        the rate.

    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).

    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
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

    # Pad the signal ectremes with values taken from the signal itself
    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode="valid")


def angle_diff(a: float, b: float) -> float:
    """Returns the smallest distance between 2 angles."""
    d = a - b
    d = ((d + math.pi) % (2 * math.pi)) - math.pi
    return d


def allow_multiturn(new_joints: list[float], prev_joints: list[float]) -> list[float]:
    """This function will always guarantee that the joint takes the shortest path to the new position.

    The practical effect is that it will allow the joint to rotate more than 2pi if it is the shortest path.
    """
    new_joints = copy.deepcopy(new_joints)
    for i in range(len(new_joints)):
        diff = angle_diff(new_joints[i], prev_joints[i])
        new_joints[i] = prev_joints[i] + diff
    return new_joints


def multiturn_safety_check(
    joints: list[float], shoulder_pitch_limit: float, elbow_yaw_limit: float, wrist_yaw_limit: float, emergency_state: str
) -> tuple[list[float], bool, str]:
    """Limit the number of turns allowed on the joints."""
    # print(f"{joints[0]:.2f}")
    joints = copy.deepcopy(joints)
    emergency_stop = False
    # Shoulder pitch
    if joints[1] > shoulder_pitch_limit:
        joints[1] = shoulder_pitch_limit
        emergency_state += "\n" + "EMERGENCY STOP: shoulder pitch limit reached"
        emergency_stop = True
    if joints[1] < -shoulder_pitch_limit:
        joints[1] = -shoulder_pitch_limit
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
    return joints, emergency_stop, emergency_state
