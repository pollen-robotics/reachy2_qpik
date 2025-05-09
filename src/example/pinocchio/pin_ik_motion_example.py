"""Pinocchio IK motion tests."""
import logging
import time

import numpy as np
import numpy.typing as npt
from metrics import combined_error, euler_error, l2_error, quat_error, rodrigues_error
from reachy2_sdk import ReachySDK
from scipy.spatial.transform import Rotation as R


def make_homogenous_matrix_from_rotation_matrix(
    rotation_matrix: npt.NDArray[np.float64], position: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Convert a 3x3 rotation matrix to a 4x4 homogenous matrix.

    Args:
        rotation_matrix: The 3x3 NumPy array representing the rotation matrix
        position: The 1x3 NumPy array representing the position of the end-effector

    Returns:
        A 4x4 NumPy array representing the pose matrix
    """
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = position
    return matrix


def goto_to_point(
    reachy: ReachySDK,
    arm: str,
    euler_angles: list[float],
    base_position: np.ndarray,
    degrees: bool = True,
    duration: float = 2.0,
) -> None:
    """This function commands Reachy's right arm to move to the specified target position."""
    rotation = R.from_euler("xyz", euler_angles, degrees=degrees).as_matrix()
    position = base_position.copy()

    target_pose = make_homogenous_matrix_from_rotation_matrix(rotation, position)

    if arm == "l_arm":
        target_pose_l = np.array(
            [
                [target_pose[0][0], -target_pose[0][1], target_pose[0][2], target_pose[0][3]],
                [-target_pose[1][0], target_pose[1][1], -target_pose[1][2], -target_pose[1][3]],
                [target_pose[2][0], -target_pose[2][1], target_pose[2][2], target_pose[2][3]],
                [0, 0, 0, 1],
            ]
        )

        target_pose = target_pose_l

    arm_ref = getattr(reachy, arm)
    start = time.time()
    arm_ref.goto(target_pose, interpolation_space="cartesian_space", duration=duration, wait=True)
    stop = time.time()

    actual_pose = arm_ref.forward_kinematics()
    p_des = target_pose[:3, 3]
    p = actual_pose[:3, 3]
    R_des = target_pose[:3, :3]
    R_curr = actual_pose[:3, :3]

    t = stop - start
    ep = l2_error(p_des, p)
    etheta = rodrigues_error(R_des, R_curr)
    equat = quat_error(R.from_matrix(R_des).as_quat(), R.from_matrix(R_curr).as_quat())
    euler = euler_error(R_des, R_curr)
    combined = combined_error(ep, etheta)

    print(f"Target pose: {target_pose}\n")
    print(f"Actual pose: {actual_pose}\n")
    print(f"== Metrics for {arm} ==")
    print(f"Time: {t:5f}s")
    print(f"L2 error: {ep:5f} m")
    print(f"Rodrigues error: {np.rad2deg(etheta):5f}°")
    print(f"Quaternion error: {np.rad2deg(equat):5f}°")
    print(f"Euler error: {np.rad2deg(euler):5f}°")
    print(f"Combined error: {combined:5f}")
    print(24 * "=")


if __name__ == "__main__":
    print("Pinocchio IK tests:")

    logging.basicConfig(level=logging.INFO)
    reachy = ReachySDK(host="localhost")

    if not reachy.is_connected:
        exit("Reachy is not connected.")

    print("Turning on Reachy")
    reachy.turn_on()
    time.sleep(0.2)

    input("Press Enter to start Tests:")
    print("Set to Elbow 90 pose ...")
    reachy.goto_posture("elbow_90", wait=True)

    configs = {
        "A": ([0, 0, 0], np.array([0, -0.2, -0.58])),  # Arms down
        "B": ([0, -90, 0], np.array([0.38, -0.2, -0.28])),  # Elbow 90°
        "C": ([0, -180, 0], np.array([0, -0.26, 0.66])),  # Arms up
    }

    test_sequence = ["A", "B", "C", "B", "A", "B", "C"]

    for i, key in enumerate(test_sequence, start=1):
        angles, position = configs[key]
        input(f"Press Enter to launch Test {i} ({key}):")
        print(f"Test {i} — Move to {position.tolist()} [m] with angles {angles} [°]...")
        goto_to_point(reachy, "l_arm", angles, position, duration=2)
        goto_to_point(reachy, "r_arm", angles, position, duration=2)

    input("Press Enter to terminate the program:")
    print("Set to Zero pose ...")
    reachy.goto_posture("default", wait=True)
    exit("Exiting tests")
