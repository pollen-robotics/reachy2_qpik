import logging
import time

import numpy as np
import numpy.typing as npt
from google.protobuf.wrappers_pb2 import FloatValue, Int32Value
from reachy2_sdk import ReachySDK
from reachy2_sdk_api.arm_pb2 import (
    ArmCartesianGoal,
    IKConstrainedMode,
    IKContinuousMode,
)
from reachy2_sdk_api.kinematics_pb2 import Matrix4x4
from scipy.spatial.transform import Rotation as R

l_gripper_angles = [0, -90, 0]
r_gripper_angles = [0, -90, 0]
l_R = R.from_euler("xyz", l_gripper_angles, degrees=True).as_matrix()
r_R = R.from_euler("xyz", r_gripper_angles, degrees=True).as_matrix()


def make_homogenous_matrix_from_rotation_matrix(
    rotation_matrix: npt.NDArray[np.float64], position: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    M = np.eye(4)
    M[:3, :3] = rotation_matrix
    M[:3, 3] = position
    return M


def go_to_pose(reachy: ReachySDK, pose: npt.NDArray[np.float64], arm: str):
    """Send a Cartesian goal to one arm."""
    part = getattr(reachy, arm)
    req = ArmCartesianGoal(
        id=part._part_id,
        goal_pose=Matrix4x4(data=pose.flatten().tolist()),
        continuous_mode=IKContinuousMode.CONTINUOUS,
        constrained_mode=IKConstrainedMode.UNCONSTRAINED,
        preferred_theta=FloatValue(value=-4 * np.pi / 6),
        d_theta_max=FloatValue(value=0.05),
        order_id=Int32Value(value=5),
    )
    part._stub.SendArmCartesianGoal(req)


def heart_curve(factor: float, n_pts: int):
    """Parametric heart curve in (y,z), returns two arrays of length n_pts."""
    s = np.linspace(0, 2 * np.pi, n_pts)
    y = 16 * np.sin(s)
    z = 13 * np.cos(s) - 5 * np.cos(2 * s) - 2 * np.cos(3 * s) - np.cos(4 * s)
    return factor * y, factor * z


def draw_heart(
    reachy: ReachySDK,
    center_x: float = 0.55,
    center_y_offset: float = 0.25,
    base_z: float = -0.05,
    scale: float = 0.01,
    duration: float = 6.0,
    freq: float = 100.0,
):
    """
    Draw two hearts with Reachy's arms.
    """
    n_pts = int(duration * freq)
    ys, zs = heart_curve(scale, n_pts)
    dt = 1.0 / freq

    left_positions = np.stack([np.full(n_pts, center_x), center_y_offset + ys, base_z + zs], axis=1)
    right_positions = np.stack([np.full(n_pts, center_x), -center_y_offset + ys, base_z + zs], axis=1)
    right_positions = right_positions[::-1]

    M_l0 = make_homogenous_matrix_from_rotation_matrix(l_R, left_positions[0])
    M_r0 = make_homogenous_matrix_from_rotation_matrix(r_R, right_positions[0])
    reachy.l_arm.goto(M_l0, interpolation_space="cartesian_space", duration=1.5)
    reachy.r_arm.goto(M_r0, interpolation_space="cartesian_space", duration=1.5)
    time.sleep(1.5)

    for lp, rp in zip(left_positions, right_positions):
        t = time.time()
        M_l = make_homogenous_matrix_from_rotation_matrix(l_R, lp)
        M_r = make_homogenous_matrix_from_rotation_matrix(r_R, rp)
        go_to_pose(reachy, M_l, "l_arm")
        go_to_pose(reachy, M_r, "r_arm")
        time.sleep(max(dt - (time.time() - t), 0.0))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reachy = ReachySDK(host="localhost")
    if not reachy.is_connected:
        raise SystemExit("Cannot connect to Reachy.")
    reachy.turn_on()
    time.sleep(0.5)

    reachy.l_arm.gripper.close()
    reachy.r_arm.gripper.close()
    time.sleep(0.5)

    print("Test - Making heart")
    draw_heart(reachy)

    reachy.turn_off()
