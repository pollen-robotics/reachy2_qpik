"""Pinocchio IK heart motion example."""

import csv
import logging
import os
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
    """Convert a 3x3 rotation matrix to a 4x4 homogenous matrix.

    Args:
        rotation_matrix: The 3x3 NumPy array representing the rotation matrix
        position: The 1x3 NumPy array representing the position of the end-effector

    Returns:
        A 4x4 NumPy array representing the pose matrix
    """
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
    base_z: float = -0.06,
    scale: float = 0.008,
    duration: float = 4.0,
    freq: float = 120.0,
    number_of_turns: int = 4,
    collect_data: bool = False,
):
    """Draw two hearts with Reachy's arms."""
    nbr_points = int(duration * freq)
    ys, zs = heart_curve(scale, nbr_points)
    dt = 1.0 / freq
    total_time = 0.0

    if collect_data:
        input("`collect_data` is set to `True`. Press `Enter` to continue if you are sure with the parameters:")
        data_lst = []

    left_positions = np.stack([np.full(nbr_points, center_x), center_y_offset + ys, base_z + zs], axis=1)
    right_positions = np.stack([np.full(nbr_points, center_x), -center_y_offset + ys, base_z + zs], axis=1)
    right_positions = right_positions[::-1]

    M_l0 = make_homogenous_matrix_from_rotation_matrix(l_R, left_positions[0])
    M_r0 = make_homogenous_matrix_from_rotation_matrix(r_R, right_positions[0])
    reachy.l_arm.goto(M_l0, interpolation_space="cartesian_space", duration=3)
    reachy.r_arm.goto(M_r0, interpolation_space="cartesian_space", duration=3, wait=True)
    time.sleep(1.5)

    for i in range(number_of_turns):
        for j, (lp, rp) in enumerate(zip(left_positions, right_positions)):
            t = time.time()
            M_l = make_homogenous_matrix_from_rotation_matrix(l_R, lp)
            M_r = make_homogenous_matrix_from_rotation_matrix(r_R, rp)
            go_to_pose(reachy, M_l, "l_arm")
            go_to_pose(reachy, M_r, "r_arm")

            time.sleep(max(dt - (time.time() - t), 0.0))
            total_time += time.time() - t

            if collect_data:
                time.sleep(0.05)
                r_real_pose = reachy.r_arm.forward_kinematics()
                l_real_pose = reachy.l_arm.forward_kinematics()

                l_joints = reachy.l_arm.get_current_positions()
                r_joints = reachy.r_arm.get_current_positions()

                data = [(i * nbr_points + j) * dt, l_joints, M_l, l_real_pose, r_joints, M_r, r_real_pose]
                data_lst.append(data)

    if collect_data:
        save_data_to_csv(data_lst, filename="qp_heart_data.csv")

    print(f"Total time: {total_time:.3f} s")
    print(f"Time for one heart: {total_time / number_of_turns:.3}s")


def save_data_to_csv(data_lst, folder: str = "data", filename: str = "data.csv") -> None:
    """Save collected data to a CSV file."""
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    header = [
        "time",
        "l_q0",
        "l_q1",
        "l_q2",
        "l_q3",
        "l_q4",
        "l_q5",
        "l_q6",
        "r_q0",
        "r_q1",
        "r_q2",
        "r_q3",
        "r_q4",
        "r_q5",
        "r_q6",
        "l_pose",
        "l_real_pose",
        "r_pose",
        "r_real_pose",
    ]

    with open(filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for data in data_lst:
            time_val = data[0]
            l_joints = data[1]
            l_pose = data[2]
            l_real_pose = data[3]
            r_joints = data[4]
            r_pose = data[5]
            r_real_pose = data[6]

            row = (
                [time_val]
                + l_joints
                + r_joints
                + [l_pose.tolist(), l_real_pose.tolist(), r_pose.tolist(), r_real_pose.tolist()]
            )

            writer.writerow(row)
    print(f"Data was succesfully saved to {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reachy = ReachySDK(host="localhost")
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        exit()
    
    reachy.turn_on()
    time.sleep(0.5)

    reachy.l_arm.gripper.close()
    reachy.r_arm.gripper.close()
    time.sleep(0.5)

    print("Example - Making heart")
    draw_heart(reachy, collect_data=False)

    time.sleep(2)

    Mr_bot = make_homogenous_matrix_from_rotation_matrix(
        R.from_euler("xyz", [0, 0, 0]).as_matrix(), np.array([0.2, -0.25, -0.58])
    )

    Ml_bot = np.array(
        [
            [Mr_bot[0, 0], -Mr_bot[0, 1], Mr_bot[0, 2], Mr_bot[0, 3]],
            [-Mr_bot[1, 0], Mr_bot[1, 1], -Mr_bot[1, 2], -Mr_bot[1, 3]],
            [Mr_bot[2, 0], -Mr_bot[2, 1], Mr_bot[2, 2], Mr_bot[2, 3]],
            [0, 0, 0, 1],
        ]
    )

    reachy.l_arm.goto(Ml_bot, interpolation_space="cartesian_space", duration=4)
    reachy.r_arm.goto(Mr_bot, interpolation_space="cartesian_space", duration=4, wait=True)
    time.sleep(2)

    reachy.turn_off()
