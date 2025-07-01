"""Pinocchio IK line motion test."""

import csv
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


def make_homogenous_matrix_from_rotation_matrix(
    position: npt.NDArray[np.float64], rotation_matrix: npt.NDArray[np.float64]
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


def go_to_pose(reachy: ReachySDK, pose: npt.NDArray[np.float64], arm: str) -> None:
    """Send a Cartesian goal to the specified arm."""
    req = ArmCartesianGoal(
        id=getattr(reachy, arm)._part_id,
        goal_pose=Matrix4x4(data=pose.flatten().tolist()),
        continuous_mode=IKContinuousMode.CONTINUOUS,
        constrained_mode=IKConstrainedMode.UNCONSTRAINED,
        preferred_theta=FloatValue(value=-4 * np.pi / 6),
        d_theta_max=FloatValue(value=0.05),
        order_id=Int32Value(value=5),
    )
    stub = getattr(reachy, arm)._stub
    stub.SendArmCartesianGoal(req)


def make_line(
    reachy: ReachySDK,
    start_pose: npt.NDArray[np.float64],
    end_pose: npt.NDArray[np.float64],
    duration: float = 4.0,
    collect_data: bool = False,
) -> None:
    start_position = start_pose[0]
    end_position = end_pose[0]
    start_orientation = start_pose[1]
    end_orientation = end_pose[1]

    if collect_data:
        input("`collect_data` is set to `True`. Press `Enter` to continue if you are sure with the parameters:")
        data_lst = []

    control_frequency = 120.0
    dt = 1.0 / control_frequency
    nbr_points = int(duration * control_frequency)
    total_time = 0.0

    l_start_position = np.array([start_position[0], -start_position[1], start_position[2]])
    l_end_position = np.array([end_position[0], -end_position[1], end_position[2]])
    l_start_orientation = np.array([-start_orientation[0], start_orientation[1], -start_orientation[2]])
    l_end_orientation = np.array([-end_orientation[0], end_orientation[1], -end_orientation[2]])

    for i in range(nbr_points):
        t = time.time()
        position = start_position + (end_position - start_position) * (i / nbr_points)
        orientation = start_orientation + (end_orientation - start_orientation) * (i / nbr_points)
        rotation_matrix = R.from_euler("xyz", orientation).as_matrix()
        r_pose = make_homogenous_matrix_from_rotation_matrix(position, rotation_matrix)
        go_to_pose(reachy, r_pose, "r_arm")

        l_position = l_start_position + (l_end_position - l_start_position) * (i / nbr_points)
        l_orientation = l_start_orientation + (l_end_orientation - l_start_orientation) * (i / nbr_points)
        l_rotation_matrix = R.from_euler("xyz", l_orientation).as_matrix()
        l_pose = make_homogenous_matrix_from_rotation_matrix(l_position, l_rotation_matrix)
        go_to_pose(reachy, l_pose, "l_arm")

        # print(f"Loop time: {(time.time() - t)*1000:.1f} ms")
        # print(f"Right arm: {1000*(t2-t1)} ms")
        # print(f"Left arm: {1000*(t4-t3)} ms")
        time.sleep(max(dt - (time.time() - t), 0.0))
        total_time += time.time() - t

        if collect_data:
            time.sleep(0.05)
            l_joints = reachy.l_arm.get_current_positions()
            r_joints = reachy.r_arm.get_current_positions()

            r_real_pose = reachy.r_arm.forward_kinematics()
            l_real_pose = reachy.l_arm.forward_kinematics()

            data = [i * dt, l_joints, l_pose, l_real_pose, r_joints, r_pose, r_real_pose]
            data_lst.append(data)

    if collect_data:
        save_data_to_csv(data_lst, filename="sym_line_data.csv")

    print(f"Total time: {total_time:.3f} s")


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


def main() -> None:
    print("Trying to connect on localhost Reachy...")
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()

    print("Test - Making a line")
    start_pose = np.array([[0.4, -0.4, -0.2], [0, -np.pi / 2, 0]])
    end_pose = np.array([[0.4, -0.1, -0.2], [0, -np.pi / 2, 0]])
    rotation_matrix = R.from_euler("xyz", [0, -np.pi / 2, 0]).as_matrix()
    Ml_0 = make_homogenous_matrix_from_rotation_matrix(np.array([0.4, 0.4, -0.2]), rotation_matrix)
    Mr_0 = make_homogenous_matrix_from_rotation_matrix(np.array([0.4, -0.4, -0.2]), rotation_matrix)

    reachy.r_arm.goto(Mr_0, interpolation_space="cartesian_space")
    reachy.l_arm.goto(Ml_0, interpolation_space="cartesian_space")
    time.sleep(3)

    make_line(reachy, start_pose, end_pose, collect_data=False)
    time.sleep(2)

    reachy.turn_off()


if __name__ == "__main__":
    main()
