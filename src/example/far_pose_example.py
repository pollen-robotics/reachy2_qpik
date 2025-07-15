"""Pinocchio IK far pose example."""

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


def main() -> None:
    """Main function."""
    print("Trying to connect on localhost Reachy...")
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()

    print("Test - Testing far target poses")

    Mr_bot = make_homogenous_matrix_from_rotation_matrix(
        np.array([0.2, -0.25, -0.58]), R.from_euler("xyz", [0, 0, 0]).as_matrix()
    )
    Mr_top = make_homogenous_matrix_from_rotation_matrix(
        np.array([0.2, -0.25, 0.45]), R.from_euler("xyz", [0, -np.pi, 0]).as_matrix()
    )

    Ml_0 = make_homogenous_matrix_from_rotation_matrix(np.array([0.2, 0.25, -0.58]), np.eye(3))
    Mr_0 = make_homogenous_matrix_from_rotation_matrix(np.array([0.2, -0.25, -0.58]), np.eye(3))

    reachy.l_arm.goto(Ml_0, interpolation_space="cartesian_space")
    reachy.r_arm.goto(Mr_0, interpolation_space="cartesian_space")
    time.sleep(3)

    Ml_top = np.array(
        [
            [Mr_top[0, 0], -Mr_top[0, 1], Mr_top[0, 2], Mr_top[0, 3]],
            [-Mr_top[1, 0], Mr_top[1, 1], -Mr_top[1, 2], -Mr_top[1, 3]],
            [Mr_top[2, 0], -Mr_top[2, 1], Mr_top[2, 2], Mr_top[2, 3]],
            [0, 0, 0, 1],
        ]
    )

    Ml_bot = np.array(
        [
            [Mr_bot[0, 0], -Mr_bot[0, 1], Mr_bot[0, 2], Mr_bot[0, 3]],
            [-Mr_bot[1, 0], Mr_bot[1, 1], -Mr_bot[1, 2], -Mr_bot[1, 3]],
            [Mr_bot[2, 0], -Mr_bot[2, 1], Mr_bot[2, 2], Mr_bot[2, 3]],
            [0, 0, 0, 1],
        ]
    )

    go_to_pose(reachy, Ml_top, "l_arm")
    go_to_pose(reachy, Mr_top, "r_arm")
    time.sleep(2)

    go_to_pose(reachy, Ml_bot, "l_arm")
    go_to_pose(reachy, Mr_bot, "r_arm")
    time.sleep(2)

    go_to_pose(reachy, Ml_top, "l_arm")
    go_to_pose(reachy, Mr_top, "r_arm")
    time.sleep(2)

    reachy.l_arm.goto(Ml_0, interpolation_space="cartesian_space")
    reachy.r_arm.goto(Mr_0, interpolation_space="cartesian_space")
    time.sleep(3)

    reachy.turn_off()


if __name__ == "__main__":
    main()
