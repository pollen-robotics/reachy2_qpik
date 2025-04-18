"""Pinocchio IK motion tests."""

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


def go_to_pose(reachy: ReachySDK, pose: npt.NDArray[np.float64], arm: str) -> None:
    """Move Reachy's arm the the specified target pose.

    Args:
        reachy: An instance of the ReachySDK to control the robot.
        pose: A 4x4 NumPy array representing the pose matrix.
        arm: An arm between left ("l_arm") or right ("r_arm").
    """
    if arm == "r_arm":
        request = ArmCartesianGoal(
            id=reachy.r_arm._part_id,
            goal_pose=Matrix4x4(data=pose.flatten().tolist()),
            continuous_mode=IKContinuousMode.CONTINUOUS,
            constrained_mode=IKConstrainedMode.UNCONSTRAINED,
            preferred_theta=FloatValue(
                value=-4 * np.pi / 6,
            ),
            d_theta_max=FloatValue(value=0.05),
            order_id=Int32Value(value=5),
        )
        reachy.r_arm._stub.SendArmCartesianGoal(request)

    elif arm == "l_arm":
        request = ArmCartesianGoal(
            id=reachy.l_arm._part_id,
            goal_pose=Matrix4x4(data=pose.flatten().tolist()),
            continuous_mode=IKContinuousMode.CONTINUOUS,
            constrained_mode=IKConstrainedMode.UNCONSTRAINED,
            preferred_theta=FloatValue(
                value=-4 * np.pi / 6,
            ),
            d_theta_max=FloatValue(value=0.05),
            order_id=Int32Value(value=5),
        )
        reachy.l_arm._stub.SendArmCartesianGoal(request)


def goto_to_point_A(reachy: ReachySDK, arm: str) -> None:
    """Move Reachy's arm to a Point in 3D space.

    This function commands Reachy's right arm to move to a specified target position.

    Args:
        reachy: An instance of the ReachySDK used to control the robot.
        arm: An arm between left arm ("l_arm") or right ("r_arm")
    """
    rotation = R.from_euler("xyz", [0, -90, 0], degrees=True).as_matrix()
    position = np.array([0.38, -0.2, -0.28])
    print(arm)

    if arm == "l_arm":
        position = np.array([0.38, 0.2, -0.28])
        target_pose = make_homogenous_matrix_from_rotation_matrix(rotation, position)
        reachy.l_arm.goto(target_pose, interpolation_space="cartesian_space", wait=True)
        print(target_pose)
        print(reachy.l_arm.forward_kinematics())
        print(f"Position error {np.linalg.norm(reachy.l_arm.forward_kinematics()[:3, 3] - target_pose[:3, 3]):5f}")
    else:
        target_pose = make_homogenous_matrix_from_rotation_matrix(rotation, position)
        reachy.r_arm.goto(target_pose, interpolation_space="cartesian_space", wait=True)
        print(target_pose)
        print(reachy.r_arm.forward_kinematics())
        print(f"Position error {np.linalg.norm(reachy.r_arm.forward_kinematics()[:3, 3] - target_pose[:3, 3]):5f}")


def goto_to_point_B(reachy: ReachySDK, arm: str) -> None:
    """Move Reachy's right arm to a Point in 3D space.

    This function commands Reachy's right arm to move to a specified target position.

    Args:
        reachy: An instance of the ReachySDK used to control the robot.
        arm: An arm between left arm ("l_arm") or right ("r_arm")
    """
    rotation = R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()
    position = np.array([0, -0.2, -0.58])
    print(arm)

    if arm == "l_arm":
        position = np.array([0, 0.2, -0.58])
        target_pose = make_homogenous_matrix_from_rotation_matrix(rotation, position)
        reachy.l_arm.goto(target_pose, interpolation_space="cartesian_space", wait=True)
        print(target_pose)
        print(reachy.l_arm.forward_kinematics())
        print(f"Position error {np.linalg.norm(reachy.l_arm.forward_kinematics()[:3, 3] - target_pose[:3, 3]):5f}")
    else:
        target_pose = make_homogenous_matrix_from_rotation_matrix(rotation, position)
        reachy.r_arm.goto(target_pose, interpolation_space="cartesian_space", wait=True)
        print(target_pose)
        print(reachy.r_arm.forward_kinematics())
        print(f"Position error {np.linalg.norm(reachy.r_arm.forward_kinematics()[:3, 3] - target_pose[:3, 3]):5f}")


def goto_to_point_C(reachy: ReachySDK, arm: str) -> None:
    """Move Reachy's right arm to a Point in 3D space.

    This function commands Reachy's right arm to move to a specified target position.

    Args:
        reachy: An instance of the ReachySDK used to control the robot.
        arm: An arm between left arm ("l_arm") or right ("r_arm")
    """
    rotation = R.from_euler("xyz", [0, -180, 0], degrees=True).as_matrix()
    position = np.array([9.98901949e-03, -2.56649267e-01, 6.57488464e-01])
    print(arm)
    if arm == "l_arm":
        position = np.array([9.98901949e-03, 2.56649267e-01, 6.57488464e-01])
        target_pose = make_homogenous_matrix_from_rotation_matrix(rotation, position)
        reachy.l_arm.goto(target_pose, interpolation_space="cartesian_space", wait=True)
        print(target_pose)
        print(reachy.l_arm.forward_kinematics())
        print(f"Position error {np.linalg.norm(reachy.l_arm.forward_kinematics()[:3, 3] - target_pose[:3, 3]):5f}")
    else:
        target_pose = make_homogenous_matrix_from_rotation_matrix(rotation, position)
        reachy.r_arm.goto(target_pose, interpolation_space="cartesian_space", wait=True)
        print(target_pose)
        print(reachy.r_arm.forward_kinematics())
        print(f"Position error {np.linalg.norm(reachy.r_arm.forward_kinematics()[:3, 3] - target_pose[:3, 3]):5f}")


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

    input("Press Enter to launch the Test 1:")
    print("Move to the point ()")
    goto_to_point_B(reachy, "r_arm")
    goto_to_point_B(reachy, "l_arm")

    input("Press Enter to launch the Test 2:")
    print("Move to the point ()")
    goto_to_point_A(reachy, "r_arm")
    goto_to_point_A(reachy, "l_arm")

    input("Press Enter to launch the Test 3:")
    print("Move to the point ()")
    goto_to_point_C(reachy, "r_arm")
    goto_to_point_C(reachy, "l_arm")

    input("Press Enter to launch the Test 4:")
    print("Move to the point ()")
    goto_to_point_A(reachy, "r_arm")
    goto_to_point_A(reachy, "l_arm")

    input("Press Enter to launch the Test 5:")
    print("Move to the point ()")
    goto_to_point_B(reachy, "r_arm")
    goto_to_point_B(reachy, "l_arm")

    input("Press Enter to launch the Test 6:")
    print("Move to the point ()")
    goto_to_point_A(reachy, "r_arm")
    goto_to_point_A(reachy, "l_arm")

    input("Press Enter to launch the Test 7:")
    print("Move to the point ()")
    goto_to_point_C(reachy, "r_arm")
    goto_to_point_C(reachy, "l_arm")

    input("Press Enter to terminate the program:")
    print("Set to Zero pose ...")
    reachy.goto_posture("default", wait=True)
    exit("Exiting tests")

    time.sleep(0.2)
