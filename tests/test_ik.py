import json
import random
import time

import numpy as np
import numpy.typing as npt
import pytest
from google.protobuf.wrappers_pb2 import FloatValue, Int32Value
from reachy2_sdk import ReachySDK
from reachy2_sdk_api.arm_pb2 import (
    ArmCartesianGoal,
    IKConstrainedMode,
    IKContinuousMode,
)
from reachy2_sdk_api.kinematics_pb2 import Matrix4x4
from scipy.spatial.transform import Rotation as R


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


def symmetrical_pose_flip(T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    T_flip = np.array(
        [
            [T[0, 0], -T[0, 1], T[0, 2], T[0, 3]],
            [-T[1, 0], T[1, 1], -T[1, 2], -T[1, 3]],
            [T[2, 0], -T[2, 1], T[2, 2], T[2, 3]],
            [0, 0, 0, 1],
        ]
    )
    return T_flip


@pytest.mark.cicd
def test_reachable_poses() -> None:
    reachy = ReachySDK(host="localhost")
    time.sleep(1.0)
    assert reachy.is_connected

    reachy.turn_on()

    r_goal_poses = np.array(
        [
            # reachable poses
            # [[0.0001, -0.2, -0.6599], [0, 0, 0]],
            [[0.38, -0.2, -0.28], [0, -np.pi / 2, 0]],
            # [[0.07, -0.2, -0.50], [0, 0, 0]],  # backwards limit
            # [[0.07, -0.85, -0.0], [-np.pi / 2, 0, 0]],  # backwards limit
            [[0.48, -0.2, -0.08], [0, -np.pi / 2, 0]],
            # reachable poses with unconstrained mode
            # unreachable poses
            # [[0.30, -0.2, -0.28], [0.0, 0.0, np.pi / 3]],  # top grasp
            # [[0.0, -0.58, -0.28], [-np.pi / 2, -np.pi / 2, 0]],  # backwards limit
            # [[0.15, 0.35, -0.10], [np.pi / 3, -np.pi / 2, 0]],  # shoulder limit
            # [[0.10, 0.20, -0.22], [np.pi / 3, -np.pi / 2, 0]],  # shoulder limit
            # [[0.0, -0.2, -0.66], [0.0, 0.0, -np.pi / 3]],  # backwards limit
            # [[0.001, -0.2, -0.68], [0.0, 0.0, -np.pi / 3]],  # pose out of reach
            # [[0.001, -0.2, -0.659], [0.0, np.pi / 2, 0.0]],  # wrist out of reach
            # [[0.38, -0.2, -0.28], [0.0, np.pi / 2, 0.0]],  # wrist limit
            # [[0.1, -0.2, 0.0], [0.0, np.pi, 0.0]],  # elbow limit
            # [[0.38, -0.2, -0.28], [0.0, 0.0, 0.0]],  # shoulder limit?
            # [[0.1, 0.2, -0.1], [0.0, -np.pi / 2, np.pi / 2]],  # shoulder limit
        ]
    )

    for goal_pose in r_goal_poses:
        rotation_matrix = R.from_euler("xyz", np.rad2deg(goal_pose[1]), degrees=True).as_matrix()
        r_goal_pose = make_homogenous_matrix_from_rotation_matrix(position=goal_pose[0], rotation_matrix=rotation_matrix)
        l_goal_pose = symmetrical_pose_flip(r_goal_pose)

        go_to_pose(reachy, r_goal_pose, "r_arm")
        go_to_pose(reachy, l_goal_pose, "l_arm")
        time.sleep(1.5)

        r_real_pose = reachy.r_arm.forward_kinematics()
        l_real_pose = reachy.l_arm.forward_kinematics()

        assert np.linalg.norm(r_real_pose[:3, 3] - r_goal_pose[:3, 3]) < 1e-3
        assert np.linalg.norm(l_real_pose[:3, 3] - l_goal_pose[:3, 3]) < 1e-3
        assert np.linalg.norm(R.from_matrix(r_goal_pose[:3, :3] @ r_real_pose[:3, :3].T).as_rotvec()) < 1e-3
        assert np.linalg.norm(R.from_matrix(l_goal_pose[:3, :3] @ l_real_pose[:3, :3].T).as_rotvec()) < 1e-3
        assert np.linalg.norm(l_real_pose[:3, 3] - symmetrical_pose_flip(r_real_pose)[:3, 3]) < 1e-5

    assert True


@pytest.mark.cicd
def test_random_teleop_poses():
    reachy = ReachySDK(host="localhost")
    time.sleep(1.0)
    assert reachy.is_connected
    reachy.turn_on()

    with open("../src/example/pinocchio/pytest1.json", "r") as f:
        data = json.load(f)
    l_joints_lst = data["l_arm"]
    r_joints_lst = data["r_arm"]

    indices = random.sample(range(len(l_joints_lst)), k=10)
    l_goal_joints = [l_joints_lst[i] for i in indices]
    r_goal_joints = [r_joints_lst[i] for i in indices]

    # l_goal_joints = l_joints_lst
    # r_goal_joints = r_joints_lst

    for l_deg, r_deg in zip(l_goal_joints, r_goal_joints):
        l_goal_pose = reachy.l_arm.forward_kinematics(l_deg)
        r_goal_pose = reachy.r_arm.forward_kinematics(r_deg)

        for _ in range(100):
            go_to_pose(reachy, l_goal_pose, "l_arm")
            go_to_pose(reachy, r_goal_pose, "r_arm")
        time.sleep(1.5)

        l_real_pose = reachy.l_arm.forward_kinematics()
        r_real_pose = reachy.r_arm.forward_kinematics()

        assert np.linalg.norm(l_real_pose[:3, 3] - l_goal_pose[:3, 3]) < 2e-1
        assert np.linalg.norm(r_real_pose[:3, 3] - r_goal_pose[:3, 3]) < 2e-1

        err_l = R.from_matrix(l_goal_pose[:3, :3] @ l_real_pose[:3, :3].T).as_rotvec()
        err_r = R.from_matrix(r_goal_pose[:3, :3] @ r_real_pose[:3, :3].T).as_rotvec()
        assert np.linalg.norm(err_l) < 4e-1
        assert np.linalg.norm(err_r) < 4e-1

        # time.sleep(1.0)

    assert True
