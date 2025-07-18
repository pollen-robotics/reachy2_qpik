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


@pytest.mark.online
def test_reachable_poses() -> None:
    reachy = ReachySDK(host="localhost")
    time.sleep(1.0)
    assert reachy.is_connected

    reachy.turn_on()

    # Reachable poses
    r_goal_poses = np.array(
        [
            [[0.0001, -0.2, -0.6599], [0, 0, 0]],
            [[0.38, -0.2, -0.28], [0, -np.pi / 2, 0]],
            # [[0.07, -0.2, -0.50], [0, 0, 0]],  # backwards limit
            # [[0.07, -0.85, -0.0], [-np.pi / 2, 0, 0]],  # backwards limit
            [[0.48, -0.2, -0.08], [0, -np.pi / 2, 0]],
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

        # Position and rotation errors
        assert np.linalg.norm(r_real_pose[:3, 3] - r_goal_pose[:3, 3]) < 1.5e-1
        assert np.linalg.norm(l_real_pose[:3, 3] - l_goal_pose[:3, 3]) < 1.5e-1
        assert np.linalg.norm(R.from_matrix(r_goal_pose[:3, :3] @ r_real_pose[:3, :3].T).as_rotvec()) < 1e-2
        assert np.linalg.norm(R.from_matrix(l_goal_pose[:3, :3] @ l_real_pose[:3, :3].T).as_rotvec()) < 1e-2

        # Symmetrism between left and right arm
        assert np.linalg.norm(l_real_pose[:3, 3] - symmetrical_pose_flip(r_real_pose)[:3, 3]) < 1e-5

    reachy.turn_off()
    assert True


@pytest.mark.online
def test_random_teleop_poses():
    reachy = ReachySDK(host="localhost")
    time.sleep(1.0)
    assert reachy.is_connected
    reachy.turn_on()

    with open("../src/config_files/pytest_teleop.json", "r") as f:
        data = json.load(f)
    l_joints_lst = data["l_arm"]
    r_joints_lst = data["r_arm"]

    indices = random.sample(range(len(l_joints_lst)), k=10)
    l_goal_joints = [l_joints_lst[i] for i in indices]
    r_goal_joints = [r_joints_lst[i] for i in indices]

    for l_deg, r_deg in zip(l_goal_joints, r_goal_joints):
        l_goal_pose = reachy.l_arm.forward_kinematics(l_deg)
        r_goal_pose = reachy.r_arm.forward_kinematics(r_deg)

        go_to_pose(reachy, l_goal_pose, "l_arm")
        go_to_pose(reachy, r_goal_pose, "r_arm")
        time.sleep(1.5)

        l_real_pose = reachy.l_arm.forward_kinematics()
        r_real_pose = reachy.r_arm.forward_kinematics()

        assert np.linalg.norm(l_real_pose[:3, 3] - l_goal_pose[:3, 3]) < 1.5e-1
        assert np.linalg.norm(r_real_pose[:3, 3] - r_goal_pose[:3, 3]) < 1.5e-1

        err_l = R.from_matrix(l_goal_pose[:3, :3] @ l_real_pose[:3, :3].T).as_rotvec()
        err_r = R.from_matrix(r_goal_pose[:3, :3] @ r_real_pose[:3, :3].T).as_rotvec()
        assert np.linalg.norm(err_l) < 1.5
        assert np.linalg.norm(err_r) < 1.5

    reachy.turn_off()
    assert True


@pytest.mark.online
def test_circle() -> None:
    reachy = ReachySDK(host="localhost")
    time.sleep(1.0)
    assert reachy.is_connected

    reachy.turn_on()

    # Circle parameters
    radius = 0.15
    center = np.array([0.4, -0.4, -0.2])
    orientation = np.array([0, -np.pi / 2, 0])
    rotation_matrix = R.from_euler("xyz", orientation).as_matrix()

    duration = 1.0
    control_frequency = 120.0
    number_of_turns = 2

    l_goal_pose = make_homogenous_matrix_from_rotation_matrix(np.array([0.4, 0.25, -0.2]), rotation_matrix)
    r_goal_pose = make_homogenous_matrix_from_rotation_matrix(np.array([0.4, -0.25, -0.2]), rotation_matrix)
    reachy.r_arm.goto(r_goal_pose, interpolation_space="cartesian_space")
    reachy.l_arm.goto(l_goal_pose, interpolation_space="cartesian_space")
    time.sleep(3)

    r_real_pose = reachy.r_arm.forward_kinematics()
    l_real_pose = reachy.l_arm.forward_kinematics()

    assert np.linalg.norm(r_real_pose[:3, 3] - r_goal_pose[:3, 3]) < 1.5e-1
    assert np.linalg.norm(l_real_pose[:3, 3] - l_goal_pose[:3, 3]) < 1.5e-1
    assert np.linalg.norm(R.from_matrix(r_goal_pose[:3, :3] @ r_real_pose[:3, :3].T).as_rotvec()) < 1e-2
    assert np.linalg.norm(R.from_matrix(l_goal_pose[:3, :3] @ l_real_pose[:3, :3].T).as_rotvec()) < 1e-2
    assert np.linalg.norm(l_real_pose[:3, 3] - symmetrical_pose_flip(r_real_pose)[:3, 3]) < 1e-3

    nbr_points = int(duration * control_frequency)

    Y_r = center[1] + radius * np.cos(np.linspace(0, 2 * np.pi, nbr_points))
    Z = center[2] + radius * np.sin(np.linspace(0, 2 * np.pi, nbr_points))
    X = center[0] * np.ones(nbr_points)
    Y_l = -center[1] - radius * np.cos(np.linspace(0, 2 * np.pi, nbr_points))

    Y_l = Y_l[::-1]
    Y_r = Y_r[::-1]
    Z = Z[::-1]

    dt = 1 / control_frequency

    l_previous_joints = reachy.l_arm.get_current_positions()
    r_previous_joints = reachy.r_arm.get_current_positions()

    for _ in range(number_of_turns):
        for j in range(nbr_points):
            t = time.time()
            position = np.array([X[j], Y_r[j], Z[j]])
            rotation_matrix = R.from_euler("xyz", orientation).as_matrix()
            r_pose = make_homogenous_matrix_from_rotation_matrix(position, rotation_matrix)
            go_to_pose(reachy, r_pose, "r_arm")

            l_position = np.array([X[j], Y_l[j], Z[j]])
            l_rotation_matrix = R.from_euler("xyz", orientation).as_matrix()
            l_pose = make_homogenous_matrix_from_rotation_matrix(l_position, l_rotation_matrix)
            go_to_pose(reachy, l_pose, "l_arm")

            time.sleep(max(dt - (time.time() - t), 0.0))

            r_real_pose = reachy.r_arm.forward_kinematics()
            l_real_pose = reachy.l_arm.forward_kinematics()

            l_joints = reachy.l_arm.get_current_positions()
            r_joints = reachy.r_arm.get_current_positions()

            assert np.linalg.norm(r_real_pose[:3, 3] - r_goal_pose[:3, 3]) < 5.5e-1
            assert np.linalg.norm(l_real_pose[:3, 3] - l_goal_pose[:3, 3]) < 5.5e-1
            assert np.linalg.norm(R.from_matrix(r_goal_pose[:3, :3] @ r_real_pose[:3, :3].T).as_rotvec()) < 1e-2
            assert np.linalg.norm(R.from_matrix(l_goal_pose[:3, :3] @ l_real_pose[:3, :3].T).as_rotvec()) < 1e-2
            assert np.linalg.norm(l_real_pose[:3, 3] - symmetrical_pose_flip(r_real_pose)[:3, 3]) < 1e-2
            assert np.allclose(l_joints, l_previous_joints, atol=5)
            assert np.allclose(r_joints, r_previous_joints, atol=5)
            l_previous_joints = l_joints
            r_previous_joints = r_joints

    reachy.turn_off()
    assert True


@pytest.mark.online
def test_random_trajectory() -> None:
    reachy = ReachySDK(host="localhost")
    time.sleep(1.0)
    assert reachy.is_connected

    reachy.turn_on()

    q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [rad]
    ik_r = q
    ik_l = q
    q0 = [-90.0, -80.0, 0.0, -65.0, 0.0, 0.0, 0.0]  # [rad]
    q_amps = [90.0, 90.0, 180.0, 65.0, 45.0, 45.0, 30.0]  # [rad]
    l_previous_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [rad]
    r_previous_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [rad]

    freq_reductor = 0.3
    freq = [0.3 * freq_reductor, 0.17 * freq_reductor, 0.39 * freq_reductor, 0.18, 0.15, 0.15, 0.15 * freq_reductor]
    control_frequency = 120  # [Hz]

    r_q = [q0[i] + q_amps[i] * np.sin(2 * np.pi * freq[i] * 11) for i in range(7)]  # [rad]
    l_q = [r_q[0], -r_q[1], -r_q[2], r_q[3], -r_q[4], r_q[5], -r_q[6]]  # [rad]
    M_r = reachy.r_arm.forward_kinematics(r_q)
    M_l = symmetrical_pose_flip(M_r)

    reachy.r_arm.goto(M_r, interpolation_space="cartesian_space")
    reachy.l_arm.goto(M_l, interpolation_space="cartesian_space")
    time.sleep(2)

    t_init = time.time()

    start = False

    while time.time() - t_init < 22:
        t = time.time()
        t_sine = t - t_init + 11

        r_q = [q0[i] + q_amps[i] * np.sin(2 * np.pi * freq[i] * t_sine) for i in range(7)]  # [rad]
        l_q = [r_q[0], -r_q[1], -r_q[2], r_q[3], -r_q[4], r_q[5], -r_q[6]]  # [rad]
        M_r = reachy.r_arm.forward_kinematics(r_q)
        M_l = symmetrical_pose_flip(M_r)

        go_to_pose(reachy, M_r, "r_arm")
        go_to_pose(reachy, M_l, "l_arm")

        ik_r = r_q
        ik_l = l_q

        r_real_pose = reachy.r_arm.forward_kinematics()
        l_real_pose = reachy.l_arm.forward_kinematics()

        time.sleep(max(0, 1.0 / control_frequency - (time.time() - t)))

        assert np.linalg.norm(l_real_pose[:3, 3] - symmetrical_pose_flip(r_real_pose)[:3, 3]) < 5e-1

        assert np.linalg.norm(r_real_pose[:3, 3] - M_r[:3, 3]) < 7e-1
        assert np.linalg.norm(l_real_pose[:3, 3] - M_l[:3, 3]) < 7e-1
        assert np.linalg.norm(R.from_matrix(M_r[:3, :3] @ r_real_pose[:3, :3].T).as_rotvec()) < 4
        assert np.linalg.norm(R.from_matrix(M_l[:3, :3] @ l_real_pose[:3, :3].T).as_rotvec()) < 4

        if not start:
            start = True
        else:
            assert np.allclose(ik_l, l_previous_joints, atol=20)
            assert np.allclose(ik_r, r_previous_joints, atol=20)

        l_previous_joints = ik_l
        r_previous_joints = ik_r

    reachy.turn_off()
    assert True
