"""Pinocchio IK random motion test."""

import time

import numpy as np
import numpy.typing as npt
from google.protobuf.wrappers_pb2 import FloatValue, Int32Value
from metrics import combined_error, l2_error, rodrigues_error
from reachy2_sdk import ReachySDK
from reachy2_sdk_api.arm_pb2 import (
    ArmCartesianGoal,
    IKConstrainedMode,
    IKContinuousMode,
)
from reachy2_sdk_api.kinematics_pb2 import Matrix4x4
from scipy.spatial.transform import Rotation as R


def go_to_pose(reachy: ReachySDK, pose: npt.NDArray[np.float64], arm: str) -> None:
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


def get_homogeneous_matrix_msg_from_euler(
    position: npt.NDArray[np.float64] = np.array([0.0, 0.0, 0.0]),  # (x, y, z)
    euler_angles: npt.NDArray[np.float64] = np.array([0.0, 0.0, 0.0]),  # (roll, pitch, yaw)
    degrees: bool = False,
) -> npt.NDArray[np.float64]:
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = R.from_euler("xyz", euler_angles, degrees=degrees).as_matrix()
    homogeneous_matrix[:3, 3] = position
    return homogeneous_matrix


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


def random_trajectory(reachy: ReachySDK, debug_pose: bool = False, bypass: bool = False) -> None:
    q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [rad]
    ik_r = q
    ik_l = q
    q0 = [-90.0, -80.0, 0.0, -65.0, 0.0, 0.0, 0.0]  # [rad]
    q_amps = [90.0, 90.0, 180.0, 65.0, 45.0, 45.0, 30.0]  # [rad]
    previous_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [rad]

    start = True

    freq_reductor = 0.3
    freq = [0.3 * freq_reductor, 0.17 * freq_reductor, 0.39 * freq_reductor, 0.18, 0.15, 0.15, 0.15 * freq_reductor]
    control_frequency = 120  # [Hz]

    t_init = time.time()
    while True:
        t = time.time()
        t_sine = t - t_init + 11
        if not debug_pose:
            r_q = [q0[i] + q_amps[i] * np.sin(2 * np.pi * freq[i] * t_sine) for i in range(7)]  # [rad]
        else:
            # Precision problem
            r_q = [
                -41.03096373192293,
                -37.647921777704,
                -29.585523143459014,
                -88.24667866025105,
                -21.052896284656175,
                9.808669854062696,
                -89.89911806297954,
            ]

        l_q = [r_q[0], -r_q[1], -r_q[2], r_q[3], -r_q[4], r_q[5], -r_q[6]]  # [rad]
        M_r = reachy.r_arm.forward_kinematics(r_q)
        M_l = np.array(
            [
                [M_r[0][0], -M_r[0][1], M_r[0][2], M_r[0][3]],
                [-M_r[1][0], M_r[1][1], -M_r[1][2], -M_r[1][3]],
                [M_r[2][0], -M_r[2][1], M_r[2][2], M_r[2][3]],
                [0, 0, 0, 1],
            ]
        )

        go_to_pose(reachy, M_r, "r_arm")
        go_to_pose(reachy, M_l, "l_arm")

        ik_r = r_q
        ik_l = l_q

        r_real_pose = reachy.r_arm.forward_kinematics()
        l_real_pose = reachy.l_arm.forward_kinematics()

        is_real_pose_correct = check_precision_and_symmetry(
            reachy,
            M_r,
            M_l,
            r_real_pose,
            l_real_pose,
            ik_r,
            ik_l,
            previous_joints,
            start,
        )

        previous_joints = ik_r
        start = False

        if not is_real_pose_correct:
            break

        # print(f"ik_r: {ik_r}, ik_l: {ik_l}, time_r: {t1-t0}, time_l: {t2-t1}")
        # print(f"Loop time: {(time.time() - t)*1000:.1f} ms")
        time.sleep(max(0, 1.0 / control_frequency - (time.time() - t)))


def check_precision_and_symmetry(
    reachy: ReachySDK,
    M_r: npt.NDArray[np.float64],
    M_l: npt.NDArray[np.float64],
    r_real_pose: npt.NDArray[np.float64],
    l_real_pose: npt.NDArray[np.float64],
    ik_r: list[float],
    ik_l: list[float],
    previous_joints: list[float],
    start: bool,
) -> bool:
    is_real_pose_correct = True

    l_mod = np.array([ik_l[0], -ik_l[1], -ik_l[2], ik_l[3], -ik_l[4], ik_l[5], -ik_l[6]])

    # calculate l2 distance between r_joints and l_mod
    l2_dist = l2_error(ik_r, l_mod)
    print(f"l2_dist: {l2_dist}")

    l_position_diff = l2_error(l_real_pose[:3, 3], M_l[:3, 3])
    r_position_diff = l2_error(r_real_pose[:3, 3], M_r[:3, 3])
    print(f"l_position_diff: {l_position_diff:.3f} m")
    print(f"r_position_diff: {r_position_diff:.3f} m")

    r_rodrigues_err = rodrigues_error(M_r[:3, :3], r_real_pose[:3, :3])
    l_rodrigues_err = rodrigues_error(M_l[:3, :3], l_real_pose[:3, :3])
    print(f"l_rotation_err: {np.rad2deg(l_rodrigues_err):.4f}°")
    print(f"r_rotation_err: {np.rad2deg(r_rodrigues_err):.4f}°")

    r_combined_err = combined_error(r_position_diff, r_rodrigues_err)
    l_combined_err = combined_error(l_position_diff, l_rodrigues_err)
    print(f"l_combined_err: {l_combined_err:.4f}")
    print(f"r_combined_err: {r_combined_err:.4f}")

    if not start:
        if np.allclose(ik_r, previous_joints, atol=40):
            print("Continuity OK")
        else:
            print("Continuity NOT OK!!")
            print(f"previous_joints {np.round(previous_joints, 3).tolist()}")
            print(f"ik_r {np.round(ik_r, 3)}")
            print(f"ik_l {np.round(ik_l, 3)}")
            print(f"r_real_pose {r_real_pose.tolist()}")
            print(f"l_real_pose {l_real_pose.tolist()}")
            is_real_pose_correct = False

    if l2_dist < 0.1:
        print("Symmetry OK")
    else:
        print("Symmetry NOT OK!!")
        print(f"ik_r {np.round(ik_r, 3).tolist()}")
        print(f"ik_l_sym {np.round(l_mod, 3).tolist()}")
        print(f"M_r {M_r.tolist()}")
        print(f"M_l {M_l.tolist()}")
        is_real_pose_correct = False
    print("_____________________")
    return is_real_pose_correct


def main() -> None:
    print("Trying to connect on localhost Reachy...")
    time.sleep(1.0)
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()
    print("Putting each joint at 0 degrees angle")
    time.sleep(0.5)
    for joint in reachy.joints.values():
        joint.goal_position = 0
    reachy.send_goal_positions()
    time.sleep(1.0)

    random_trajectory(reachy, debug_pose=False, bypass=False)

    print("Finished testing, disconnecting from Reachy...")
    time.sleep(0.5)
    reachy.disconnect()


if __name__ == "__main__":
    main()
