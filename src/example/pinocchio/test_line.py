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


# def go_to_pose(reachy: ReachySDK, pose: npt.NDArray[np.float64], arm: str) -> None:
#     """Send a Cartesian goal to the specified arm."""
#     req = ArmCartesianGoal(
#         id=getattr(reachy, arm)._part_id,
#         goal_pose=Matrix4x4(data=pose.flatten().tolist()),
#         continuous_mode=IKContinuousMode.CONTINUOUS,
#         constrained_mode=IKConstrainedMode.UNCONSTRAINED,
#         preferred_theta=FloatValue(value=-4 * np.pi / 6),
#         d_theta_max=FloatValue(value=0.05),
#         order_id=Int32Value(value=5),
#     )
#     stub = getattr(reachy, arm)._stub
#     stub.SendArmCartesianGoal(req)


def go_to_pose(reachy: ReachySDK, pose: npt.NDArray[np.float64], arm: str) -> None:
    """Build a 4x4 pose matrix for a given position in 3D space, with the effector at a fixed orientation.

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


def make_line(
    reachy: ReachySDK, start_pose: npt.NDArray[np.float64], end_pose: np.ndarray[np.float64], duration: float = 2.0
) -> None:
    start_position = start_pose[0]
    end_position = end_pose[0]
    start_orientation = start_pose[1]
    end_orientation = end_pose[1]

    control_frequency = 500.0
    dt = 1.0 / control_frequency
    nbr_points = int(duration * control_frequency)

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

        # r_real_pose = reachy.r_arm.forward_kinematics()
        # l_real_pose = reachy.l_arm.forward_kinematics()

        # compute_metrics(r_pose, l_pose, r_real_pose, l_real_pose)
        print(f"Loop time: {(time.time() - t)*1000:.1f}")
        time.sleep(max(dt - (time.time() - t), 0.0))


def compute_metrics(M_r, M_l, r_real_pose, l_real_pose):
    r_ep = l2_error(M_r[:3, 3], r_real_pose[:3, 3])
    l_ep = l2_error(M_l[:3, 3], l_real_pose[:3, 3])

    r_etheta = rodrigues_error(M_r[:3, :3], r_real_pose[:3, :3])
    l_etheta = rodrigues_error(M_l[:3, :3], l_real_pose[:3, :3])

    l_combined = combined_error(r_ep, r_etheta)
    r_combined = combined_error(l_ep, l_etheta)

    print(f"Right arm - pos error: {r_ep:.4f}, rot error: {r_etheta:.4f}, combined: {r_combined:.4f}")
    print(f"Left arm  - pos error: {l_ep:.4f}, rot error: {l_etheta:.4f}, combined: {l_combined:.4f}")
    print("_" * 20)


def main() -> None:
    print("Trying to connect on localhost Reachy...")
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()

    ik = [0.0, 15.0, -10.0, 0.0, 0.0, 0.0, 0.0]
    reachy.r_arm.goto(ik, 3.0, degrees=True, interpolation_mode="minimum_jerk")
    ik = [0.0, -15.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    reachy.l_arm.goto(ik, 3.0, degrees=True, interpolation_mode="minimum_jerk")
    time.sleep(5.0)

    print("Making a line")
    start_pose = np.array([[0.3, -0.2, -0.6599], [0, 0, 0]])
    end_pose = np.array([[0.3, -0.2, 0.50], [0, -np.pi, 0]])
    make_line(reachy, start_pose, end_pose)

    time.sleep(2)

    reachy.goto_posture("default", wait=True)
    reachy.turn_off()


if __name__ == "__main__":
    main()
