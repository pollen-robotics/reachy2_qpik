"""Pinocchio IK wrist motion test."""

import time

import numpy as np
from google.protobuf.wrappers_pb2 import FloatValue, Int32Value
from reachy2_sdk import ReachySDK
from reachy2_sdk_api.arm_pb2 import (
    ArmCartesianGoal,
    IKConstrainedMode,
    IKContinuousMode,
)
from reachy2_sdk_api.kinematics_pb2 import Matrix4x4
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def make_homogenous_matrix_from_rotation_matrix(rotation_matrix: np.ndarray, position: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = rotation_matrix
    M[:3, 3] = position
    return M


def go_to_pose(reachy: ReachySDK, pose: np.ndarray, arm: str) -> None:
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


def wrist_tilt_test(
    reachy: ReachySDK,
    arm: str = "r_arm",
    amp_deg: float = 45.0,
    n_cycles: int = 3,
    cycle_duration: float = 4.0,
    control_frequency: float = 120.0,
):
    q0 = np.array([0, 10, -10, -90, 0, 0, 0])
    M_target_r = np.array(reachy.r_arm.forward_kinematics(q0.tolist()))

    M_target_l = np.array(
        [
            [M_target_r[0, 0], -M_target_r[0, 1], M_target_r[0, 2], M_target_r[0, 3]],
            [-M_target_r[1, 0], M_target_r[1, 1], -M_target_r[1, 2], -M_target_r[1, 3]],
            [M_target_r[2, 0], -M_target_r[2, 1], M_target_r[2, 2], M_target_r[2, 3]],
            [0, 0, 0, 1],
        ]
    )

    reachy.r_arm.goto(M_target_r, interpolation_space="cartesian_space", duration=1.5)
    reachy.l_arm.goto(M_target_l, interpolation_space="cartesian_space", duration=1.5)
    time.sleep(1.7)

    base_pose = getattr(reachy, arm).forward_kinematics()
    R0 = base_pose[:3, :3]
    p0 = base_pose[:3, 3]

    amp_rad = np.deg2rad(amp_deg)
    rot0 = R.from_matrix(R0)
    Rx_fwd = R.from_euler("z", amp_rad)
    Rx_bwd = R.from_euler("z", -amp_rad)
    rot_fwd = rot0 * Rx_fwd
    rot_bwd = rot0 * Rx_bwd

    s = [0.0, 1.0]
    slerp_fwd = Slerp(s, R.concatenate([rot0, rot_fwd]))
    slerp_bwd = Slerp(s, R.concatenate([rot0, rot_bwd]))

    pts_per_half = int((cycle_duration / 2) * control_frequency)
    s_samples = np.linspace(0, 1, pts_per_half)
    seq = []
    for _ in range(n_cycles):
        for ss in s_samples:
            Rq = slerp_fwd([ss])[0].as_matrix()
            seq.append(make_homogenous_matrix_from_rotation_matrix(Rq, p0))

        for ss in s_samples:
            Rq = slerp_bwd([ss])[0].as_matrix()
            seq.append(make_homogenous_matrix_from_rotation_matrix(Rq, p0))

    final_pose = make_homogenous_matrix_from_rotation_matrix(R0, p0)
    seq.append(final_pose)

    dt = 1.0 / control_frequency
    for pose in seq:
        t0 = time.time()
        go_to_pose(reachy, pose, arm)
        time.sleep(max(dt - (time.time() - t0), 0.0))


if __name__ == "__main__":
    reachy = ReachySDK(host="localhost")
    if not reachy.is_connected:
        raise SystemExit("Reachy not connected")
    reachy.turn_on()
    time.sleep(0.5)

    print("Wrist tilt on right arm…")
    wrist_tilt_test(reachy, arm="r_arm", amp_deg=45, n_cycles=2, cycle_duration=3.0)
    time.sleep(1)

    print("…and now on left arm")
    wrist_tilt_test(reachy, arm="l_arm", amp_deg=-45, n_cycles=2, cycle_duration=3.0)
    time.sleep(1)

    reachy.turn_off()
