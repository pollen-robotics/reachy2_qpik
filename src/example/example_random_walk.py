"""Pinocchio IK random walk example."""

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
    """Send the IK goal to the specified arm."""
    request = ArmCartesianGoal(
        id=getattr(reachy, arm)._part_id,
        goal_pose=Matrix4x4(data=pose.flatten().tolist()),
        continuous_mode=IKContinuousMode.CONTINUOUS,
        constrained_mode=IKConstrainedMode.UNCONSTRAINED,
        preferred_theta=FloatValue(value=-4 * np.pi / 6),
        d_theta_max=FloatValue(value=0.05),
        order_id=Int32Value(value=5),
    )
    getattr(reachy, arm)._stub.SendArmCartesianGoal(request)


def random_walk_test(
    reachy: ReachySDK,
    n_steps: int = 100,
    step_radius: float = 0.05,
    angle_radius_deg: float = 5.0,
    control_frequency: float = 120.0,
) -> dict[str, np.ndarray]:
    """Perform a random walk."""
    JOINT_LIMITS_DEG = np.array([90.0, 90.0, 180.0, 65.0, 45.0, 45.0, 30.0])

    metrics: dict = {
        "l2_dist": [],
        "r_pos_err": [],
        "l_pos_err": [],
        "r_rot_err": [],
        "l_rot_err": [],
        "r_comb_err": [],
        "l_comb_err": [],
    }

    q0 = np.random.uniform(-JOINT_LIMITS_DEG, JOINT_LIMITS_DEG)
    q0 = np.array([0, 10, -10, -90, 0, 0, 0])
    M_prev_r = np.array(reachy.r_arm.forward_kinematics(q0.tolist()))
    dt = 1 / control_frequency

    for _ in range(n_steps):
        t = time.time()
        delta_trans = np.random.normal(size=3)
        delta_trans = delta_trans / np.linalg.norm(delta_trans) * np.random.uniform(0, step_radius)

        axis = np.random.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = np.deg2rad(np.random.uniform(-angle_radius_deg, angle_radius_deg))
        delta_rot = R.from_rotvec(axis * angle).as_matrix()

        M_target_r = M_prev_r.copy()
        M_target_r[:3, :3] = delta_rot @ M_target_r[:3, :3]
        M_target_r[:3, 3] += delta_trans

        M_target_l = np.array(
            [
                [M_target_r[0, 0], -M_target_r[0, 1], M_target_r[0, 2], M_target_r[0, 3]],
                [-M_target_r[1, 0], M_target_r[1, 1], -M_target_r[1, 2], -M_target_r[1, 3]],
                [M_target_r[2, 0], -M_target_r[2, 1], M_target_r[2, 2], M_target_r[2, 3]],
                [0, 0, 0, 1],
            ]
        )

        go_to_pose(reachy, M_target_r, "r_arm")
        go_to_pose(reachy, M_target_l, "l_arm")

        # time.sleep(0.05)

        r_real = np.array(reachy.r_arm.forward_kinematics())
        l_real = np.array(reachy.l_arm.forward_kinematics())

        l_real_flip = np.array(
            [
                [l_real[0, 0], -l_real[0, 1], l_real[0, 2], l_real[0, 3]],
                [-l_real[1, 0], l_real[1, 1], -l_real[1, 2], -l_real[1, 3]],
                [l_real[2, 0], -l_real[2, 1], l_real[2, 2], l_real[2, 3]],
                [0, 0, 0, 1],
            ]
        )

        l2 = l2_error(r_real[:3, 3], l_real_flip[:3, 3])
        r_pos = l2_error(r_real[:3, 3], M_target_r[:3, 3])
        l_pos = l2_error(l_real[:3, 3], M_target_l[:3, 3])
        r_rot = rodrigues_error(M_target_r[:3, :3], r_real[:3, :3])
        l_rot = rodrigues_error(M_target_l[:3, :3], l_real[:3, :3])
        r_comb = combined_error(r_pos, r_rot)
        l_comb = combined_error(l_pos, l_rot)

        metrics["l2_dist"].append(l2)
        metrics["r_pos_err"].append(r_pos)
        metrics["l_pos_err"].append(l_pos)
        metrics["r_rot_err"].append(r_rot)
        metrics["l_rot_err"].append(l_rot)
        metrics["r_comb_err"].append(r_comb)
        metrics["l_comb_err"].append(l_comb)

        M_prev_r = M_target_r
        time.sleep(max(dt - (time.time() - t), 0.0))

    stats = {}
    for key, vals in metrics.items():
        arr = np.array(vals)
        stats[f"{key}_mean"] = arr.mean()
        stats[f"{key}_std"] = arr.std()
    return stats


def main():
    """Main function."""
    reachy = ReachySDK(host="localhost")
    if reachy._grpc_status != "connected":
        raise RuntimeError("Cannot connect to Reachy")
    reachy.turn_on()

    results = random_walk_test(
        reachy,
        n_steps=1000,
        step_radius=0.01,
        angle_radius_deg=2.5,
    )

    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    reachy.disconnect()


if __name__ == "__main__":
    main()
