import time

import numpy as np
from google.protobuf.wrappers_pb2 import FloatValue, Int32Value
from metrics import combined_error, l2_error, rodrigues_error, velocity_error
from reachy2_sdk import ReachySDK
from reachy2_sdk_api.arm_pb2 import (
    ArmCartesianGoal,
    IKConstrainedMode,
    IKContinuousMode,
)
from reachy2_sdk_api.kinematics_pb2 import Matrix4x4
from scipy.spatial.transform import Rotation as R


def make_homogenous_matrix_from_rotation_matrix(position: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix + position into a 4x4 homogeneous matrix."""
    mat = np.eye(4)
    mat[:3, :3] = rotation_matrix
    mat[:3, 3] = position
    return mat


def go_to_pose(reachy: ReachySDK, pose: np.ndarray, arm: str) -> None:
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
    reachy: ReachySDK, start_pose: np.ndarray, end_pose: np.ndarray, duration: float = 10.0, control_frequency: float = 100.0
) -> None:
    """
    Move both arms in a straight-line Cartesian path from start_pose to end_pose,
    while logging error metrics at each step.

    start_pose: [[x,y,z], [roll,pitch,yaw]]
    end_pose:   [[x,y,z], [roll,pitch,yaw]]
    duration: seconds
    control_frequency: Hz
    """
    # Unpack positions and orientations
    p0, ori0 = start_pose
    p1, ori1 = end_pose

    nbr = int(duration * control_frequency)
    dt = 1.0 / control_frequency

    # Mirror for left arm
    p0_l = np.array([p0[0], -p0[1], p0[2]])
    p1_l = np.array([p1[0], -p1[1], p1[2]])
    ori0_l = np.array([-ori0[0], ori0[1], -ori0[2]])
    ori1_l = np.array([-ori1[0], ori1[1], -ori1[2]])

    # Previous transforms for velocity error
    prev_M_r = None
    prev_r_real = None
    prev_M_l = None
    prev_l_real = None

    for i in range(nbr + 1):
        t0 = time.time()
        alpha = i / nbr

        # Interpolate right arm
        pos_r = p0 + (p1 - p0) * alpha
        ang_r = ori0 + (ori1 - ori0) * alpha
        R_r = R.from_euler("xyz", ang_r).as_matrix()
        M_r = make_homogenous_matrix_from_rotation_matrix(pos_r, R_r)
        go_to_pose(reachy, M_r, "r_arm")

        # Interpolate left arm
        pos_l = p0_l + (p1_l - p0_l) * alpha
        ang_l = ori0_l + (ori1_l - ori0_l) * alpha
        R_l = R.from_euler("xyz", ang_l).as_matrix()
        M_l = make_homogenous_matrix_from_rotation_matrix(pos_l, R_l)
        go_to_pose(reachy, M_l, "l_arm")

        # Maintain control rate
        elapsed = time.time() - t0
        time.sleep(max(dt - elapsed, 0.0))

        # Read real poses
        r_real = reachy.r_arm.forward_kinematics()
        l_real = reachy.l_arm.forward_kinematics()

        # --- Compute metrics ---
        # Position and orientation errors
        epr = l2_error(M_r[:3, 3], r_real[:3, 3])
        epl = l2_error(M_l[:3, 3], l_real[:3, 3])
        err_r = rodrigues_error(M_r[:3, :3], r_real[:3, :3])
        err_l = rodrigues_error(M_l[:3, :3], l_real[:3, :3])
        comb_r = combined_error(epr, err_r)
        comb_l = combined_error(epl, err_l)
        print(f"Step {i}/{nbr} | PosErr R: {epr:.4f}, RotErr R: {err_r:.4f}, Comb R: {comb_r:.4f}")
        print(f"           | PosErr L: {epl:.4f}, RotErr L: {err_l:.4f}, Comb L: {comb_l:.4f}")

        # Velocity errors
        if prev_M_r is not None:
            v_err_r, _ = velocity_error(prev_M_r[:3, 3], prev_r_real[:3, 3], M_r[:3, 3], r_real[:3, 3], dt)
            v_err_l, _ = velocity_error(prev_M_l[:3, 3], prev_l_real[:3, 3], M_l[:3, 3], l_real[:3, 3], dt)
            print(f"           | VelErr R: {v_err_r:.4f}, VelErr L: {v_err_l:.4f}")

        # Update previous
        prev_M_r, prev_r_real = M_r.copy(), r_real.copy()
        prev_M_l, prev_l_real = M_l.copy(), l_real.copy()
        print("---")

    print("Completed straight-line motion with metrics.")


def main() -> None:
    print("Connecting to Reachy…")
    reachy = ReachySDK(host="localhost")
    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy.")
        return
    reachy.turn_on()

    # Move arms to start posture
    reachy.r_arm.goto([0, 15, -10, 0, 0, 0, 0], 3.0, degrees=True, interpolation_mode="minimum_jerk")
    reachy.l_arm.goto([0, -15, 10, 0, 0, 0, 0], 3.0, degrees=True, interpolation_mode="minimum_jerk")
    time.sleep(5.0)

    # Define line in Cartesian space
    start = np.array([0.3, -0.2, -0.6599]), np.array([0.0, 0.0, 0.0])
    end = np.array([0.3, -0.2, 0.50]), np.array([0.0, -np.pi, 0.0])

    print("Starting straight-line motion with metrics…")
    make_line(reachy, start, end, duration=10.0, control_frequency=100.0)

    # Return to default posture and shutdown
    reachy.goto_posture("default", wait=True)
    reachy.turn_off()


if __name__ == "__main__":
    main()
