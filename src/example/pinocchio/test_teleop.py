import json
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


def go_to_pose(reachy, pose: np.ndarray, arm: str):
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


JSON_FILE = "pytest1.json"


def load_trajectory(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["time"], data["l_arm"], data["r_arm"]


def main():
    control_frequency = 200.0
    dt = 1 / control_frequency
    reachy = ReachySDK(host="localhost")
    time.sleep(1.0)
    assert reachy.is_connected, "Could not connect to Reachy"
    reachy.turn_on()

    times, l_traj_deg, r_traj_deg = load_trajectory(JSON_FILE)

    for _, l_deg, r_deg in zip(times, l_traj_deg, r_traj_deg):
        t = time.time()

        T_l = reachy.l_arm.forward_kinematics(l_deg)
        T_r = reachy.r_arm.forward_kinematics(r_deg)

        go_to_pose(reachy, T_l, "l_arm")
        go_to_pose(reachy, T_r, "r_arm")

        time.sleep(max(dt - (time.time() - t), 0.0))

    print("Playback complete.")


if __name__ == "__main__":
    main()
