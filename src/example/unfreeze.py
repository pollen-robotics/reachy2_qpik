"""Unfreeze Reachy2 function."""

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


def main_test() -> None:
    """Main test to unfreeze the robot when stuck."""
    print("Trying to connect on localhost Reachy...")
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    pose = reachy.r_arm.forward_kinematics()
    request = ArmCartesianGoal(
        id=reachy.r_arm._part_id,
        goal_pose=Matrix4x4(data=pose.flatten().tolist()),
        continuous_mode=IKContinuousMode.UNFREEZE,
        constrained_mode=IKConstrainedMode.UNCONSTRAINED,
        preferred_theta=FloatValue(
            value=-4 * np.pi / 6,
        ),
        d_theta_max=FloatValue(value=0.05),
        order_id=Int32Value(value=5),
    )
    reachy.r_arm._stub.SendArmCartesianGoal(request)


if __name__ == "__main__":
    main_test()
