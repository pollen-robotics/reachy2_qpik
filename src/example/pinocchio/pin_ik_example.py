"""Pinocchio Inverse Kinematics example."""

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

from reachy2_placo_ik.pinocchio_ik import PinocchioIK


def main() -> None:
    """Run the main function."""
    urdf_path = ...
    # urdf_path = r"/home/reachy/dev/reachy_urdf/reachy.urdf"

    pinik = PinocchioIK(urdf_path=urdf_path, arm="l_arm")

    rotation_matrix = R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()
    position = np.array([9.98901949e-03, -2.56649267e-01, -6.57488464e-01])

    goal_pose = np.eye(4)
    goal_pose[:3, :3] = rotation_matrix
    goal_pose[:3, 3] = position

    current_joints = pin.neutral(pinik.model)

    for _ in range(10):
        sol, is_reachable, state = pinik.inverse_kinematics(goal_pose=goal_pose, current_joints=current_joints)
        current_joints = sol

    print(f"IK Solution: {sol}")
    print(f"Reachability: {is_reachable}")
    print(f"Convergence state: {state}")


if __name__ == "__main__":
    main()
