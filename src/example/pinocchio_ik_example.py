"""Pinocchio Inverse Kinematics example."""

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

from reachy2_qpik.pinocchio_ik import PinocchioIK


def main() -> None:
    """Run the main function."""
    urdf_path = r"../config_files/reachy.urdf"

    pinik_l = PinocchioIK(urdf_path=urdf_path, arm="l_arm")
    pinik_r = PinocchioIK(urdf_path=urdf_path, arm="r_arm")

    rotation_matrix = R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()
    position = np.array([9.98901949e-03, -2.56649267e-01, -6.57488464e-01])

    goal_pose = np.eye(4)
    goal_pose[:3, :3] = rotation_matrix
    goal_pose[:3, 3] = position

    current_joints = pin.neutral(pinik_l.model)

    sol, is_reachable, state = pinik_l.inverse_kinematics(goal_pose=goal_pose, current_joints=current_joints)
    current_joints = sol

    print(f"Left IK Solution: {sol}")

    rotation_matrix = R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()
    position = np.array([9.98901949e-03, 2.56649267e-01, -6.57488464e-01])

    goal_pose = np.eye(4)
    goal_pose[:3, :3] = rotation_matrix
    goal_pose[:3, 3] = position

    current_joints = pin.neutral(pinik_r.model)

    sol, is_reachable, state = pinik_r.inverse_kinematics(goal_pose=goal_pose, current_joints=current_joints)
    current_joints = sol

    print(f"Right IK Solution: {sol}")


if __name__ == "__main__":
    main()
