"""Pinocchio Inverse Kinematics example."""

import numpy as np
import numpy.typing as npt
import pinocchio as pin
from numpy.linalg import norm, solve
from scipy.spatial.transform import Rotation as R

from reachy2_placo_ik.pinocchio_ik import PinocchioIK


def main():
    urdf_path = ...

    pinik = PinocchioIK(urdf_path=urdf_path, arm="l_arm")

    rotation_matrix = R.from_euler("xyz", [-20, -90, 0], degrees=True).as_matrix()
    position = np.array([0.3, 0.3, 0.0])

    goal_pose = np.eye(4)
    goal_pose[:3, :3] = rotation_matrix
    goal_pose[:3, 3] = position

    current_joints = pin.neutral(pinik.model)

    sol, is_reachable, state = pinik.inverse_kinematics(goal_pose=goal_pose, current_joints=current_joints)

    print(f"IK Solution: {sol}")
    print(f"Reachability: {is_reachable}")
    print(f"Convergence state: {state}")


if __name__ == "__main__":
    main()
