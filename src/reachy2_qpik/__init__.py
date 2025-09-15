"""
Reachy2 Quadratic Programming Inverse Kinematics (QPIK) package.

A kinematics library for Reachy2 7 DoF arms, using quadratic programming
for precise and robust motion control. Provides task-space control algorithms
for acceleration and speed with joint limits and singularity handling.

Key Features:
    - Quadratic Programming Inverse Kinematics:
        * Solves IK by minimizing joint accelerations under task-space constraints.
        * Handles accelerations, speed, and joint limits.
        * Robust against singularities.
    - Task-Space Control Algorithm:
        * Pose tracking suitable for teleoperation.
        * Graceful handling of unreachable poses.
        * Configurable workspace and parameters.

Submodules:
    - `pinocchio_ik`: Classes for inverse kinematics using Pinocchio.
    - `pinocchio_control`: Acceleration control loop for IK.
    - `pinocchio_speed_control`: Speed control loop for IK.
    - `utils`: Utility functions for joint limits, Savitzky-Golay filtering, and multiturn safety checks.

Example:
    >>> import reachy2_qpik
    >>> from reachy2_qpik.pinocchio_ik import Reachy2QPIK
    >>> ik_solver = Reachy2QPIK("src/config_files/reachy.urdf", arm="l_arm")
    >>> target_pose = ...  # some SE3 target
    >>> q, success, state = ik_solver.inverse_kinematics(target_pose, current_joints=None)

License:
    Apache 2.0 License. See the LICENSE file for details.
"""
