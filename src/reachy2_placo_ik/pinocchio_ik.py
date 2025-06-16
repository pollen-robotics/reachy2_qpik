"""Reachy2 Pinocchio Inverse Kinematics class."""

import copy
import os
from typing import Optional

import numpy as np
import numpy.typing as npt
import pinocchio as pin
from numpy.linalg import norm, solve


class PinocchioIK:
    """Pinocchio IK class for Reachy2."""

    def __init__(
        self,
        urdf_path: str,
        arm: str = "l_arm",
        locked_joints: Optional[list[str]] = None,
    ) -> None:
        """Initialize the class."""
        robot = pin.RobotWrapper.BuildFromURDF(urdf_path, os.path.dirname(urdf_path))

        if locked_joints is None:
            locked_joints = self.default_locked_joints(arm)

        robot = robot.buildReducedRobot(locked_joints)

        self.robot = robot
        self.model, self.data = robot.model, robot.data

        self.arm = arm
        self.ee_frame = f"{arm}_tip"
        self.ee_frame_id = self.model.getFrameId(self.ee_frame)
        self.joint_id = self.model.frames[self.ee_frame_id].parent

        self.IT_MAX = 1  # 00
        self.eps = 1e-4  # Error precision (if IT_MAX >1)
        self.damp = 7.5e-4  # Damping factor
        self.Kp = 0.05  # Proportional gain
        self.dt = 0.0025  # Time step

    def default_locked_joints(self, arm: str) -> list[str]:
        """List of the default joints to lock before computation."""
        shared_joints = [
            "tripod_joint",
            "l_hand_finger",
            "l_hand_finger_proximal",
            "l_hand_finger_distal",
            "l_hand_finger_proximal_mimic",
            "l_hand_finger_distal_mimic",
            "left_bar_joint_mimic",
            "left_bar_prism_joint_mimic",
            "neck_roll",
            "neck_pitch",
            "neck_yaw",
            "antenna_left",
            "antenna_right",
            "r_hand_finger",
            "r_hand_finger_proximal",
            "r_hand_finger_distal",
            "r_hand_finger_proximal_mimic",
            "r_hand_finger_distal_mimic",
            "right_bar_joint_mimic",
            "right_bar_prism_joint_mimic",
        ]

        r_arm_joints = [
            "r_shoulder_pitch",
            "r_shoulder_roll",
            "r_elbow_yaw",
            "r_elbow_pitch",
            "r_wrist_roll",
            "r_wrist_pitch",
            "r_wrist_yaw",
        ]

        l_arm_joints = [
            "l_shoulder_pitch",
            "l_shoulder_roll",
            "l_elbow_yaw",
            "l_elbow_pitch",
            "l_wrist_roll",
            "l_wrist_pitch",
            "l_wrist_yaw",
        ]

        if arm == "r_arm":
            return shared_joints + l_arm_joints
        elif arm == "l_arm":
            return shared_joints + r_arm_joints
        else:
            return shared_joints

    def is_pose_in_robot_reach(self, goal_pose: npt.NDArray[np.float64]) -> tuple[bool, npt.NDArray[np.float64], str]:
        """Reduce the goal pose if it's out of reach and prevent backward tip."""
        goal_pose = copy.deepcopy(goal_pose)
        goal_position: npt.NDArray[np.float64] = np.array(goal_pose[:3, 3], dtype=np.float64)

        ik_params: dict = {
            "r_arm_shoulder_position": np.array([0.0, -0.2, 0.0], dtype=np.float64),
            "l_arm_shoulder_position": np.array([0.0, 0.2, 0.0], dtype=np.float64),
            "max_arm_length": 0.60,
            "backward_limit": 0.0,
        }

        shoulder: npt.NDArray[np.float64] = ik_params[f"{self.arm}_shoulder_position"]
        max_arm_length: float = float(ik_params["max_arm_length"])
        backward_limit: float = float(ik_params["backward_limit"])

        is_reachable: bool = True
        state: str = ""

        vec = goal_position - shoulder
        dist = norm(vec)
        if dist > max_arm_length:
            is_reachable = False
            direction = vec / (dist + 1e-9)
            goal_position = shoulder + direction * max_arm_length
            state = "Pose out of reach"

        if goal_position[0] < backward_limit:
            is_reachable = False
            goal_position[0] = backward_limit
            state = state or "Backward pose"

        goal_pose[:3, 3] = goal_position
        return is_reachable, goal_pose, state

    def inverse_kinematics(
        self, goal_pose: npt.NDArray[np.float64], current_joints: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], bool, str]:
        """Get the joints from the Inverse Kinematics."""
        _, goal_pose, _ = self.is_pose_in_robot_reach(goal_pose)
        R_goal = goal_pose[:3, :3]
        p_goal = goal_pose[:3, 3]
        oMdes_torso = pin.SE3(R_goal, p_goal)

        q_neutral = pin.neutral(self.model)  # [rad]

        pin.framesForwardKinematics(self.model, self.data, q_neutral)
        pin.updateFramePlacements(self.model, self.data)

        T_baselink_torso = self.data.oMf[self.model.getFrameId("torso")].copy()
        oMdes = T_baselink_torso * oMdes_torso

        if current_joints is None:
            q = q_neutral
        else:
            q = current_joints.copy()

        success = False
        state = ""

        i = 0
        while i < self.IT_MAX:
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            current_ee = self.data.oMf[self.ee_frame_id]
            iMd = current_ee.actInv(oMdes)
            err = pin.log(iMd).vector  # [m, m, m, rad, rad, rad]

            v = self.Kp * (err / self.dt)  # [m.s⁻¹, m.s⁻¹, m.s⁻¹, rad.s⁻¹, rad.s⁻¹, rad.s⁻¹]

            if norm(err) < self.eps:
                success = True
                state = "Convergence reached."
                break

            J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)

            # Closed-Loop Inverse Kinematics
            q_dot = -J.T.dot(solve(J.dot(J.T) + self.damp * np.eye(6), v))  # [rad.s⁻¹]
            q = pin.integrate(self.model, q, q_dot * self.dt)

            if not success:
                state = "Convergence not reached."

            i += 1

        # Forward Kinematics obtained by Pinocchio
        # print(self.arm)
        # pin.forwardKinematics(self.model, self.data, q)
        # pin.updateFramePlacements(self.model, self.data)
        # final_ee_pose = T_baselink_torso.inverse() * self.data.oMf[self.ee_frame_id]
        # print(final_ee_pose)

        return q, True, state

    def compute_velocity(
        self, goal_pose: npt.NDArray[np.float64], current_joints: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute one IK velocity step."""
        _, goal_pose, _ = self.is_pose_in_robot_reach(goal_pose)
        R_goal = goal_pose[:3, :3]
        p_goal = goal_pose[:3, 3]
        oMdes_torso = pin.SE3(R_goal, p_goal)

        q = current_joints.copy()  # [rad]
        pin.framesForwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        T_baselink_torso = self.data.oMf[self.model.getFrameId("torso")].copy()
        oMdes = T_baselink_torso * oMdes_torso

        current_ee = self.data.oMf[self.ee_frame_id]
        iMd = current_ee.actInv(oMdes)
        err = pin.log(iMd).vector  # [m, m, m, rad, rad, rad]

        v = self.Kp * (err / self.dt)  # [m.s⁻¹, m.s⁻¹, m.s⁻¹, rad.s⁻¹, rad.s⁻¹, rad.s⁻¹]
        J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL)
        J = -np.dot(pin.Jlog6(iMd.inverse()), J)

        q_dot = -J.T.dot(solve(J.dot(J.T) + self.damp * np.eye(6), v))  # [rad.s⁻¹]

        return q_dot
