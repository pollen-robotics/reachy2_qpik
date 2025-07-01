"""Reachy2 Pinocchio Quadratic Programming IK class."""

import copy
import os
from typing import Optional

import numpy as np
import numpy.typing as npt
import pinocchio as pin
import qpsolvers
from numpy.linalg import norm


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
        self.model, self.data, self.q0 = robot.model, robot.data, robot.q0
        self.nv = self.model.nv

        self.arm = arm
        self.ee_frame = f"{arm}_tip"
        self.ee_frame_id = self.model.getFrameId(self.ee_frame)
        self.joint_id = self.model.frames[self.ee_frame_id].parent

        self.IT_MAX = 1  # 00
        self.eps = 1e-4  # Error precision (if IT_MAX >1)
        self.damp = 1e-6
        self.Kp = 0.4  # Proportional gain
        self.Ka = 1.0
        self.dt = 0.0025  # Time step
        self.W = np.diag([1.725] * 3 + [0.1] * 3)

        self.K_lim = 1.0
        self.q_min = self.model.lowerPositionLimit
        self.q_max = self.model.upperPositionLimit

        self.v_max = np.array([6.5] * 7)
        self.v_min = -self.v_max

        if arm == "l_arm":
            # self.q0_pref = [
            #     -0.26365717475226036,
            #     6.088962100244157,
            #     -11.43633214248595,
            #     -90.00000250447816,
            #     20.753570774218765,
            #     3.8409658443799564,
            #     20.753570774218765,
            # ]
            self.q0_pref = np.deg2rad([0, -10, 10, -90, 0, 0, 0])
        else:
            # self.q0_pref = [
            #     -0.26365717475226036,
            #     -6.088962100244157,
            #     11.43633214248595,
            #     -90.00000250447816,
            #     -20.753570774218765,
            #     3.8409658443799564,
            #     -20.753570774218765,
            # ]
            self.q0_pref = np.deg2rad([0, 10, -10, -90, 0, 0, 0])
        self.alpha = 1e-9

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

            q_dot_posture = (self.q0_pref - q) / self.dt

            if norm(err) < self.eps:
                success = True
                state = "Convergence reached."
                break

            J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL)

            # QP terms
            P = J.T @ self.W @ J + self.damp * np.eye(self.nv) + self.alpha * np.eye(self.nv)
            r = -J.T @ self.W @ v + -self.alpha * q_dot_posture

            G = np.vstack([np.eye(self.nv), -np.eye(self.nv)])
            h = np.hstack([self.v_max, -self.v_min])

            q_dot = qpsolvers.solve_qp(P, r, G, h, solver="quadprog")  # [rad.s⁻¹]
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

        return q, success, state

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

        # v_lim_upper = (self.q_max - q) / (self.K_lim * self.dt)
        # v_lim_lower = (self.q_min - q) / (self.K_lim * self.dt)

        # v_upper = np.minimum(self.v_max, v_lim_upper)
        # v_lower = np.maximum(self.v_min, v_lim_lower)

        q_dot_posture = (self.q0_pref - q) / self.dt

        J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL)

        # QP terms
        P = J.T @ self.W @ J + self.damp * np.eye(self.nv) + self.alpha * np.eye(self.nv)
        r = -J.T @ self.W @ v + -self.alpha * q_dot_posture

        G = np.vstack([np.eye(self.nv), -np.eye(self.nv)])
        # h = np.hstack([v_upper, -v_lower])
        h = np.hstack([self.v_max, -self.v_min])

        q_dot = qpsolvers.solve_qp(P, r, G, h, solver="quadprog")  # [rad.s⁻¹]

        return q_dot

    def compute_acceleration(
        self,
        goal_pose: npt.NDArray[np.float64],
        current_joints: npt.NDArray[np.float64],
        previous_joints: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], bool, str]:
        """Compute one IK acceleration step."""
        current_velocities = (current_joints - previous_joints) / self.dt  # [rad.s⁻¹]

        _, goal_pose, _ = self.is_pose_in_robot_reach(goal_pose)
        R_goal, p_goal = goal_pose[:3, :3], goal_pose[:3, 3]
        oMdes_tors = pin.SE3(R_goal, p_goal)

        q = current_joints.copy()  # [rad]
        pin.framesForwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        T_baselink_torso = self.data.oMf[self.model.getFrameId("torso")]
        oMdes = T_baselink_torso * oMdes_tors

        J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL)
        Jdot = pin.computeFrameJacobianTimeVariation(
            self.model, self.data, q, current_velocities, self.ee_frame_id, pin.ReferenceFrame.LOCAL
        )

        current_ee = self.data.oMf[self.ee_frame_id]
        iMd = current_ee.actInv(oMdes)
        err = pin.log(iMd).vector  # [m, m, m, rad, rad, rad]

        v_des = self.Kp * (err / self.dt)  # [m.s⁻¹, m.s⁻¹, m.s⁻¹, rad.s⁻¹, rad.s⁻¹, rad.s⁻¹]
        v_cur = J.dot(current_velocities)  # [m.s⁻¹, m.s⁻¹, m.s⁻¹, rad.s⁻¹, rad.s⁻¹, rad.s⁻¹]
        a_des = self.Ka * (v_des - v_cur) / self.dt  # [m.s⁻², m.s⁻², m.s⁻², rad.s⁻², rad.s⁻², rad.s⁻²]

        e_a = a_des - Jdot.dot(current_velocities)
        P = J.T @ self.W @ J + self.damp * np.eye(self.nv)
        r = -2 * J.T @ self.W @ e_a

        a_max = (self.v_max - current_velocities) / (self.K_lim * self.dt)
        a_min = (self.v_min - current_velocities) / (self.K_lim * self.dt)
        G = np.vstack([np.eye(self.nv), -np.eye(self.nv)])
        h = np.hstack([a_max, -a_min])

        q_ddot = qpsolvers.solve_qp(P, r, G, h, solver="quadprog")  # [rad.s⁻²]

        return q_ddot
