"""Position Unit Step example with selectable plotting via flags.

  --uni  : Cartesian unit step only
  --pos  : Joint positions only
  --spe  : Joint speeds only
  --acc  : Joint accelerations only
(If none provided, shows all 4 in a 2×2 grid.).
"""

import argparse
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pinocchio as pin
import pycapacity as pycap

from reachy2_qpik.pinocchio_qpik import PinocchioIK
from reachy2_qpik.utils import savitzky_golay


def unit_step(pinik: PinocchioIK, step_amp: float, duration: float, t0: float):
    """Position Unit Step simulation."""
    # Parameters
    ik_step = pinik.dt
    dt = 1 / 500
    steps = int(duration / dt)

    sg_window = 9
    sg_order = 3
    sg_half = (sg_window - 1) // 2

    i_step_start = int(t0 / dt)

    buffer: list[deque[float]] = [deque(maxlen=sg_window) for _ in range(pinik.nv)]

    t_list = np.arange(steps) * dt
    cartesian_list = np.zeros(steps)
    step_input = np.zeros(steps)
    joint_positions = np.zeros((steps, pinik.nv))
    joint_speeds = np.zeros((steps, pinik.nv))
    joint_accels = np.zeros((steps, pinik.nv))
    acc_max = np.zeros(steps)

    q0 = np.deg2rad([0, 0, -10, -90, 0, 0, 0])  # Elbow 90°
    pin.framesForwardKinematics(pinik.model, pinik.data, q0)
    pin.updateFramePlacements(pinik.model, pinik.data)
    ee_baselink = pinik.data.oMf[pinik.ee_frame_id].copy()
    T_baselink_torso = pinik.data.oMf[pinik.model.getFrameId("torso")].copy()
    ee_torso = T_baselink_torso.inverse() * ee_baselink
    goal_pose = ee_torso.homogeneous
    goal_flat = goal_pose.copy()
    goal_step = goal_pose.copy()
    goal_step[0, 3] += step_amp

    q_prev = q0.copy()
    q_current = q0.copy()
    q_dot_current = np.zeros_like(q_current)

    tau_max = np.ones(pinik.nv) * 15.0
    tau_min = -tau_max

    # Control loop
    i = 0
    while i < steps:
        t = time.time()
        if i < i_step_start:
            goal = goal_flat
            step_input[i] = 0.0
        else:
            goal = goal_step
            step_input[i] = step_amp

        if i == 0:
            q_dot_current = np.zeros_like(q_current)
        else:
            q_dot_current = (q_current - q_prev) / ik_step

        for j in range(pinik.nv):
            buffer[j].append(q_dot_current[j])
        if len(buffer[0]) == sg_window:
            q_dot_smooth = np.zeros_like(q_dot_current)
            for j in range(pinik.nv):
                arr = np.array(buffer[j])
                q_dot_smooth[j] = savitzky_golay(arr, window_size=sg_window, order=sg_order, rate=ik_step)[sg_half]
            q_dot_current = q_dot_smooth

        # for j in range(pinik.nv):
        #     buffer[j].append(q_current[j])
        # if len(buffer[0]) == sg_window:
        #     q_dot_smooth = np.zeros_like(q_current)
        #     for j in range(pinik.nv):
        #         arr = np.array(buffer[j])
        #         q_dot_smooth[j] = savitzky_golay(arr, window_size=sg_window, deriv=1, order=sg_order, rate=dt)[sg_half]
        #     q_dot_current = q_dot_smooth

        q_ddot = pinik.compute_acceleration(goal, q_current, q_dot_current)
        if q_ddot is None:
            q_ddot = np.zeros_like(q_current)

        q_dot = q_dot_current + q_ddot * ik_step
        q_updated = pin.integrate(pinik.model, q_current, q_dot * ik_step)

        pin.framesForwardKinematics(pinik.model, pinik.data, q_updated)
        pin.updateFramePlacements(pinik.model, pinik.data)

        J_pos = pin.computeFrameJacobian(
            pinik.model, pinik.data, q_updated, pinik.model.getFrameId(pinik.ee_frame), pin.ReferenceFrame.LOCAL
        )[:3, :]
        M = pin.crba(pinik.model, pinik.data, q_updated)
        opt = {"calculate_faces": True}

        ee = pinik.data.oMf[pinik.ee_frame_id]
        tee = ee.translation
        Ree = ee.rotation

        cartesian_list[i] = tee[0] - ee_baselink.translation[0]
        acc_poly = pycap.robot.acceleration_polytope(J_pos, M, tau_max, tau_min, options=opt)
        acc_vertices = (Ree @ acc_poly.vertices).T + tee
        amax = np.max(acc_vertices[:, 0])

        acc_max[i] = amax
        joint_positions[i, :] = q_current
        joint_speeds[i, :] = q_dot
        joint_accels[i, :] = q_ddot

        q_prev[:] = q_current
        q_current[:] = q_updated
        i += 1
        time.sleep(max(dt - (time.time() - t), 0.0))

    return t_list, cartesian_list, step_input, joint_positions, joint_speeds, joint_accels, acc_max


def plot_results(
    pinik: PinocchioIK,
    mode: int,
    t: npt.NDArray[np.float64],
    cart_list: npt.NDArray[np.float64],
    step_in: npt.NDArray[np.float64],
    q_pos: npt.NDArray[np.float64],
    q_vel: npt.NDArray[np.float64],
    q_acc: npt.NDArray[np.float64],
    acc_max: npt.NDArray[np.float64],
    setting_time: float,
):
    """Display the results of the unit_step."""
    if mode == 4:
        _, axis = plt.subplots(2, 2, figsize=(12, 8))
        ax = axis[0, 0]
        ax.plot(t, cart_list, label="response")
        ax.plot(t, step_in, "r--", label="unit step")
        ax.set_title(f"Cartesian Unit Step - 5% Setting time: {setting_time:.4f}s")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("x (m)")

        ax.text(
            0.97,
            0.50,
            "Cartesian gains:\n" f"$K_{{pc}}$ = {pinik.Kpc}\n" f"$K_{{dc}}$ = {pinik.Kdc:.1f}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
        )

        ax.text(
            0.97,
            0.25,
            "Joint‑space gains:\n" f"$K_{{pa}}$ = {pinik.Kpa}\n" f"$K_{{da}}$ = {pinik.Kda:.1f}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
        )

        ax.grid(True)
        ax.legend()

        ax = axis[0, 1]
        for j in range(q_pos.shape[1]):
            ax.plot(t, q_pos[:, j], label=f"$q_{j}$")
        ax.set_title("Joint Positions")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (rad)")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize="small", ncol=2)

        ax = axis[1, 0]
        for j in range(q_vel.shape[1]):
            ax.plot(t, q_vel[:, j], label=f"$q̇_{j}$")
        ax.set_title("Joint Speeds")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (rad.s⁻¹)")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize="small", ncol=2)

        ax = axis[1, 1]
        for j in range(q_acc.shape[1]):
            ax.plot(t, q_acc[:, j], label=f"$q̈_{j}$")
        ax.plot(t, acc_max, "r--", label="$q̈_{\\max}$")
        ax.plot(t, -acc_max, "r--")
        ax.set_title("Joint Accelerations")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (rad.s⁻²)")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize="small", ncol=2)

        plt.tight_layout()
        plt.show()

    else:
        _, ax = plt.subplots(figsize=(12, 8))
        if mode == 0:
            ax.plot(t, cart_list, label="response")
            ax.plot(t, step_in, "r--", label="unit step")
            ax.set_title(f"Cartesian Unit Step - 5% Setting time: {setting_time:.4f}s")
            ax.set_ylabel("x (m)")

        elif mode == 1:
            for j in range(q_pos.shape[1]):
                ax.plot(t, q_pos[:, j], label=f"$q_{j}$")
            ax.set_title("Joint Positions")
            ax.set_ylabel("Position (rad)")
        elif mode == 2:
            for j in range(q_vel.shape[1]):
                ax.plot(t, q_vel[:, j], label=f"$q̇_{j}$")
            ax.set_title("Joint Speeds")
            ax.set_ylabel("Speed (rad.s⁻¹)")
        elif mode == 3:
            for j in range(q_acc.shape[1]):
                ax.plot(t, q_acc[:, j], label=f"q̈{j}")
            ax.plot(t, acc_max, "r--", label="$q̈_{\\max}$")
            ax.plot(t, -acc_max, "r--")
            ax.set_title("Joint Accelerations")
            ax.set_ylabel("Acceleration (rad.s⁻²)")
        else:
            raise ValueError("mode must be in [0-4]")

        ax.set_xlabel("Time (s)")
        ax.text(
            0.99,
            0.22,
            "Cartesian gains:\n" f"$K_{{pc}}$ = {pinik.Kpc}\n" f"$K_{{dc}}$ = {pinik.Kdc:.1f}",
            transform=ax.transAxes,
            fontsize=11,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
        )

        ax.text(
            0.99,
            0.1,
            "Joint‑space gains:\n" f"$K_{{pa}}$ = {pinik.Kpa}\n" f"$K_{{da}}$ = {pinik.Kda:.1f}",
            transform=ax.transAxes,
            fontsize=11,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
        )
        ax.grid(True)
        ax.legend(loc="best", fontsize="small", ncol=2)
        plt.tight_layout()
        plt.show()


def main():
    """Main example."""
    parser = argparse.ArgumentParser(description="Unit-step response plotting modes")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--uni", action="store_true", help="Cartesian unit step only")
    group.add_argument("--pos", action="store_true", help="Joint positions only")
    group.add_argument("--spe", action="store_true", help="Joint speeds only")
    group.add_argument("--acc", action="store_true", help="Joint accelerations only")
    args = parser.parse_args()

    if args.uni:
        mode = 0
    elif args.pos:
        mode = 1
    elif args.spe:
        mode = 2
    elif args.acc:
        mode = 3
    else:
        mode = 4

    # Parameters
    step_amp = 0.01
    duration = 0.5
    t0 = 0.2
    band = 0.05 * step_amp

    urdf = r"../config_files/reachy.urdf"
    pinik = PinocchioIK(urdf_path=urdf, arm="l_arm")

    t, cart, step_in, q_pos, q_vel, q_acc, acc_max = unit_step(pinik, step_amp, duration, t0)

    within_band = np.logical_and(cart >= step_amp - band, cart <= step_amp + band)

    setting_time = np.inf
    for i in range(len(t)):
        if all(within_band[i:]):
            setting_time = t[i] - t0
            break
    plot_results(pinik, mode, t, cart, step_in, q_pos, q_vel, q_acc, acc_max, setting_time)


if __name__ == "__main__":
    main()
