"""Position Unitary Step example."""

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

from reachy2_qpik.pinocchio_qpik import PinocchioIK
from reachy2_qpik.utils import savitzky_golay

# Parameters
urdf = r"../config_files/reachy.urdf"
pinik = PinocchioIK(urdf_path=urdf, arm="r_arm")
dt = pinik.dt
duration = 4.0
steps = int(duration / dt)

sg_window = 7
sg_order = 2
sg_half = (sg_window - 1) // 2

step_amp = 0.01
t0 = 0.5
i_step_start = int(t0 / dt)

buffer: list[deque[np.float64]] = [deque(maxlen=sg_window) for _ in range(pinik.nv)]

q0 = pin.neutral(pinik.model).copy()
pin.framesForwardKinematics(pinik.model, pinik.data, q0)
pin.updateFramePlacements(pinik.model, pinik.data)
T_torso = pinik.data.oMf[pinik.ee_frame_id].copy()
T_baselink_torso = pinik.data.oMf[pinik.model.getFrameId("torso")].copy()
oMdes = T_baselink_torso.inverse() * T_torso
T0 = oMdes.homogeneous

goal_flat = T0.copy()
goal_step = T0.copy()
goal_step[0, 3] += step_amp

t_list = np.arange(steps) * dt
position_list = np.zeros(steps)
goal_list = np.zeros(steps)

q_previous = q0.copy()
q_current = q0.copy()

for i in range(steps):
    if i < i_step_start:
        goal = goal_flat
        goal_list[i] = 0.0
    else:
        goal = goal_step
        goal_list[i] = step_amp

    q_ddot = pinik.compute_acceleration(goal, q_current, q_previous)
    if q_ddot is None:
        q_ddot = np.zeros_like(q_current)

    if i == 0:
        q_dot_current = np.zeros_like(q_current)
    else:
        q_dot_current = (q_current - q_previous) / dt

    for j in range(pinik.nv):
        buffer[j].append(q_dot_current[j])
    if len(buffer[0]) == sg_window:
        q_dot_smooth = np.zeros_like(q_dot_current)
        for j in range(pinik.nv):
            arr = np.array(buffer[j])
            smooth_sig = savitzky_golay(arr, window_size=sg_window, order=sg_order)
            q_dot_smooth[j] = smooth_sig[sg_half]
        q_dot_current = q_dot_smooth

    q_dot = q_dot_current + q_ddot * dt

    q_updated = pin.integrate(pinik.model, q_current, q_dot * dt)

    pin.framesForwardKinematics(pinik.model, pinik.data, q_updated)
    pin.updateFramePlacements(pinik.model, pinik.data)
    ee = pinik.data.oMf[pinik.ee_frame_id]
    position_list[i] = ee.translation[0] - T_torso.translation[0]

    q_previous[:] = q_current
    q_current[:] = q_updated

plt.plot(t_list, position_list, label="response")
plt.plot(t_list, goal_list, "r--", label="unit step")
plt.xlabel("Time (s)")
plt.ylabel("x (m)")
plt.title("Cartesian unit step response")
plt.grid(True)
plt.legend()
plt.show()
