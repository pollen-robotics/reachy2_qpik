"""Node to subscribe to target poses topics and collecting real time data."""

import argparse
import csv
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt
import rclpy
from pollen_msgs.msg import IKRequest
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.subscription import Subscription
from reachy2_sdk import ReachySDK
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64MultiArray


def make_homogenous_from_pose(position: Any, quat: Any) -> npt.NDArray[np.float64]:
    """Convert a translation and a quaternion to a 4x4 homogeneous matrix."""
    pos = np.asarray([position.x, position.y, position.z], dtype=float)
    q = np.asarray([quat.x, quat.y, quat.z, quat.w], dtype=float)
    rotation_matrix = R.from_quat(q).as_matrix()
    M = np.eye(4)
    M[:3, :3] = rotation_matrix
    M[:3, 3] = pos
    return M


class LiveDataNode(Node):
    """Node for collecting live data."""

    def __init__(
        self, reachy_host: str = "localhost", save_folder: str = "data/teleop", save_file: str = "live_data.csv"
    ) -> None:
        """Initializing the class."""
        super().__init__("live_data_node")

        os.makedirs(save_folder, exist_ok=True)
        self.csv_path = os.path.join(save_folder, save_file)
        self.csv_file = open(self.csv_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        header = [
            "time",
            "l_q0",
            "l_q1",
            "l_q2",
            "l_q3",
            "l_q4",
            "l_q5",
            "l_q6",
            "r_q0",
            "r_q1",
            "r_q2",
            "r_q3",
            "r_q4",
            "r_q5",
            "r_q6",
            "l_pose",
            "r_pose",
        ]
        self.csv_writer.writerow(header)
        self.csv_file.flush()
        self.get_logger().info(f"Logging CSV data to {self.csv_path}")

        self.get_logger().info(f"Connecting to Reachy at {reachy_host}...")

        try:
            self.reachy = ReachySDK(host=reachy_host)
        except Exception as e:
            self.get_logger().error(f"Failed to connect to Reachy, exiting...: {e}")
            raise

        if getattr(self.reachy, "_grpc_status", None) == "disconnected":
            self.get_logger().error("Reachy is disconnected, exiting node.")
            raise RuntimeError("Reachy disconnected")

        self.qos = QoSProfile(depth=10)
        self.qos.reliability = ReliabilityPolicy.BEST_EFFORT

        available_topics = self.get_topic_names_and_types()
        topic_type_map = {name: types for name, types in available_topics}

        self.r_topic = "/r_arm/ik_target_pose"
        self.l_topic = "/l_arm/ik_target_pose"

        self.r_ctrl_topic = "/r_arm_forward_position_controller/commands"
        self.l_ctrl_topic = "/l_arm_forward_position_controller/commands"

        self.r_sub: Optional[Subscription] = None
        self.l_sub: Optional[Subscription] = None
        self.r_ctrl_sub: Optional[Subscription] = None
        self.l_ctrl_sub: Optional[Subscription] = None

        self.last_r_joints_topic: Optional[list] = None
        self.last_l_joints_topic: Optional[list] = None

        self.last_r_target: Optional[npt.NDArray[np.float64]] = None
        self.last_l_target: Optional[npt.NDArray[np.float64]] = None

        if self.r_topic in topic_type_map:
            self.r_msg_type = IKRequest
            self.r_sub = self.create_subscription(self.r_msg_type, self.r_topic, self.r_callback, qos_profile=self.qos)
            self.get_logger().info(f"Subscribed to {self.r_topic}")
        else:
            self.get_logger().warning(f"Topic {self.r_topic} not present on the ROS graph!")

        if self.l_topic in topic_type_map:
            self.l_msg_type = IKRequest
            self.l_sub = self.create_subscription(self.l_msg_type, self.l_topic, self.l_callback, qos_profile=self.qos)
            self.get_logger().info(f"Subscribed to {self.l_topic}")
        else:
            self.get_logger().warning(f"Topic {self.l_topic} not present on the ROS graph!")

        if self.r_ctrl_topic in topic_type_map:
            msg_type = Float64MultiArray
            self.r_ctrl_sub = self.create_subscription(msg_type, self.r_ctrl_topic, self.r_ctrl_callback, qos_profile=self.qos)
            self.get_logger().info(f"Subscribed to {self.r_ctrl_topic}")
        else:
            self.get_logger().warning(f"Topic {self.r_ctrl_topic} not present on the ROS graph!")

        if self.l_ctrl_topic in topic_type_map:
            msg_type = Float64MultiArray
            self.l_ctrl_sub = self.create_subscription(msg_type, self.l_ctrl_topic, self.l_ctrl_callback, qos_profile=self.qos)
            self.get_logger().info(f"Subscribed to {self.l_ctrl_topic}")
        else:
            self.get_logger().warning(f"Topic {self.l_ctrl_topic} not present on the ROS graph!")

        self.log_period = 0.004
        self.last_save_time = 0.0
        self._prev_saved_state: Dict[str, Optional[Any]] = {
            "l_joints": None,
            "r_joints": None,
            "l_target": None,
            "r_target": None,
        }
        self.create_timer(self.log_period, self._compute_and_save_data)

    def r_callback(self, msg: IKRequest) -> None:
        """Callback for the right arm (IK target)."""
        try:
            M = self._msg_to_matrix(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to parse message on {self.r_topic}: {e}")
            try:
                self.get_logger().debug(f"Msg repr: {repr(msg)}")
            except Exception:
                pass
            return

        self.last_r_target = M

    def l_callback(self, msg: IKRequest) -> None:
        """Callback for the left arm (IK target)."""
        try:
            M = self._msg_to_matrix(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to parse message on {self.l_topic}: {e}")
            try:
                self.get_logger().debug(f"Msg repr: {repr(msg)}")
            except Exception:
                pass
            return

        self.last_l_target = M

    def r_ctrl_callback(self, msg: Float64MultiArray) -> None:
        """Callback for right controller command topic."""
        try:
            arr = list(msg.data) if msg is not None else None
            if arr is not None:
                self.last_r_joints_topic = arr[:7] if len(arr) >= 7 else arr
        except Exception as e:
            self.get_logger().error(f"Failed to parse controller message on {self.r_ctrl_topic}: {e}")

    def l_ctrl_callback(self, msg: Float64MultiArray) -> None:
        """Callback for left controller command topic (same handling)."""
        try:
            arr = list(msg.data) if msg is not None else None
            if arr is not None:
                self.last_l_joints_topic = arr[:7] if len(arr) >= 7 else arr
        except Exception as e:
            self.get_logger().error(f"Failed to parse controller message on {self.l_ctrl_topic}: {e}")

    def _msg_to_matrix(self, msg: IKRequest) -> Optional[npt.NDArray[np.float64]]:
        """Convert an IKRequest msg to a NumPy matrix."""
        if IKRequest is not None and isinstance(msg, IKRequest):
            try:
                ps = msg.pose
                return make_homogenous_from_pose(ps.pose.position, ps.pose.orientation)
            except Exception:
                return None
        else:
            return None

    def _compute_and_save_data(self) -> None:
        """Compute the data and save it to a CSV file."""
        try:
            now = time.time()
            if (now - getattr(self, "last_save_time", 0.0)) < (self.log_period * 0.9):
                return

            if not hasattr(self, "t0"):
                self.t0 = now
            time_val = now - self.t0

            l_joints = (
                self.last_l_joints_topic if self.last_l_joints_topic is not None else self.reachy.l_arm.get_current_positions()
            )
            r_joints = (
                self.last_r_joints_topic if self.last_r_joints_topic is not None else self.reachy.r_arm.get_current_positions()
            )

            lt = self.last_l_target
            rt = self.last_r_target

            prev = self._prev_saved_state

            l_list = list(l_joints) if l_joints is not None else [None] * 7
            r_list = list(r_joints) if r_joints is not None else [None] * 7

            row = (
                [time_val]
                + (l_list if l_list is not None else [None] * 7)
                + (r_list if r_list is not None else [None] * 7)
                + [
                    lt.tolist() if lt is not None else None,
                    rt.tolist() if rt is not None else None,
                ]
            )

            self.csv_writer.writerow(row)
            self.csv_file.flush()

            prev["l_joints"] = np.array(l_list, dtype=float) if l_list is not None else None
            prev["r_joints"] = np.array(r_list, dtype=float) if r_list is not None else None
            prev["l_target"] = (
                np.copy(lt) if isinstance(lt, np.ndarray) else (np.array(lt, dtype=float) if lt is not None else None)
            )
            prev["r_target"] = (
                np.copy(rt) if isinstance(rt, np.ndarray) else (np.array(rt, dtype=float) if rt is not None else None)
            )

            self.last_save_time = now

        except Exception as e:
            self.get_logger().error(f"Failed to log CSV row: {e}")


def main(argv=None):
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", help="Reachy SDK host")
    parser.add_argument("--filename", default="live_data.csv", help="CSV filename")
    args = parser.parse_args(argv)

    rclpy.init()
    try:
        node = LiveDataNode(reachy_host=args.host, save_file=args.filename)
    except Exception as e:
        print(f"Failed to initialize node: {e}")
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.csv_file.close()
        node.destroy_node()
        node.reachy.disconnect()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
