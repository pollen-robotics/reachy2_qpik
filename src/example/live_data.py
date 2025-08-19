"""Node to subscribe to target poses topics and collecting real time data."""

import argparse
import csv
import os
import time
from typing import Optional

import numpy as np
import numpy.typing as npt
import rclpy
from pollen_msgs.msg import IKRequest
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from reachy2_sdk import ReachySDK
from scipy.spatial.transform import Rotation as R


def make_homogenous_from_pose(position: npt.NDArray[np.float64], quat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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

    def __init__(self, reachy_host: str = "localhost", save_folder: str = "data", save_file: str = "live_data.csv") -> None:
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
            "l_real_pose",
            "r_pose",
            "r_real_pose",
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

        if self.r_topic in topic_type_map:
            self.r_msg_type = IKRequest
            self.r_sub = self.create_subscription(self.r_msg_type, self.r_topic, self.r_callback, qos_profile=self.qos)
            self.get_logger().info(f"Subscribed to {self.r_topic}")
        else:
            self.get_logger().warning(f"Topic {self.r_topic} not present on the ROS graph!")
            self.r_sub = None

        if self.l_topic in topic_type_map:
            self.l_msg_type = IKRequest
            self.l_sub = self.create_subscription(self.l_msg_type, self.l_topic, self.l_callback, qos_profile=self.qos)
            self.get_logger().info(f"Subscribed to {self.l_topic}")
        else:
            self.get_logger().warning(f"Topic {self.l_topic} not present on the ROS graph!")
            self.l_sub = None

        self.last_r_target: Optional[np.ndarray] = None
        self.last_l_target: Optional[np.ndarray] = None

    def r_callback(self, msg: IKRequest) -> None:
        """Callback for the right arm."""
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
        self._compute_and_save_data()

    def l_callback(self, msg: IKRequest) -> None:
        """Callback for the left arm."""
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
        self._compute_and_save_data()

    def _msg_to_matrix(self, msg: IKRequest) -> npt.NDArray[np.float64]:
        """Convert a msg to a NumPy matrix."""
        if IKRequest is not None and isinstance(msg, IKRequest):
            try:
                ps = msg.pose
                return make_homogenous_from_pose(ps.pose.position, ps.pose.orientation)
            except Exception:
                pass

    def _compute_and_save_data(self) -> None:
        """Compute the data and save it to a CSV file."""
        try:
            now = time.time()
            if not hasattr(self, "t0"):
                self.t0 = now
            time_val = now - self.t0

            l_joints = self.reachy.l_arm.get_current_positions()
            r_joints = self.reachy.r_arm.get_current_positions()
            l_real_pose = self.reachy.l_arm.forward_kinematics()
            r_real_pose = self.reachy.r_arm.forward_kinematics()

            row = (
                [time_val]
                + l_joints
                + r_joints
                + [
                    self.last_l_target.tolist() if self.last_l_target is not None else None,
                    l_real_pose.tolist(),
                    self.last_r_target.tolist() if self.last_r_target is not None else None,
                    r_real_pose.tolist(),
                ]
            )
            self.csv_writer.writerow(row)
            self.csv_file.flush()
        except Exception as e:
            self.get_logger().error(f"Failed to log CSV row: {e}")


def main(argv=None):
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", help="Reachy SDK host")
    parser.add_argument("--save-file", default="live_data.csv", help="CSV filename")
    args = parser.parse_args(argv)

    rclpy.init()
    try:
        node = LiveDataNode(reachy_host=args.host, save_file=args.save_file)
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
