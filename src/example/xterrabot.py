"""XTerraBot.

Illustrate Maths notation based on
    https://www.mecharithm.com/
    homogenous-transformation-matrices-configurations-in-robotics/.
"""

import logging

import numpy as np
import numpy.typing as npt


class XTerraBot:
    """XTerraBot class.

    This class illustrates the use of homogeneous transformation matrices to represent the
    configuration of a robot in the context of robotics. It demonstrates the calculation of
    the transformation matrix of an object with respect to the gripper frame using matrix
    multiplication and inversion.
    """

    def __init__(self) -> None:
        """Constructor.

        Initialize the logger and the homogeneous transformation matrices representing the
        configuration of the robot.
        """
        self._logger = logging.getLogger(__name__)
        # b is the mobile base
        # d is the camera
        self._T_d_b = np.array([[0, 0, -1, 250], [0, -1, 0, -150], [-1, 0, 0, 200], [0, 0, 0, 1]])

        self._T_d_e = np.array([[0, 0, -1, 300], [0, -1, 0, 100], [-1, 0, 0, 120], [0, 0, 0, 1]])  # e is the object
        self._T_b_c = np.array(
            [
                [0, -1 / np.sqrt(2), -1 / np.sqrt(2), 30],
                [0, 1 / np.sqrt(2), -1 / np.sqrt(2), -40],
                [1, 0, 0, 25],
                [0, 0, 0, 1],
            ]
        )  # c is the joint of the gripper
        self._T_a_d = np.array([[0, 0, -1, 400], [0, -1, 0, 50], [-1, 0, 0, 300], [0, 0, 0, 1]])  # a is the root

    def get_object_in_gripper_frame(self) -> npt.NDArray[np.float64]:
        """Get the object in the gripper frame.

        Calculate and return the homogeneous transformation matrix representing the
        configuration of the object with respect to the gripper frame.

        This method calculates the transformation matrix (T_c_e) by performing matrix
        multiplication and inversion on the given transformation matrices. It returns the
        resulting matrix as a NumPy array.

        Returns:
            np.ndarray: The homogeneous transformation matrix representing the configuration
                of the object with respect to the gripper frame.
        """
        T_c_e = (
            np.linalg.inv(self._T_b_c)
            @ np.linalg.inv(self._T_d_b)
            # @ np.linalg.inv(self._T_a_d)
            # @ self._T_a_d ## inv(self._T_a_d) @ self._T_a_d = 1
            @ self._T_d_e
        )
        return np.array(T_c_e)
