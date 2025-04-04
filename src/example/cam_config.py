"""Camera Configuration Management.

This module provides classes and functions for managing camera configuration data stored in JSON
files. It includes a class for reading and displaying camera configuration data, as well as
functions for retrieving the names of available configuration files and their paths.
"""

import json
from importlib.resources import files
from typing import Any, List


class CamConfig:
    """A class to manage camera configuration data from a JSON file.

    This class reads a JSON file containing camera configuration data and provides methods to
    access and display the information.

    Attributes:
        cam_config_json (str): The path to the JSON file containing the camera configuration data.
        socket_to_name (dict): A dictionary mapping socket IDs to camera names.
        inverted (bool): A boolean indicating whether the camera is inverted.
        fisheye (bool): A boolean indicating whether the camera is a fisheye camera.
        mono (bool): A boolean indicating whether the camera is a monochrome camera.
    """

    def __init__(self, cam_config_json: str) -> None:
        """Initialize the camera configuration data from the given JSON file.

        Args:
            cam_config_json (str): The path to the JSON file containing the camera configuration data.
        """
        self.cam_config_json = cam_config_json

        with open(self.cam_config_json, "rb") as f:
            config = json.load(f)
            self.socket_to_name = config["socket_to_name"]
            self.inverted = config["inverted"]
            self.fisheye = config["fisheye"]
            self.mono = config["mono"]

    def to_string(self) -> str:
        """Return a string representation of the camera configuration data.

        Returns:
            str: A string containing the camera configuration data in a human-readable format.
        """
        ret_string = "Camera Config: \n"
        ret_string += "Inverted: {}\n".format(self.inverted)
        ret_string += "Fisheye: {}\n".format(self.fisheye)
        ret_string += "Mono: {}\n".format(self.mono)

        return ret_string


def get_config_files_names() -> List[str]:
    """Return a list of the names of the JSON configuration files in the config_files package.

    Returns:
        List[str]: A list of the names of the JSON configuration files in the config_files package.
    """
    path = files("config_files")
    return [file.stem for file in path.glob("**/*.json")]  # type: ignore[attr-defined]


def get_config_file_path(name: str) -> Any:
    """Return the path to the JSON configuration file with the given name in the config_files package.

    Args:
        name (str): The name of the JSON configuration file.

    Returns:
        Any: The path to the JSON configuration file with the given name in the config_files package.
            If the file is not found, returns None.
    """
    path = files("config_files")
    for file in path.glob("**/*"):  # type: ignore[attr-defined]
        if file.stem == name:
            return file.resolve()
    return None
