import pytest

from src.example.cam_config import (
    CamConfig,
    get_config_file_path,
    get_config_files_names,
)


def test_config() -> None:
    list_config = get_config_files_names()
    for conf in list_config:
        c = CamConfig(get_config_file_path(conf))
        assert c is not None
