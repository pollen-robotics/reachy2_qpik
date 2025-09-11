# Reachy2 quadratic programming inverse kinematics

A kinematics library for Reachy2 7 DoF arms, using quadratic programming for precise and robust motion control.

[![Licence](https://img.shields.io/badge/licence-Apache%202.0-blue)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![linter](https://github.com/pollen-robotics/python-template/actions/workflows/lint.yml/badge.svg)
![pytest](https://github.com/pollen-robotics/python-template/actions/workflows/pytest.yml/badge.svg)
![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/FabienDanieau/58642e8fe4589e710e26627e39ff92d7/raw/covbadge.json)
![Docs](https://github.com/pollen-robotics/python-template/actions/workflows/docs.yml/badge.svg)

<p align="center">
  <img width="346" height="461" alt="Reachy 2 Heart GIF" src="https://github.com/user-attachments/assets/fd3be9ea-df6f-410e-a7e1-3bdb481a22ce"/>
</p>

## Table of contents

| Section           | Description                                            |
|-------------------|--------------------------------------------------------|
| [Introduction](#reachy2-quadratic-programming-inverse-kinematics) | Overview of the Reachy2 IK library  |
| [Key features](#key-features) | Highlights of main capabilities and advantages of the library  |
| [Installation](#installation)           | How to install dependencies and the package          |
| [Usage](#usage)                         | How to import and use the package                    |
| [Unit tests](#unit-tests)               | How to run online unit tests                         |
| [URDF](#urdf)                           | Location of the robot description file (URDF)        |
| [Contribution](#contribution)           | How to contribute to the repository                  |
| [License](#license)                     | Licensing information                                |

## Key features

1. **Quadratic Programming Inverse Kinematics:**
   * Solves inverse kinematics with [quadratic programming](https://scaron.info/blog/quadratic-programming-in-python.html) by minimizing joint accelerations under task-space constraints.
   * Handles accelerations, speed and joints limits.
   * Robustness against singularities — a fancy way of saying we can avoid unstable joint configurations.
2. **Task-Space Control Algorithm:**
   * Pose tracking suitable for teleoperation, ensures joint-space continuity.
   * Handles unreachable poses gracefully within trajectories.
   * Customizable workspace and configuration parameters.

## Installation

Dependencies are detailed in the [`setup.cfg`](./setup.cfg) file. To install this package locally, run:

```bash
pip install -e .[dev]
```

_Include [dev] for optional development tools._

## Usage

Once this is done, you should be able to import the Python package in your codes with:

```python
import reachy2_qpik
```

Check the [example](./src/example) folder for complete examples.

## Unit tests

To ensure everything is functionning correctly, you can run unit tests. The tests need to be done **online**: they require a connection to a simulated robot (e.g., in rviz), and the virtual robot should exhibit movement during these tests.

Example:

```bash
pytest -m online
```

or:

```bash
python3 -m pytest -m online
```

## URDF

A URDF file is provided in ['src/config_files/reachy.urdf'](./src/config_files/reachy.urdf).

## Contribution

All contributions are welcome!  

* **Report Issues**: Found a bug or have a feature request? Create a new issue [here](https://github.com/pollen-robotics/reachy2_qpik/issues/new/choose).  
* **Fix Bugs & Add Features**: Find out where you can lend a hand by checking out [existing issues](https://github.com/pollen-robotics/reachy2_qpik/issues).

## License

This project is licensed under the **Apache 2.0 License.** See the [LICENSE](./LICENSE) file for more details.
