# Reachy2 quadratic programming inverse kinematics

A kinematics library for Reachy2 7 DoF arms, using quadratic programming for precise and robust motion control.

[![Licence](https://img.shields.io/badge/licence-Apache%202.0-blue)](LICENSE) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 
![linter](https://github.com/pollen-robotics/python-template/actions/workflows/lint.yml/badge.svg) 
![pytest](https://github.com/pollen-robotics/python-template/actions/workflows/pytest.yml/badge.svg) 
![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/FabienDanieau/58642e8fe4589e710e26627e39ff92d7/raw/covbadge.json) 
![Docs](https://github.com/pollen-robotics/python-template/actions/workflows/docs.yml/badge.svg)

## Table of contents

| Section           | Description                                            |
|-------------------|--------------------------------------------------------|
| [Introduction](#reachy2-quadratic-programming-inverse-kinematics) | Overview of the Reachy2 IK library                     |
| [Key features](#key-features) | Highlights of main capabilities and advantages of the library
| [Installation](#installation)           | How to install dependencies and the package          |
| [Usage](#usage)                       | How to import and use the package                      |
| [Unit tests](#unit-tests)               | How to run online unit tests                            |
| [URDF](#urdf)                         | Location of the robot description file (URDF)          |
| [License](#license)                     | Licensing information                                  |

## Key features

## Installation

Dependencies are detailed in the `setup.cfg` file. To install this package locally, run:

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

 ### Unit tests and test coverage

 Unit tests must be written to ensure code robustness. [Pytest](https://docs.pytest.org) is the recommended tool. Examples are provided in the *tests* folder.
 
It is recommended to have at least 90% of the code tested. The [coverage](https://coverage.readthedocs.io) package provide this metric.

 The developer must run the test locally before committing any new code. Make sure that *pytest* and *coverage* are installed and run at the root level:
 ```
 coverage run -m pytest
 ```
Then, if all tests are sucessful:
 ```
 coverage report
 ```
 These tests are automatically performed by a github action when a pull request is created.

 _Note that when creating a new repo from this template, you will need to first configure the Make badge action in the [pytest.yml](https://github.com/pollen-robotics/python-template/blob/develop/.github/workflows/pytest.yml#L42-L53) file. Follow [those instructions](https://github.com/schneegans/dynamic-badges-action/tree/v1.6.0/#configuration) if you don't have a gist secret and id yet._ 

 ## Documentation

The documentation is generated with [pdoc](https://pdoc.dev), automatically with the CI. To generate it locally you can run:

```
 pdoc example --output-dir docs --logo https://www.pollen-robotics.com/wp-content/themes/bambi-theme-main/assets/images/pollen_robotics_logo.webp --logo-link https://www.pollen-robotics.com --docformat google
 ```

 The documentation relies on the provided docstings with the google style. [pydocstyle](http://www.pydocstyle.org/en/stable/) is used to enforced this style.
 ```
 pydocstyle src/ --convention google --count
 ```

## Unit tests

To ensure everything is functionning correctly, you can run unit tests. The tests need to be done online: they require a connection to a simulated robot (e.g., in rviz), and the virtual robot should exhibit movement during these tests.

Example:

```bash
pytest -m cicd
```

or:

```bash
python3 -m pytest -m cicd
```

## URDF

A URDF file is provided in ['src/config_files/reachy.urdf'](./src/config_files/reachy.urdf).

## License

This project is licensed under the **Apache 2.0 License.** See the [LICENSE](./LICENSE) file for more details.
