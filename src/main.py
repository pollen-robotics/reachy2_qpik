"""Main script for the application.

This script processes command-line arguments, performs various operations using classes and
methods from the `example` module, and uses data stored in configuration files. It logs
information about the provided arguments and demonstrates the usage of the application's
features.

"""

import argparse
import logging
import sys

from example.cam_config import CamConfig, get_config_file_path, get_config_files_names
from example.celcius import Celsius
from example.foo import Foo
from example.xterrabot import XTerraBot


def main(args: argparse.Namespace) -> int:
    """The main function that processes command-line arguments and performs various operations.

    This function logs information about the provided arguments, demonstrates the usage of
    classes and methods from the example module, and uses data stored in configuration
    files. It returns 0 to indicate successful execution.

    Args:
        args (argparse.Namespace): The namespace containing the command-line arguments.

    Returns:
        int: Exit code indicating the success (0) of the application.


    """
    logging.info("str param: {}".format(args.str_param))
    logging.info("bool param: {}".format(args.bool_param))
    logging.info("int param: {}".format(args.int_param))
    logging.info("float param: {}".format(args.float_param))
    logging.error("verbose: {}".format(args.verbose))
    logging.warning("this is a warning")

    foo = Foo()
    logging.info(foo.private_variable)

    temp = Celsius(37)
    temp.temperature = -30
    logging.info(temp.to_fahrenheit())

    xt_bot = XTerraBot()
    logging.info(xt_bot.get_object_in_gripper_frame())

    # usage of data stored in config_files
    cam_conf = CamConfig(get_config_file_path(args.config))
    logging.info(cam_conf.to_string())

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # examples
    parser.add_argument("--str_param", type=str, required=True)
    parser.add_argument("--bool_param", action="store_true", default=False)
    parser.add_argument("--int_param", type=int, default=5)
    parser.add_argument("--float_param", type=float, default=2.5)
    parser.add_argument("--verbose", action="store_true", default=False)

    valid_configs = get_config_files_names()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=valid_configs,
        help=f"Configutation file name : {valid_configs}",
    )

    args = parser.parse_args()

    # activate the --verbose to see more output
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    sys.exit(main(args))
