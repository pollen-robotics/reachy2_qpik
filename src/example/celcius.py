"""Celsius Temperature Conversion and Management.

This module provides a class for managing Celsius temperatures and converting them to Fahrenheit.
It also includes a main function that demonstrates the usage of the class.
"""

import logging


class Celsius:
    """A class to manage Celsius temperature and convert it to other formats.

    This class provides a way to store and manipulate Celsius temperatures, as well as convert
    them to Fahrenheit. It also includes a check to ensure that the temperature is not below
    absolute zero (-273.15°C).

    Attributes:
        _temperature (float): The current temperature in Celsius.
    """

    def __init__(self, temperature: float = 0):
        """Initialize the logger and the temperature attribute.

        Args:
            temperature (float, optional): The initial temperature in Celsius. Defaults to 0.
        """
        self._logger = logging.getLogger(__name__)
        self._temperature = temperature

    def to_fahrenheit(self) -> float:
        """Convert the current temperature from Celsius to Fahrenheit.

        Returns:
            float: The temperature in Fahrenheit.
        """
        return (self._temperature * 1.8) + 32

    @property
    def temperature(self) -> float:
        """A property decorator that allows access to the temperature attribute.

        This property decorator provides a way to access the temperature attribute from outside
        the class. It also logs a message indicating that the value is being retrieved.

        Returns:
            float: The current temperature in Celsius.
        """
        self._logger.info("Getting value...")
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """A setter for the temperature property.

        This method allows the value of the temperature attribute to be changed from outside the
        class. It also logs a message indicating that the value is being set and checks that the
        temperature is not below absolute zero.

        Args:
            value (float): The new temperature in Celsius.

        Raises:
            ValueError: If the temperature is below -273.15°C.
        """
        self._logger.info("Setting value...")
        if value < -273.15:
            raise ValueError("Temperature below -273 is not possible")
        self._temperature = value


def main() -> None:
    """The main function that demonstrates the usage of the Celsius class.

    This function creates an instance of the Celsius class, sets its temperature, and prints
    the equivalent temperature in Fahrenheit. It also activates logging at the INFO level.
    """
    logging.basicConfig(level=logging.INFO)

    print("Test entry point")
    temp = Celsius(37)
    temp.temperature = -30
    print(temp.to_fahrenheit())
