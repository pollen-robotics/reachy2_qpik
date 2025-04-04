"""Example class."""

import logging
from typing import Any, overload


class Foo:
    """This is a template class."""

    def __init__(self) -> None:
        """Set up empty slots and initialize the logger and private variable.

        This method is called when an object of the class is created. It sets up the logger and
        initializes the private variable.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.info("Constructor")
        self._private_variable = "private"
        self.public_variable = "public"

    @property
    def private_variable(self) -> str:
        """A property decorator that allows access to the private variable.

        This property decorator provides a way to access the private variable from outside
        the class. It returns the value of the private variable.
        """
        return self._private_variable

    @private_variable.setter
    def private_variable(self, value: str) -> None:
        """A setter for the private_variable property.

        This method allows the value of the private variable to be changed from outside the
        class. It sets the value of the private variable to the provided argument.
        """
        self._private_variable = value

    def __del__(self) -> None:
        """The destructor method called when the object is about to be destroyed.

        This method is called when an object of the class is about to be destroyed. It logs a
        message indicating that the destructor has been called.
        """
        self._logger.info("Destructor")

    @overload
    def doingstuffs(self, var: int, var2: float) -> None: ...

    @overload
    def doingstuffs(self, var: int) -> None: ...

    def doingstuffs(self, var: Any = None, var2: Any = None) -> None:
        """An overloaded method that takes one or two arguments and logs their values and types.

        This method demonstrates the use of overloading in Python. It takes one or two arguments
        and logs their values and types using the logger. If no arguments are provided, it does
        nothing.
        """
        if var is not None:
            self._logger.info(f"{var} {type(var)} ")
        if var2 is not None:
            self._logger.info(f"{var2} {type(var2)} ")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    f = Foo()
    f.doingstuffs(3)
    f.doingstuffs(3, 5.6)
