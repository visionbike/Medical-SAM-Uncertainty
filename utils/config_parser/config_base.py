from abc import ABC, abstractmethod
from typing import Any
import argparse

__all__ = ["ConfigBase"]


class ConfigBase(ABC):
    """
    The Base ConfigParser.
    """

    def __init__(self, **kwargs) -> None:
        self.parser = argparse.ArgumentParser(
            description="Base Configuration",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.args = None

    def _print_args(self) -> None:
        """
        Print config arguments.
        """

        msg = "---------------- Options ----------------\n"
        cmt = ""
        for k, v in sorted(vars(self.args).items()):
            default = self.parser.get_default(k)
            if v != default:
                cmt = f"\t[default: {str(default)}]"
            msg += f"{str(k):>25} : {str(v):<30} {cmt}\n"
        msg += "----------------   End   ----------------\n"
        print(msg)

    @abstractmethod
    def _add_arguments(self) -> None:
        pass

    @abstractmethod
    def init_args(self) -> None:
        pass

    @abstractmethod
    def parse(self) -> Any:
        pass
