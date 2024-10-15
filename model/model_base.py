from abc import ABC, abstractmethod

__all__ = [
    "ModelBase"
]

class ModelBase(ABC):
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def train(self, **kwargs) -> None:
        pass

    @abstractmethod
    def validate(self, **kwargs) -> None:
        pass

    @abstractmethod
    def test(self, **kwargs) -> None:
        pass
