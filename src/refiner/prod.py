from abc import ABC, abstractmethod

from src.gresh import Gresh


class Production(ABC):
    @abstractmethod
    def check(self, g: Gresh, center: int) -> list[int] | None:
        pass

    @abstractmethod
    def transform(self, g: Gresh, center: int) -> bool:
        pass
