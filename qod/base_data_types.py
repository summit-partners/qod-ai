from enum import IntEnum
from abc import ABC, abstractmethod


class BaseAttributeType(ABC, IntEnum):
    @abstractmethod
    def get_attributes(self):
        pass
