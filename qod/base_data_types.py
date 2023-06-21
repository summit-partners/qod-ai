from enum import IntEnum
from abc import abstractmethod


class BaseAttributeType(IntEnum):
    @abstractmethod
    def get_attributes(self):
        pass
