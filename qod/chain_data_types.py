from pydantic.dataclasses import dataclass
from enum import unique
from qod.base_data_types import BaseAttributeType


@unique
class ChainType(BaseAttributeType):
    """Enum for the type of chains"""

    STUFFED = 1
    MAP_REDUCE = 2
    REFINE = 3
    MAP_RERANK = 4

    def get_attributes(self) -> "ChainAttributes":
        """Get the attributes associated with an chain type"""
        attr = CHAIN_ATTRIBUTES.get(self, None)
        if attr is not None:
            return attr
        raise Exception(f"Could not find the attributes for the chain type {self}")


@unique
class SummaryChainType(BaseAttributeType):
    """Enum for the type of chains that can be used to summarize documents"""

    STUFFED = ChainType.STUFFED
    MAP_REDUCE = ChainType.MAP_REDUCE
    REFINE = ChainType.REFINE

    def get_attributes(self) -> "ChainAttributes":
        """Get the attributes associated with an chain type"""
        attr = SUMMARY_CHAIN_ATTRIBUTES.get(self, None)
        if attr is not None:
            return attr
        raise Exception(f"Could not find the attributes for the chain type {self}")


@dataclass
class ChainAttributes:
    """Object for the attributes of a chain"""

    type: ChainType
    model: str
    friendly_name: str

    def __init__(self, type: ChainType, model: str, friendly_name: str):
        """Constructor for a EmbeddingsAttribites
        :param type: Type of the chain
        :param model: formal name of the chain
        :param friendly_name: friendly name of the chain
        """
        self.type = type
        self.model = model
        self.friendly_name = friendly_name


# Mapping from an chain type to its attributes
CHAIN_ATTRIBUTES = {
    ChainType.STUFFED: ChainAttributes(
        type=ChainType.STUFFED, model="stuffed", friendly_name="None"
    ),
    ChainType.MAP_REDUCE: ChainAttributes(
        type=ChainType.MAP_REDUCE, model="map_reduce", friendly_name="Map reduce"
    ),
    ChainType.MAP_RERANK: ChainAttributes(
        type=ChainType.MAP_RERANK, model="map_rerank", friendly_name="Map rerank"
    ),
    ChainType.REFINE: ChainAttributes(
        type=ChainType.REFINE, model="refine", friendly_name="Refine"
    ),
}

# Mapping from a chain type to its attributes
SUMMARY_CHAIN_ATTRIBUTES = {
    SummaryChainType.STUFFED: ChainAttributes(
        type=ChainType.STUFFED, model="stuffed", friendly_name="None"
    ),
    SummaryChainType.MAP_REDUCE: ChainAttributes(
        type=ChainType.MAP_REDUCE, model="map_reduce", friendly_name="Map reduce"
    ),
    SummaryChainType.REFINE: ChainAttributes(
        type=ChainType.REFINE, model="refine", friendly_name="Refine"
    ),
}
