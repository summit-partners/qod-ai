from pydantic.dataclasses import dataclass
from enum import unique, IntEnum
from qod.base_data_types import BaseAttributeType


@unique
class LLMFamily(IntEnum):
    """Enum for the family of large language model"""

    LLAMA = 1
    GPT4ALL = 2
    OPEN_AI = 3


@unique
class LLMType(BaseAttributeType):
    """Enum for the type of large language model"""

    VICUNA_13B = 1
    GROOVY = 2
    MPT_7B = 3
    TEXT_DAVINCI_3 = 4

    def get_attributes(self) -> "LLMAttributes":
        """Get the attributes associated with an LLM type"""
        attr = LLM_ATTRIBUTES.get(self, None)
        if attr is not None:
            return attr
        raise Exception(f"Could not find the attributes for the LLM type {self}")


@dataclass
class LLMAttributes:
    """Object for the attributes of an LLM"""

    type: LLMType
    family: LLMFamily
    model: str
    friendly_name: str

    def __init__(
        self, type: LLMType, family: LLMFamily, model: str, friendly_name: str
    ):
        """Constructor for a LLMAttribites
        :param type: Type of the LLM
        :param family: Family of the LLM
        :param model: path or name of the LLM model
        :param friendly_name: friendly name of the LLM model
        """
        self.type = type
        self.family = family
        self.model = model
        self.friendly_name = friendly_name


# Mapping from an LLM type to its attributes
LLM_ATTRIBUTES = {
    LLMType.VICUNA_13B: LLMAttributes(
        type=LLMType.VICUNA_13B,
        family=LLMFamily.LLAMA,
        model="models/vicuna/ggml-vic13b-q5_1.bin",
        friendly_name="Vicuna 13B",
    ),
    LLMType.GROOVY: LLMAttributes(
        type=LLMType.GROOVY,
        family=LLMFamily.GPT4ALL,
        model="",
        friendly_name="Groovy",
    ),
    LLMType.MPT_7B: LLMAttributes(
        type=LLMType.MPT_7B,
        family=LLMFamily.GPT4ALL,
        model="",
        friendly_name="MPT 7B",
    ),
    LLMType.TEXT_DAVINCI_3: LLMAttributes(
        type=LLMType.TEXT_DAVINCI_3,
        family=LLMFamily.OPEN_AI,
        model="text-davinci-003",
        friendly_name="Text DaVinvi 3",
    ),
}
