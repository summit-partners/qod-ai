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

    LLAMA2_70B = 1
    LLAMA2_13B = 2
    NOUS_HERMES_LLAMA2_13B = 3
    NOUS_HERMES_13B = 4
    VICUNA_13B = 5
    WIZARDLM_13B = 6
    LLAMA1_UPSTAGE_65B = 7
    GPT4_ALPACA_LORA_65B = 8
    GPLATY_30B = 9

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
    LLMType.LLAMA2_70B: LLMAttributes(
        type=LLMType.LLAMA2_70B,
        family=LLMFamily.LLAMA,
        model="models/inference/ultralm-13b/ultralm-13b.ggmlv3.q8_0.bin",
        friendly_name="UltraLM (as Llama2 70B)",
    ),
    LLMType.LLAMA2_13B: LLMAttributes(
        type=LLMType.LLAMA2_13B,
        family=LLMFamily.LLAMA,
        model="models/inference/llama2-13b/llama-2-13b-chat.ggmlv3.q8_0.bin",
        friendly_name="LLAMA2 13B",
    ),
    LLMType.NOUS_HERMES_LLAMA2_13B: LLMAttributes(
        type=LLMType.NOUS_HERMES_LLAMA2_13B,
        family=LLMFamily.LLAMA,
        model="models/inference/nous-hermes-llama2-13b/nous-hermes-llama2-13b.ggmlv3.q8_0.bin",
        friendly_name="Nous-Hermes-Llama2",
    ),
    LLMType.NOUS_HERMES_13B: LLMAttributes(
        type=LLMType.NOUS_HERMES_13B,
        family=LLMFamily.LLAMA,
        model="models/inference/nous-hermes-13b/nous-hermes-13b.ggmlv3.q4_0.bin",
        friendly_name="Nous-Hermes 13B",
    ),
    LLMType.VICUNA_13B: LLMAttributes(
        type=LLMType.VICUNA_13B,
        family=LLMFamily.LLAMA,
        model="models/inference/vicuna/ggml-vic13b-q5_1.bin",
        friendly_name="Vicuna 13B",
    ),
    LLMType.WIZARDLM_13B: LLMAttributes(
        type=LLMType.WIZARDLM_13B,
        family=LLMFamily.LLAMA,
        model="models/inference/wizardlm-13b/wizardlm-13b-v1.2.ggmlv3.q8_0.bin",
        friendly_name="WizardLM 13B",
    ),
    LLMType.LLAMA1_UPSTAGE_65B: LLMAttributes(
        type=LLMType.LLAMA1_UPSTAGE_65B,
        family=LLMFamily.LLAMA,
        model="models/inference/llama1-upstage/upstage-llama-65b-instruct.ggmlv3.q5_K_M.bin",
        friendly_name="Llama1-Upstage 65B",
    ),
    LLMType.GPT4_ALPACA_LORA_65B: LLMAttributes(
        type=LLMType.GPT4_ALPACA_LORA_65B,
        family=LLMFamily.LLAMA,
        model="models/inference/gpt4-alpaca-lora/gpt4-alpaca-lora_mlp-65B.ggmlv3.q5_K_S.bin",
        friendly_name="GPT4 Alpaca Lora 65B",
    ),
    LLMType.GPLATY_30B: LLMAttributes(
        type=LLMType.GPLATY_30B,
        family=LLMFamily.LLAMA,
        model="models/inference/gplaty-30b/gplatty-30b.ggmlv3.q8_0.bin",
        friendly_name="Gplaty 30B",
    ),
}
"""
    LLMType.GROOVY
    LLMType.GPT4_ALPACA_LORA_65B: LLMAttributes(
        type=LLMType.GPT4_ALPACA_LORA_65B,
        family=LLMFamily.LLAMA,
        model="models/inference/gpt4-alpaca-lora/gpt4-alpaca-lora_mlp-65B.ggmlv3.q5_K_S.bin",
        friendly_name="GPT4 Alpaca Lora 65B",
    ),: LLMAttributes(
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
"""
