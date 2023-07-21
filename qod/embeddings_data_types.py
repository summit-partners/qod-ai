from pydantic.dataclasses import dataclass
from enum import IntEnum, unique
from qod.base_data_types import BaseAttributeType


@unique
class EmbeddingsFamily(IntEnum):
    """Enum for the family of embeddings"""

    HUGGING_FACE = 1
    LLAMA_CPP = 2
    OPEN_AI = 3


@unique
class EmbeddingsType(BaseAttributeType):
    """Enum for the type of embeddings"""

    SENTENCE_TRANSFORMER_HF = 1
    VICUNA_13B = 2
    TEXT_EMBED_ADA_2 = 3

    def get_attributes(self) -> "EmbeddingsAttributes":
        """Get the attributes associated with an embeddings type"""
        attr = EMBEDDINGS_ATTRIBUTES.get(self, None)
        if attr is not None:
            return attr
        raise Exception(f"Could not find the attributes for the embedding type {self}")


@dataclass
class EmbeddingsAttributes:
    """Object for the attributes of an embeddings"""

    type: EmbeddingsType
    family: EmbeddingsFamily
    model: str
    friendly_name: str

    def __init__(
        self,
        type: EmbeddingsType,
        family: EmbeddingsFamily,
        model: str,
        friendly_name: str,
    ):
        """Constructor for a EmbeddingsAttribites
        :param type: Type of the embeddings
        :param family: Family of the embeddings
        :param model: path or name of the embeddings model
        :param friendly_name: friendly name of the embeddings model
        """
        self.type = type
        self.family = family
        self.model = model
        self.friendly_name = friendly_name


# Mapping from an embedding type to its attributes
EMBEDDINGS_ATTRIBUTES = {
    EmbeddingsType.SENTENCE_TRANSFORMER_HF: EmbeddingsAttributes(
        type=EmbeddingsType.SENTENCE_TRANSFORMER_HF,
        family=EmbeddingsFamily.HUGGING_FACE,
        model="models/sentence_tansformers/all-MiniLM-L6-v2_local/",
        friendly_name="Hugging Face Sentence Transformer",
    ),
    EmbeddingsType.VICUNA_13B: EmbeddingsAttributes(
        type=EmbeddingsType.VICUNA_13B,
        family=EmbeddingsFamily.LLAMA_CPP,
        model="models/vicuna/ggml-vic13b-q5_1.bin",
        friendly_name="Vicuna 13B",
    ),
    EmbeddingsType.TEXT_EMBED_ADA_2: EmbeddingsAttributes(
        type=EmbeddingsType.TEXT_EMBED_ADA_2,
        family=EmbeddingsFamily.OPEN_AI,
        model="text-embedding-ada-002",
        friendly_name="Text embedding Ada 2",
    ),
}
