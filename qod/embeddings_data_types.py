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

    # SENTENCE_TRANSFORMER_HF = 1
    # VICUNA_13B = 2
    # TEXT_EMBED_ADA_2 = 3
    MULTI_QA_MINI_LM_L6_COS_V1 = 1
    MULTI_QA_DISTILBERT_COS_V1 = 2
    MULTI_QA_MPNET_BASE_DOT_V1 = 3
    ALL_MINI_LM_6_V2 = 4
    ALL_MINI_LM_L12_V2 = 5
    ALL_DISTILROBERTA_V1 = 6
    ALL_MPNET_BASE_V2 = 7

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
    # EmbeddingsType.SENTENCE_TRANSFORMER_HF: EmbeddingsAttributes(
    #     type=EmbeddingsType.SENTENCE_TRANSFORMER_HF,
    #     family=EmbeddingsFamily.HUGGING_FACE,
    #     model="models/sentence_tansformers/all-MiniLM-L6-v2_local/",
    #     friendly_name="Hugging Face Sentence Transformer",
    # ),
    # EmbeddingsType.VICUNA_13B: EmbeddingsAttributes(
    #     type=EmbeddingsType.VICUNA_13B,
    #     family=EmbeddingsFamily.LLAMA_CPP,
    #     model="models/vicuna/ggml-vic13b-q5_1.bin",
    #     friendly_name="Vicuna 13B",
    # ),
    # EmbeddingsType.TEXT_EMBED_ADA_2: EmbeddingsAttributes(
    #     type=EmbeddingsType.TEXT_EMBED_ADA_2,
    #     family=EmbeddingsFamily.OPEN_AI,
    #     model="text-embedding-ada-002",
    #     friendly_name="Text embedding Ada 2",
    # ),
    EmbeddingsType.MULTI_QA_MINI_LM_L6_COS_V1: EmbeddingsAttributes(
        type=EmbeddingsType.MULTI_QA_MINI_LM_L6_COS_V1,
        family=EmbeddingsFamily.HUGGING_FACE,
        model="models/sentence_tansformers/multi-qa-MiniLM-L6-cos-v1_local/",
        friendly_name="Multi QA MiniLM L6 Cos v1",
    ),
    EmbeddingsType.MULTI_QA_DISTILBERT_COS_V1: EmbeddingsAttributes(
        type=EmbeddingsType.MULTI_QA_DISTILBERT_COS_V1,
        family=EmbeddingsFamily.HUGGING_FACE,
        model="models/sentence_tansformers/multi-qa-distilbert-cos-v1_local/",
        friendly_name="Multi QA Distilbert Cos v1",
    ),
    EmbeddingsType.MULTI_QA_MPNET_BASE_DOT_V1: EmbeddingsAttributes(
        type=EmbeddingsType.MULTI_QA_MPNET_BASE_DOT_V1,
        family=EmbeddingsFamily.HUGGING_FACE,
        model="models/sentence_tansformers/multi-qa-mpnet-base-dot-v1_local/",
        friendly_name="Multi QA Mpnet Base Dot v1",
    ),
    EmbeddingsType.ALL_MINI_LM_6_V2: EmbeddingsAttributes(
        type=EmbeddingsType.ALL_MINI_LM_6_V2,
        family=EmbeddingsFamily.HUGGING_FACE,
        model="models/sentence_tansformers/all-MiniLM-L6-v2_local/",
        friendly_name="All MiniLM L6 V2",
    ),
    EmbeddingsType.ALL_MINI_LM_L12_V2: EmbeddingsAttributes(
        type=EmbeddingsType.ALL_MINI_LM_L12_V2,
        family=EmbeddingsFamily.HUGGING_FACE,
        model="models/sentence_tansformers/all-MiniLM-L12-v2_local/",
        friendly_name="All MiniLM L12 v2",
    ),
    EmbeddingsType.ALL_DISTILROBERTA_V1: EmbeddingsAttributes(
        type=EmbeddingsType.ALL_DISTILROBERTA_V1,
        family=EmbeddingsFamily.HUGGING_FACE,
        model="models/sentence_tansformers/all-distilroberta-v1_local/",
        friendly_name="All Distilroberta v1",
    ),
    EmbeddingsType.ALL_MPNET_BASE_V2: EmbeddingsAttributes(
        type=EmbeddingsType.ALL_MPNET_BASE_V2,
        family=EmbeddingsFamily.HUGGING_FACE,
        model="models/sentence_tansformers/all-mpnet-base-v2_local/",
        friendly_name="All Mpnet Base v2",
    ),
}
