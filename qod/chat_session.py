from typing import Union, Tuple, List

from langchain.llms import GPT4All, LlamaCpp, OpenAI
from langchain.embeddings import (
    LlamaCppEmbeddings,
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
)
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain.vectorstores import Chroma
from langchain.schema import Document

from qod.embeddings_data_types import EmbeddingsType
from qod.llm_data_types import LLMType
from qod.chain_data_types import ChainType


class ChatSession:
    """Object to capture a chat session"""

    csid: str
    llm: Union[GPT4All, LlamaCpp, OpenAI]
    embedings: Union[LlamaCppEmbeddings, HuggingFaceEmbeddings, OpenAIEmbeddings]
    chain: Union[ConversationalRetrievalChain, BaseConversationalRetrievalChain]
    vectorstore: Chroma
    history: List[Tuple[str, str]]
    documents_directory: str
    db_directory: str
    llm_type: LLMType
    embeddings_type: EmbeddingsType
    chain_type: ChainType

    def __init__(
        self,
        csid: str,
        llm: Union[GPT4All, LlamaCpp, OpenAI],
        embeddings: Union[LlamaCppEmbeddings, HuggingFaceEmbeddings, OpenAIEmbeddings],
        chain: Union[ConversationalRetrievalChain, BaseConversationalRetrievalChain],
        vectorstore: Chroma,
        history: List[Tuple[str, str]],
        documents_directory: str,
        db_directory: str,
        llm_type: LLMType,
        embeddings_type: EmbeddingsType,
        chain_type: ChainType,
    ):
        """Constructor for a ChatSession
        :param csid: Identifier for a chat session
        :param llm: Large language model used by the chat session
        :param embedings: Embedings used to convert documents/questions into vectors
        :param chain: Chain used to answer questions
        :param vectorstore: Vector store used to store the documents embeddings
        :param history: Chat of the history
        :param documents_directory: Path to the documents questioned
        :param db_directory: Path of the document embeddings stored on disk
        :param llm_type: Type of LLM used
        :param embeddings_type: Type of embeddings used
        :param chain_type: Type of chain used
        """
        self.csid = csid
        self.llm = llm
        self.embeddings = embeddings
        self.chain = chain
        self.vectorstore = vectorstore
        self.history = history
        self.documents_directory = documents_directory
        self.db_directory = db_directory
        self.llm_type = llm_type
        self.embeddings_type = embeddings_type
        self.chain_type = chain_type

    def flush_history(self) -> None:
        """Clear the chat history"""
        self.history = []

    def answer_question(self, question: str) -> Tuple[str, List[Document]]:
        """Answer a question and provide sources
        :param question: Question to answer
        :return answer: Answer to the question
        :return sources: List of source documents used to assemble the answer
        """
        result = self.chain({"question": question, "chat_history": self.history})
        answer = result.get("answer", "")
        sources = result.get("source_documents", [])
        self.history.append((question, answer))
        return answer, sources
