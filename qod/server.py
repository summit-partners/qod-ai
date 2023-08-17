from datetime import datetime

import secrets
import os
import pydantic
import socket
from fastapi import FastAPI, Depends, Response
from functools import cache
from typing import Optional, MutableMapping, List, Tuple

from langchain.schema import Document
from langchain.llms import LlamaCpp

from qod.embeddings_data_types import EmbeddingsType
from qod.llm_data_types import LLMType
from qod.chain_data_types import ChainType, SummaryChainType
from qod.chat_session import ChatSession
from qod.summary_session import SummarySession
from qod.langchain_wrappers import get_llm, get_embeddings, get_chain, get_vectorstore


app = FastAPI()
chat_sessions: MutableMapping[str, ChatSession] = {}


class AppStatus(pydantic.BaseModel):
    version: str = os.environ.get("GIT_VERSION", "unknown")
    start_time: datetime = datetime.utcnow().replace(microsecond=0)
    hostname: str = socket.gethostname()


@cache
def app_status():
    return AppStatus()


@app.get("/", response_model=AppStatus)
def root(_: Response, app_status: AppStatus = Depends(app_status)) -> AppStatus:
    """
    Returns the current status of the deployed instance.
    """
    return app_status


@app.get("/setup", response_model=str)
def setup_chat_session(
    llm_type: LLMType = LLMType.LLAMA2_13B,
    embeddings_type: EmbeddingsType = EmbeddingsType.MULTI_QA_MINI_LM_L6_COS_V1,
    chain_type: ChainType = ChainType.STUFFED,
    documents_directory: str = "docs",
    db_directory: Optional[str] = None,
) -> str:
    """Create and setup a new chat session
    :param llm_type: Type of LLM to use
    :param embeddings_type: Type of embedding to use
    :param chain_type: Type of chain to use
    :param document_directory: Path to the directory where the documents to
    question are stored
    :param db_directory: Path to the directory where the embeddings for
    the documents are stored
    :return csid: The identifier of the chat session created
    """
    # Choose the LLM to use for Q&A
    llm = get_llm(attr=llm_type.get_attributes())

    # Choose the LLM to index the documents
    embeddings = get_embeddings(attr=embeddings_type.get_attributes())

    # Create or load an indexed DB
    vectorstore, filled_db_directory = get_vectorstore(
        embeddings=embeddings,
        db_directory=db_directory,
        documents_directory=documents_directory,
    )

    # Select the chain type
    chain = get_chain(
        attr=chain_type.get_attributes(), llm=llm, vectorstore=vectorstore
    )

    # Create a chat session id
    csid: str = "csid_" + secrets.token_hex(12)

    # Create a session
    chat_sessions[csid] = ChatSession(
        csid=csid,
        llm=llm,
        embeddings=embeddings,
        chain=chain,
        vectorstore=vectorstore,
        history=[],
        documents_directory=documents_directory,
        db_directory=filled_db_directory,
        llm_type=llm_type,
        embeddings_type=embeddings_type,
        chain_type=chain_type,
    )
    return csid


@app.get("/ask", response_model=Tuple[str, List[Document]])
def ask_question(
    response: Response, csid: str, question: str
) -> Tuple[str, List[Document]]:
    """Ask a question to the LLM and collect the answer & sources
    :param response: Response
    :param csid: Identifier of the chat session in which the question is asked
    :param question: Question to ask to the LLM
    :return Answer to the question and list of the document used to assemble
    this answer
    """
    if csid not in chat_sessions:
        response.status_code = 400
        response.body = b"The chat session does not exist"
        response.headers["Content-Type"] = "text/plain"
        return "", []

    return chat_sessions[csid].answer_question(question=question)


@app.get("/flush")
def flush_chat_history(response: Response, csid: str) -> None:
    """Erase the chat history of a session
    :param response: Response
    :param csid: Identifier of the chat session of which we want to
    clear the history
    """
    if csid not in chat_sessions:
        response.status_code = 400
        response.body = b"The chat session does not exist"
        response.headers["Content-Type"] = "text/plain"

    chat_sessions[csid].flush_history()


@app.get("/summarize", response_model=Tuple[str, str])
def summarize_document(
    document_path: str,
    llm_type: LLMType = LLMType.LLAMA2_13B,
    summary_chain_type: SummaryChainType = SummaryChainType.STUFFED,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None,
) -> Tuple[str, str]:
    """Create and setup a new chat session
    :param document_path: Path of the document to summarize
    :param llm_type: Type of LLM to use
    :param summary_chain_type: Type of summary chain to use
    :param first_page: First page of the document's content to summarize
    :param last_page: Last page of the document's content to summarize
    :return summary: Summary of the document
    :return csid: The identifier of the summary session created
    """
    # Creating an identifier for the session
    ssid: str = "ssid_" + secrets.token_hex(12)
    llm = get_llm(attr=llm_type.get_attributes())
    if not isinstance(llm, LlamaCpp):
        raise Exception(
            "Only Llama model are currently supported \
for summarization. Please select another model."
        )

    # Create the summary session
    summary_session = SummarySession(
        ssid=ssid,
        llm=llm,
        document_path=document_path,
        summary_chain_type=summary_chain_type,
        first_page=first_page,
        last_page=last_page,
    )
    summary: str = summary_session.summarize_documents()
    return summary, ssid
