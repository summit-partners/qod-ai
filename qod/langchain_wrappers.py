import os
import hashlib
import shutil
import sys
import re
from typing import Union, List, Optional, Tuple

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.schema import Document

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
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from qod.embeddings_data_types import EmbeddingsAttributes, EmbeddingsFamily
from qod.llm_data_types import LLMAttributes, LLMFamily
from qod.chain_data_types import ChainAttributes, ChainType


__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


def chunk_documents(
    path, chunk_size=500, chunk_overlap=100
) -> Tuple[List[Document], List[str]]:
    """Convert pdf, docx and txt document in a directory into text chunks
    :param path: Path of the document to process or a folder contaning
    the documents to process
    :param chunk_size: maximum amount of chracter in a chunk
    :param chunk_overlap: Amount of character overlaping between successive chunks
    :return chunks: List of text chunks
    :return files: List of name of the files processed
    """
    processed_files = []
    file_paths = []
    loaders = []
    first_page = 0
    last_page = float("Inf")
    # Getting the name of the file to process
    if os.path.isdir(path):
        file_paths = [os.path.join(path, file_name) for file_name in os.listdir(path)]
    elif os.path.isfile(path):
        file_paths = [path]
        if path.endswith(".pdf"):
            first_page = int(
                input("Indicate the first page of the document to summarize: ")
            )
            last_page = int(
                input("Indicate the last page of the document to summarize: ")
            )

    else:
        raise ValueError(
            f"{path} is not a valid path for the documents \
to proceed"
        )
    for file_path in file_paths:
        print(f"Segmenting file {file_path} into chunks")
        if file_path.endswith(".pdf"):
            loader_pdf = PyPDFLoader(file_path)
            # print(f"type(loader_pdf): {type(loader_pdf)}")
            # print(f"type(loader_pdf.load()): {type(loader_pdf.load())}")
            # print(f"type(loader_pdf.load()[0]): {type(loader_pdf.load()[0])}")
            # print(f"loader_pdf.load()[0]: {loader_pdf.load()[0]}")
            pdf_docs = loader_pdf.load()
            content = ""
            for doc in pdf_docs:
                if doc.metadata["page"] >= int(first_page) - 1 and doc.metadata[
                    "page"
                ] < int(last_page):
                    content += doc.page_content
            # Process content
            # Replace single '\n' with space
            # content = re.sub('(?<!\n)\n(?!\n)', ' ', content)
            # Replace any '\n' that is not after the end of
            # a sentence (., !, ?) by a space
            content = re.sub("(?<=[^.\n!?])\n", " ", content)
            # Replace any sequence of '\n' with more than 2 '\n' by a single '\n'
            content = re.sub("\n{2,}", "\n\n", content)
            # print(f"content: {content}")

            aggregated_pdf_doc = Document(page_content=content)
            # loaders.extend(loader_pdf.load())
            loaders.extend([aggregated_pdf_doc])
            processed_files.append(file_path)
        elif file_path.endswith(".docx") or file_path.endswith(".doc"):
            loader_doc = Docx2txtLoader(file_path)
            loaders.extend(loader_doc.load())
            processed_files.append(file_path)
        elif file_path.endswith(".txt"):
            loader_txt = TextLoader(file_path)
            loaders.extend(loader_txt.load())
            processed_files.append(file_path)

    char_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap  # , separator = "\n"
    )
    chunks = char_text_splitter.split_documents(loaders)
    return chunks, processed_files


def get_embeddings(
    attr: EmbeddingsAttributes,
) -> Union[LlamaCppEmbeddings, HuggingFaceEmbeddings, OpenAIEmbeddings]:
    """Get an embedding object from embedding attributes
    :param attr: Attributes of the embedding object
    :return An embedding object
    """
    if attr.family == EmbeddingsFamily.HUGGING_FACE:
        return HuggingFaceEmbeddings(
            model_name=attr.model, model_kwargs={"device": "cuda"}
        )

    if attr.family == EmbeddingsFamily.LLAMA_CPP:
        return LlamaCppEmbeddings(
            model_path=attr.model,
            n_ctx=2048,
            n_threads=-1,
            n_batch=2048,
            client=None,
            n_parts=-1,
            seed=-1,
            f16_kv=True,
            logits_all=False,
            vocab_only=False,
            use_mlock=False,
            n_gpu_layers=40,
        )

    if attr.family == EmbeddingsFamily.OPEN_AI:
        return OpenAIEmbeddings(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            model=attr.model,
            client=None,
        )
    raise Exception(f"Could not load the embedding for the family {attr.family}")


def get_llm(attr: LLMAttributes) -> Union[GPT4All, LlamaCpp, OpenAI]:
    """Get an LLM object from LLM attributes
    :param attr: Attributes of the LLM object
    :return A LLM object
    """
    if attr.family == LLMFamily.GPT4ALL:
        return GPT4All(
            model=attr.model,
            callbacks=[StreamingStdOutCallbackHandler()],
            verbose=True,
            max_tokens=2048,
            n_threads=-1,
            backend=None,
            n_parts=-1,
            seed=-1,
            f16_kv=False,
            logits_all=False,
            vocab_only=False,
            use_mlock=False,
            embedding=False,
            n_batch=-1,
        )

    if attr.family == LLMFamily.LLAMA:
        return LlamaCpp(
            model_path=attr.model,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=True,
            n_ctx=2048,
            max_tokens=2048,
            temperature=0,
            n_threads=-1,
            n_batch=512,
            client=None,
            n_parts=-1,
            seed=-1,
            f16_kv=True,
            logits_all=False,
            vocab_only=False,
            use_mlock=False,
            n_gpu_layers=75,
            suffix=None,
            logprobs=None,
        )

    if attr.family == LLMFamily.OPEN_AI:
        return OpenAI(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            model=attr.model,
            temperature=0,
            client=None,
        )
    raise Exception(f"Could not load the LLM for the family {attr.family}")


def get_chain(
    attr: ChainAttributes, llm, vectorstore
) -> Union[ConversationalRetrievalChain, BaseConversationalRetrievalChain]:
    """Get a conversational retrieval chain for chain type
    :param attr: Attributes of chain
    :param llm: LLM object to use in the chain
    :param vectorstore: Vectorstore from which documents are retrieved
    :return A conversational retrieval chain
    """
    if attr.type == ChainType.STUFFED:
        return ConversationalRetrievalChain.from_llm(
            llm,
            vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5}),
            condense_question_llm=llm,
            return_source_documents=True,
        )
    return ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5}),
        question_generator=LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT),
        combine_docs_chain=load_qa_chain(llm, chain_type=attr.model),
    )


def get_vectorstore(
    embeddings: Union[LlamaCppEmbeddings, HuggingFaceEmbeddings, OpenAIEmbeddings],
    db_directory: Optional[str] = None,
    documents_directory: Optional[str] = None,
) -> Tuple[Chroma, str]:
    """Get a vectorstore from documents (new) or a directory (loaded)
    :param embeddings: Embeddings used to convert the document into vectors
    :param db_directory: Path of the directory where the vectorstore is stored
    :param documents_directory: Path of the directory where the documents are stored
    :return vectorstore: A vectorstore
    :return db_directory: The path to the directory used to store the vectorstore
    """
    if documents_directory is not None:
        print(f"Converting documents in {documents_directory:} into text chunks")
        documents, files = chunk_documents(path=documents_directory)
        # Create a directory name from the vectorstore if none was provided
        if db_directory is None:
            db_directory = hashlib.md5(
                documents_directory.join(sorted(files)).encode()
            ).hexdigest()
        # Erase the DB directory if it already exists
        if os.path.exists(db_directory) and os.path.isdir(db_directory):
            print(
                f"Erasing the existing directory {db_directory} to store the embeddings"
            )
            shutil.rmtree(db_directory)

        print("Creating embeddings for the documents and storing then in a vectorstore")
        vectorstore = Chroma.from_documents(
            documents,
            embedding=embeddings,
            persist_directory=db_directory,
        )
    else:
        if db_directory is None:
            raise Exception(
                "You must provide at least a vectorstore directory to \
load pre-calculated embeddings or a documents directory to calculate and \
store new embeddings"
            )
        print(
            "Loading an existing vectorstore (CAUTION: the embedding function must \
be the same than the one used when creating the vectorstore)"
        )
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=db_directory,
        )
    return vectorstore, db_directory
