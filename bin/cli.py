#!/usr/bin/env python3

from typing import Union, Optional, Tuple, Type
from qod.base_data_types import BaseAttributeType
import typer
import sys
import time
import secrets

from langchain.embeddings import (
    LlamaCppEmbeddings,
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
)
from langchain.llms import GPT4All, LlamaCpp, OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain.vectorstores import Chroma

from qod.embeddings_data_types import EmbeddingsType, EmbeddingsAttributes
from qod.llm_data_types import LLMType, LLMAttributes
from qod.chain_data_types import ChainType, ChainAttributes
from qod.langchain_wrappers import get_llm, get_embeddings, get_chain, get_vectorstore
from qod.chat_session import ChatSession
from qod.display_msg import (
    cli_input,
    display_error,
    display_cli_notification,
    display_chain_final,
    display_chain_temp,
)


def get_selected_attribute_value(
    attribute_function: str, attribute_types: Type[BaseAttributeType]
):
    """Allow a user to select a value over the value of an enum for a session attribute
    :param attribute_function: A string capturing the function of the attribute
    (e.g., LLM)
    :param attribute_types: A type of enum that captures the options for a session
    attribute
    :return The value of an enum from the specified type
    """
    msg = f"Enter the number for the {attribute_function} you wish to select?\n"
    for enum_type in attribute_types:
        value = enum_type.value
        name = enum_type.get_attributes().friendly_name
        msg += f"    {value}: {name}\n"
    selected_value = cli_input(msg)
    return int(selected_value)


def select_llm() -> Union[GPT4All, LlamaCpp, OpenAI]:
    """Allow a user to select the LLM to use to answer questions
    :return An LLM object
    """
    valid_type = False
    while not valid_type:
        llm_type_value = get_selected_attribute_value(
            attribute_function="LLM", attribute_types=LLMType
        )
        if llm_type_value in LLMType.__members__.values():
            valid_type = True
        else:
            display_error("Invalid value - Please enter one of the provided choices.")
    llm_attributes: LLMAttributes = LLMType(llm_type_value).get_attributes()
    return get_llm(attr=llm_attributes)


def select_embeddings() -> (
    Union[LlamaCppEmbeddings, HuggingFaceEmbeddings, OpenAIEmbeddings]
):
    """Allow a user to select the embeddings to use to convert text into vectors
    :return An embeddings object
    """
    valid_type = False
    while not valid_type:
        llm_type_value = get_selected_attribute_value(
            attribute_function="embeddings", attribute_types=EmbeddingsType
        )
        if llm_type_value in EmbeddingsType.__members__.values():
            valid_type = True
        else:
            display_error("Invalid value - Please enter one of the provided choices.")
    embeddings_attributes: EmbeddingsAttributes = EmbeddingsType(
        llm_type_value
    ).get_attributes()
    return get_embeddings(attr=embeddings_attributes)


def select_chain(
    llm, vectorstore
) -> Union[ConversationalRetrievalChain, BaseConversationalRetrievalChain]:
    """Allow a user to select the type of chain to use when answering questions
    :return A conversational retrieval chain object
    """
    valid_type = False
    while not valid_type:
        chain_type_value = get_selected_attribute_value(
            attribute_function="chain", attribute_types=ChainType
        )
        if chain_type_value in ChainType.__members__.values():
            valid_type = True
        else:
            display_error("Invalid value - Please enter one of the provided choice.")
    chain_attributes: ChainAttributes = ChainType(chain_type_value).get_attributes()
    return get_chain(chain_attributes, llm, vectorstore)


def load_create_vectorstore(
    embeddings: Union[LlamaCppEmbeddings, HuggingFaceEmbeddings, OpenAIEmbeddings]
) -> Tuple[Chroma, str]:
    """Allow a user to select an existing vectorstore or to create a new one from
    documents
    :param embeddings: Embedding function to convert text into vectors for the
    vectorstore
    """
    db_directory: Optional[str] = None
    documents_directory: Optional[str] = None
    valid_type = False
    while not valid_type:
        choice = cli_input(
            "Enter 'load' if you want to load an existing vector store or 'new' to \
create a new one\n"
        )
        if choice in ["load", "new"]:
            valid_type = True
        else:
            display_error("Invalid value - Please enter one of the provided choices.")
            continue
        if choice == "load":
            db_directory = cli_input("Enter the path of the vectorstore directory\n")
        if choice == "new":
            documents_directory = cli_input(
                "Enter the path of the documents directory\n"
            )
            naming_choice = cli_input(
                "Do you wish to name the directory in which the vectorstore will \
be stored? (y: yes, anything else: no)\n"
            )
            if naming_choice == "y":
                db_directory = cli_input(
                    "Enter the name of the directory where the vectorstore will \
be stored\n"
                )
    return get_vectorstore(
        embeddings=embeddings,
        db_directory=db_directory,
        documents_directory=documents_directory,
    )


def main():
    # Choose the LLM to use for Q&A
    llm = select_llm()

    # Choose the LLM to index the documents
    embedding = select_embeddings()

    # Create or load an indexed DB
    vectorstore, db_directory = load_create_vectorstore(embeddings=embedding)
    display_cli_notification(f"Embeddings stored at: {db_directory}")

    # Select the chain type
    qa = select_chain(llm, vectorstore)

    chat_session = ChatSession(
        csid="csid_" + secrets.token_hex(12),
        llm=llm,
        embeddings=embedding,
        chain=qa,
        vectorstore=vectorstore,
        history=[],
        documents_directory="",  # TODO
        db_directory=db_directory,
        llm_type=1,  # TODO
        embeddings_type=1,  # TODO
        chain_type=1,  # TODO
    )

    chat_history = []
    display_cli_notification(
        "---------------------------------------------------------------------"
    )
    display_cli_notification(
        "The system is now ready to interact with your document. Please ask your \
questions."
    )
    display_cli_notification('Enter "exit" to stop')
    display_cli_notification('Enter "flush" to flush the history')
    display_cli_notification(
        'Enter "llm" to change the language model used to ask questions'
    )
    display_cli_notification(
        'Enter "chain" to change the chain type used to ask questions'
    )
    display_cli_notification(
        "---------------------------------------------------------------------"
    )

    while True:
        query = cli_input("Question: ")

        if query == "exit" or query == "quit" or query == "q" or query == "f":
            display_cli_notification("Exiting")
            sys.exit()
        if query == "":
            continue
        if query == "flush":
            display_cli_notification("Erasing the history")
            chat_history = []
            continue
        if query == "llm":
            llm = select_llm()
            continue
        if query == "chain":
            qa = select_chain(llm, vectorstore)
            continue

        answer_start = time.time()
        result = chat_session.chain({"question": query, "chat_history": chat_history})
        answer_end = time.time()
        display_chain_final(f"Answer: {result['answer']}")
        display_chain_final(f"Time: {round(answer_end - answer_start, 2)} sec")
        display_chain_temp("sources: ")
        for s in result.get("source_documents", []):
            display_chain_temp(f"    {s}")
        chat_history.append((query, result["answer"]))


if __name__ == "__main__":
    typer.run(main)
