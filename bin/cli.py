#!/usr/bin/env python3

from typing import Union, Optional, Tuple, Type
from qod.base_data_types import BaseAttributeType
import typer
import sys
import time

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
    print(f"Enter the number for the {attribute_function} you wish to select?")
    for enum_type in attribute_types:
        print(f"    {enum_type.value}: {enum_type.get_attributes().friendly_name}")
    selected_value = input("")
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
            print("Invalid value - Please enter one of the provided choices.")
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
            print("Invalid value - Please enter one of the provided choices.")
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
            print("Invalid value - Please enter one of the provided choice.")
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
        print(
            "Enter 'load' if you want to load an existing vector store or 'new' to \
create a new one"
        )
        choice = input("")
        if choice in ["load", "new"]:
            valid_type = True
        else:
            print("Invalid value - Please enter one of the provided choices.")
            continue
        if choice == "load":
            print("Enter the path of the vectorstore directory")
            db_directory = input("")
        if choice == "new":
            print("Enter the path of the documents directory")
            documents_directory = input("")
            print(
                "Do you wish to name the directory in which the vectorstore will \
be stored? (y: yes, anything else: no)"
            )
            naming_choice = input("")
            if naming_choice == "y":
                print(
                    "Enter the name of the directory where the vectorstore will \
be stored"
                )
                db_directory = input("")
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
    print(f"Embeddings stored at: {db_directory}")

    # Select the chain type
    qa = select_chain(llm, vectorstore)

    red = "\033[0;31m"
    yellow = "\033[0;33m"
    green = "\033[0;32m"
    blue = "\033[0;34m"

    chat_history = []
    print(
        f"{yellow}---------------------------------------------------------------------"
    )
    print(
        "The system is now ready to interact with your document. Please ask your \
questions."
    )
    print('Enter "exit" to stop')
    print('Enter "flush" to flush the history')
    print('Enter "llm" to change the language model used to ask questions')
    print('Enter "chain" to change the chain type used to ask questions')
    print("---------------------------------------------------------------------")

    while True:
        query = input(f"{green}Prompt: ")

        if query == "exit" or query == "quit" or query == "q" or query == "f":
            print("Exiting")
            sys.exit()
        if query == "":
            continue
        if query == "flush":
            print("Erasing the history")
            chat_history = []
            continue
        if query == "llm":
            llm = select_llm()
            continue
        if query == "chain":
            qa = select_chain(llm)
            continue

        answer_start = time.time()
        result = qa({"question": query, "chat_history": chat_history})
        answer_end = time.time()
        print()
        print(f"{blue}Answer: " + result["answer"])
        print(f"{red}Time: {round(answer_end - answer_start, 2)} sec")
        print(f"{yellow}sources: ")
        for s in result.get("source_documents", []):
            print(f"{yellow}    {s}")
        chat_history.append((query, result["answer"]))


if __name__ == "__main__":
    typer.run(main)
