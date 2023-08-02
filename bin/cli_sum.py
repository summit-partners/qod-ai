#!/usr/bin/env python3

from typing import Union, Type, List
from qod.base_data_types import BaseAttributeType
import typer
import secrets

from langchain.llms import GPT4All, LlamaCpp, OpenAI
from langchain.schema import Document

from qod.llm_data_types import LLMType, LLMAttributes
from qod.langchain_wrappers import get_llm, chunk_documents
from qod.summary_session import SummarySession
from qod.chain_data_types import ChainType


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


def select_document() -> List[Document]:
    """Allow a user to import the document to summarize"""
    document_path = input("Enter the path of the document to summarize\n")
    documents, _ = chunk_documents(path=document_path, chunk_size=1000, chunk_overlap=0)
    return documents


def main():
    red = "\033[0;31m"
    yellow = "\033[0;33m"
    green = "\033[0;32m"
    blue = "\033[0;34m"

    # Choose the LLM to use for Q&A
    llm = select_llm()

    # Get segmented document
    documents = select_document()
    print(f"Document segmented into {len(documents)} chunks")

    # Printing the segmented text (TODO - Improve)
    print("Segmented documents")
    colors = [red, yellow, blue, green]
    for ind, doc in enumerate(documents):
        color = colors[ind % len(colors)]
        text = doc.page_content
        print(f"{color} {text}")
    print(f"{green}\n")

    summary_session = SummarySession(
        ssid="ssid_" + secrets.token_hex(12),
        llm=llm,
        document_path="",  # TODO
        documents=documents,
        chain_type=ChainType.MAP_REDUCE,
    )
    summary = summary_session.summarize_documents()
    print(f"{blue}Summary: {summary}{green}")


if __name__ == "__main__":
    typer.run(main)
