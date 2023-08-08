#!/usr/bin/env python3

from typing import Union, Type
from qod.base_data_types import BaseAttributeType
import typer
import secrets
import time

from langchain.llms import GPT4All, LlamaCpp, OpenAI

from qod.llm_data_types import LLMType, LLMAttributes
from qod.langchain_wrappers import get_llm
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


def main():
    red = "\033[0;31m"
    # yellow = "\033[0;33m"
    green = "\033[0;32m"
    blue = "\033[0;34m"

    # Choose the LLM to use for Q&A
    llm = select_llm()

    # Get segmented document
    document_path = input("Enter the path of the document to summarize\n")

    summary_session = SummarySession(
        ssid="ssid_" + secrets.token_hex(12),
        llm=llm,
        document_path=document_path,
        chain_type=ChainType.STUFFED,
    )
    start_stuffed = time.time()
    summary_stuffed = summary_session.summarize_documents()
    end_stuffed = time.time()
    print(f"{blue}Summary: {summary_stuffed}{green}")
    print(f"{red}Time: {round(end_stuffed - start_stuffed, 2)}{green}")


if __name__ == "__main__":
    typer.run(main)
