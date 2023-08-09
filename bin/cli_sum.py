#!/usr/bin/env python3

from typing import Union, Type, Optional, Tuple
from qod.base_data_types import BaseAttributeType
import typer
import secrets
import time
import sys

from langchain.llms import GPT4All, LlamaCpp, OpenAI

from qod.llm_data_types import LLMType, LLMAttributes
from qod.langchain_wrappers import get_llm
from qod.summary_session import SummarySession
from qod.chain_data_types import SummaryChainType
from qod.display_msg import (
    cli_input,
    display_error,
    display_cli_notification,
    display_chain_final,
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


def select_summary_chain_type() -> SummaryChainType:
    """Allow a user to select the type of chain to sumarizing a document
    :return A type of chain
    """
    while True:
        summary_chain_type_value = get_selected_attribute_value(
            attribute_function="chain", attribute_types=SummaryChainType
        )
        if summary_chain_type_value in SummaryChainType.__members__.values():
            return SummaryChainType(summary_chain_type_value)
        else:
            display_error("Invalid value - Please enter one of the provided choice.")


def select_document() -> Tuple[str, Optional[int], Optional[int]]:
    """Allow a user to select a document to summarize"""
    first_page: Optional[int] = None
    last_page: Optional[int] = None
    document_path = cli_input("Enter the path of the document to summarize\n")
    if document_path.endswith(".pdf"):
        first_page = int(
            cli_input("Indicate the first page of the document to summarize: \n")
        )
        last_page = int(
            cli_input("Indicate the last page of the document to summarize: \n")
        )
    return document_path, first_page, last_page


def display_commands():
    """Display the list of commands that can be used in the CLI"""
    display_cli_notification(
        "---------------------------------------------------------------------"
    )
    display_cli_notification('Enter "exit" to stop')
    display_cli_notification('Enter "summarize" to summarize your document')
    display_cli_notification('Enter "llm" to select another language model')
    display_cli_notification('Enter "chain" to select another chain')
    display_cli_notification('Enter "document" to select a new document to summarize')
    display_cli_notification(
        'Enter "chunks" to see the most recent document \
summarized decomposed into chunks'
    )
    display_cli_notification('Enter "help" to display this list of commands')
    display_cli_notification(
        "---------------------------------------------------------------------\n"
    )


def summarize(summary_session: Optional[SummarySession] = None) -> SummarySession:
    """Assemble a summary for a document
    :return summary_session: A summary session
    """
    ssid: str = "ssid_" + secrets.token_hex(12)

    # The user must select and LLM, a docuemnt and a chain
    # for her first request
    if summary_session is None:
        # Choose the LLM to use for Q&A
        llm = select_llm()
        if not isinstance(llm, LlamaCpp):
            display_error(msg="", reset=False)
            raise Exception(
                "Only Llama model are currently supported \
for summarization. Please select another model."
            )
        # Get segmented document
        document_path, first_page, last_page = select_document()
        # Choose the chain type
        summary_chain_type = select_summary_chain_type()

        # Create the summary session
        summary_session = SummarySession(
            ssid=ssid,
            llm=llm,
            document_path=document_path,
            summary_chain_type=summary_chain_type,
            first_page=first_page,
            last_page=last_page,
        )

    # Assembling the summary
    start_time = time.time()
    summary = summary_session.summarize_documents()
    end_time = time.time()
    display_chain_final(f"\n\nFinal summary: {summary}")
    display_chain_final(f"Time: {round(end_time - start_time, 2)} sec")
    return summary_session


def change_llm(summary_session: SummarySession) -> SummarySession:
    """Enable the user to change the LLM used to assemble the summary
    :parameter summary_session: The most recent summary sessions
    :return The summary session updated with a new LLM
    """
    llm = select_llm()
    if not isinstance(llm, LlamaCpp):
        display_error(msg="", reset=False)
        raise Exception(
            "Only Llama model are currently supported \
for summarization. Please select another model."
        )
    summary_session.llm = llm
    return summary_session


def change_chain(summary_session: SummarySession) -> SummarySession:
    """Enable the user to change the chain used to assemble the summary
    :parameter summary_session: The most recent summary sessions
    :return The summary session updated with a new chain"""
    summary_session.summary_chain_type = select_summary_chain_type()
    return summary_session


def change_document(summary_session: SummarySession) -> SummarySession:
    """Enable the user to change the document to summarize
    :parameter summary_session: The most recent summary sessions
    :return The summary session updated with a new document"""
    path, first_page, last_page = select_document()
    summary_session.document_path = path
    summary_session.first_page = first_page
    summary_session.last_page = last_page
    return summary_session


def main():
    """Runs a CLI enabling a user to summarize documents using a LLM"""
    summary_session: SummarySession = summarize()
    display_commands()
    while True:
        command: str = cli_input("Enter your command: \n")

        # Exiting the system
        if command == "exit":
            display_cli_notification("Exiting")
            sys.exit()

        # Summarize a document
        if command == "summarize":
            summary_session = summarize(summary_session)
            continue

        # Select a new LLM
        if command == "llm":
            summary_session = change_llm(summary_session)
            llm_name = summary_session.llm.get_attributes().friendly_name
            display_cli_notification(f"The llm {llm_name} has been selected\n")
            continue

        # Select a new chain
        if command == "chain":
            summary_session = change_chain(summary_session)
            chain_name = (
                summary_session.summary_chain_type.get_attributes().friendly_name
            )
            display_cli_notification(f"The chain {chain_name} has been selected\n")
            continue

        # Select a new document
        if command == "document":
            summary_session = change_document(summary_session)
            display_cli_notification(
                f'The document "{summary_session.document_path}" \
(pages {summary_session.first_page} to {summary_session.last_page}) \
has been selected\n'
            )
            continue

        # Display document as chunks
        if command == "chunks":
            summary_session.display_chunked_documents()
            continue

        # Display commands
        if command == "help":
            display_commands()
            continue

        display_error(
            "This command is not recognized. \
Please use a command from the folowing list:"
        )
        display_commands()


if __name__ == "__main__":
    typer.run(main)
