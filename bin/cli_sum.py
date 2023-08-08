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


reset = "\033[0m"
red = "\033[0;31m"
green = "\033[0;32m"
yellow = "\033[0;33m"
blue = "\033[0;34m"
purple = "\033[0;35m"
cyan = "\033[0;36m"
white = "\033[0;37m"


b_red = "\033[1;31m"
b_green = "\033[1;32m"
b_yellow = "\033[1;33m"
b_blue = "\033[1;34m"
b_purple = "\033[1;35m"
b_cyan = "\033[1;36m"
b_white = "\033[1;37m"


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
    print(
        f"{b_cyan}Enter the number for the \
{attribute_function} you wish to select?{reset}"
    )
    for enum_type in attribute_types:
        print(
            f"{cyan}    {enum_type.value}: \
{enum_type.get_attributes().friendly_name}{reset}"
        )
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
            print(
                f"{red}Invalid value - Please enter one of the provided choices.{reset}"
            )
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
            print(
                f"{red}Invalid value - Please enter one of the provided choice.{reset}"
            )


def select_document() -> Tuple[str, Optional[int], Optional[int]]:
    first_page: Optional[int] = None
    last_page: Optional[int] = None
    document_path = input(
        f"{b_cyan}Enter the path of the document to summarize\n{reset}"
    )
    if document_path.endswith(".pdf"):
        first_page = int(
            input(
                f"{b_cyan}Indicate the first page of the \
document to summarize: \n{reset}"
            )
        )
        last_page = int(
            input(
                f"{b_cyan}Indicate the last page of the \
document to summarize: \n{reset}"
            )
        )
    return document_path, first_page, last_page


def display_commands():
    print(
        f"{white}---------------------------------------------------------------------"
    )
    print('Enter "exit" to stop')
    print('Enter "summarize" to summarize your document')
    print('Enter "llm" to select another language model')
    print('Enter "chain" to select another chain')
    print('Enter "document" to select a new document to summarize')
    print(
        'Enter "chunks" to see the most recent document \
summarized decomposed into chunks'
    )

    print('Enter "help" to display this list of commands')
    print("---------------------------------------------------------------------\n")
    print(f"{reset}")


def summarize(summary_session: Optional[SummarySession] = None) -> SummarySession:
    """ """
    ssid: str = "ssid_" + secrets.token_hex(12)

    # The user must select and LLM, a docuemnt and a chain
    # for her first request
    if summary_session is None:
        # Choose the LLM to use for Q&A
        llm = select_llm()
        if not isinstance(llm, LlamaCpp):
            raise Exception(
                f"{red} Only Llama model are currently supported \
for summarization. Please select another model.{reset}"
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
    print(f"{blue}\n\nFinal summary: {summary}{reset}")
    print(f"{red}Time: {round(end_time - start_time, 2)}{reset}")
    return summary_session


def change_llm(summary_session: SummarySession) -> SummarySession:
    """ """
    llm = select_llm()
    if not isinstance(llm, LlamaCpp):
        raise Exception(
            f"{red} Only Llama model are currently supported \
for summarization. Please select another model.{reset}"
        )
    summary_session.llm = llm
    return summary_session


def change_chain(summary_session: SummarySession) -> SummarySession:
    """ """
    summary_session.summary_chain_type = select_summary_chain_type()
    return summary_session


def change_document(summary_session: SummarySession) -> SummarySession:
    """ """
    path, first_page, last_page = select_document()
    summary_session.document_path = path
    summary_session.first_page = first_page
    summary_session.last_page = last_page
    return summary_session


def main():
    summary_session: SummarySession = summarize()
    display_commands()
    while True:
        command: str = input(f"{b_cyan}Enter your command: {reset}")

        # Exiting the system
        if command == "exit":
            print(f"{red}Exiting{reset}")
            sys.exit()

        # Summarize a document
        if command == "summarize":
            summary_session = summarize(summary_session)
            continue

        # Select a new LLM
        if command == "llm":
            summary_session = change_llm(summary_session)
            llm_name = summary_session.llm.get_attributes().friendly_name
            print(f"{white}The llm {llm_name} has been selected\n{reset}")
            continue

        # Select a new chain
        if command == "chain":
            summary_session = change_chain(summary_session)
            chain_name = (
                summary_session.summary_chain_type.get_attributes().friendly_name
            )
            print(f"{white}The chain {chain_name} has been selected\n{reset}")
            continue

        # Select a new document
        if command == "document":
            summary_session = change_document(summary_session)
            print(
                f'{white}The document "{summary_session.document_path}" \
(pages {summary_session.first_page} to {summary_session.last_page}) \
has been selected\n{reset}'
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

        print(
            f"{red}This command is not recognized. \
Please use a command from the folowing list: {reset}"
        )
        display_commands()


if __name__ == "__main__":
    typer.run(main)
