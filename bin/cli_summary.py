#!/usr/bin/env python3

from typing import Union, Type, List
from qod.base_data_types import BaseAttributeType
import typer
import time

from langchain.llms import GPT4All, LlamaCpp, OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from qod.llm_data_types import LLMType, LLMAttributes
from qod.langchain_wrappers import get_llm, chunk_documents


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
    doc_path = input("Enter the path of the document to summarize\n")
    chunks, _ = chunk_documents(path=doc_path, chunk_size=1000, chunk_overlap=100)
    if chunks and doc_path.endswith(".pdf"):
        first_page = input("Indicate the first page of the document to summarize: ")
        last_page = input("Indicate the last page of the document to summarize: ")
        print(chunks[:5])
        print(first_page, int(first_page))
        print(last_page, int(last_page))
        chunks = [
            c
            for c in chunks
            if (
                c.metadata["page"] >= int(first_page) - 1
                and c.metadata["page"] < int(last_page)
            )
        ]  # 19-33 # 19-85
    return chunks


def main():
    red = "\033[0;31m"
    yellow = "\033[0;33m"
    green = "\033[0;32m"
    blue = "\033[0;34m"

    # Choose the LLM to use for Q&A
    llm = select_llm()

    # Get segmented document
    chunks = select_document()
    print(f"Document segmented into {len(chunks)} chunks")

    # Printing the segmented text (TODO - Improve)
    print("Text to parse")
    colors = [red, yellow, blue, green]
    for ind, c in enumerate(chunks):
        color = colors[ind % len(colors)]
        chunk_text = c.page_content.replace("\n", " ")
        print(f"{color} {chunk_text}")
    print(f"{green}\n")

    # Setting the prompts
    prompt_template = (
        "Write a concise summary of the following text."
        "The summary must be in a single block of text."
        "Do not add explanations, only provide a summary"
        "\n"
        "Text: {text}"
        "\n"
        "Summary:"
    )
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a \
certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "You cannot refine the summary using content that is not in this context"
        "\n"
        "context: {text}\n"
        "\n"
        "Given the new context, refine the original summary."
        "If the context isn't useful, return the original summary."
        "Only return the refined summary."
        "Do not provide explanation about your choices."
        "Do not explain the refinement."
        "Do not address me"
        "\n"
        "\n"
        "Refined summary:"
    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    # Creating the chain
    chain = load_summarize_chain(
        llm,
        chain_type="refine",
        return_intermediate_steps=True,
        question_prompt=PROMPT,
        refine_prompt=refine_prompt,
    )

    # Assemble the summary
    answer_start = time.time()
    result = chain({"input_documents": chunks}, return_only_outputs=True)
    answer_end = time.time()

    # Print the summary
    print()
    print(f"{yellow}Intermediate steps ")
    for inter in result["intermediate_steps"]:
        print(f"{yellow}    {inter}")
    print(f"{blue}Answer: " + result["output_text"])
    print(f"{red}Time: {round(answer_end - answer_start, 2)} sec")
    print(f"{green}")


if __name__ == "__main__":
    typer.run(main)
