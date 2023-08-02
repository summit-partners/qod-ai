#!/usr/bin/env python3

from typing import Union, Type, List
from qod.base_data_types import BaseAttributeType
import typer
import time

from langchain.llms import GPT4All, LlamaCpp, OpenAI

# from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain import LLMChain
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
    chunks, _ = chunk_documents(path=doc_path, chunk_size=3000, chunk_overlap=100)
    """
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
        """
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
    """
    prompt_template = (
        "Write a concise summary of the following text."
        "The summary must be in a single block of text."
        "Do not add explanations, only provide a summary."
        "Do not add any information that is not specified in the text."
        "\n"
        "Text: {text}"
        "\n"
        "Summary:"
    )
    """
    # PROMPT = PromptTemplate(
    #         template=prompt_template, input_variables=["text"]
    # )
    """
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
        "Do not address me."
        "Do not insert any information that is not clearly specified \
in the existing summary or in the context."
        "\n"
        "\n"
        "Refined summary:"
    )
    """
    # refine_prompt = PromptTemplate(
    #     input_variables=["existing_answer", "text"],
    #     template=refine_template,
    # )

    #######
    refine_template_me = (
        "Refine the existing summary by enriching it using information \
you can extract from the additional context.\n"
        "If the additional context is not useful, return the existing \
summary unmodified.\n"
        "The refined summary must keep the core information communicated \
within the existing summary\n"
        "The refined summary can only include information specifically \
mentionned in the existing summary or the additional context.\n"
        "Your answer must not refer the additional context nor include \
any comments, explanation, or notes from you to me.\n"
        "The refined summary must keep a formal and mostly descriptive tone. \
It cannot include any dialog unless it is used as a citation.\n\n"
        "The desired size for the refined summary is to contain 3000 words.\n\n"
        "existing summary: {existing_answer}\n\n"
        "additional context: {text}\n\n"
        "refined summary:"
    )
    refine_prompt_me = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template_me,
    )
    start_template = (
        "Write a concise summary of the following text: \n\n"
        "text: {text} \n\n"
        "If the text is truncated, do not extend it. Only return the summary.\n\n"
        "Summary: "
    )
    start_prompt_me = PromptTemplate(input_variables=["text"], template=start_template)
    collapse_template = (
        "Collapse the following text: \n\n"
        "text: {text}\n\n"
        "Ignore incomplete sentences at the begining of \
the text or at the end of the text.\n"
        "Do not extend the text. Only return its compressed version..\n\n"
        "Collapsed text: "
    )
    collapse_prompt = PromptTemplate(
        input_variables=["text"], template=collapse_template
    )
    #######

    # Creating the chain
    # chain = load_summarize_chain(
    #     llm,
    #     chain_type="refine",
    #     return_intermediate_steps=True,
    #     question_prompt=start_prompt_me,  # PROMPT,
    #     refine_prompt=refine_prompt_me,  # refine_prompt,
    # )

    # Assemble the summary
    answer_start = time.time()
    # result = chain({"input_documents": chunks}, return_only_outputs=True)
    answer_end = time.time()

    # Print the summary
    # print()
    # print(f"{yellow}Intermediate steps ")
    # for inter in result["intermediate_steps"]:
    #     print(f"{yellow}    {inter}")
    # print(f"{blue}Answer: " + result["output_text"])
    # print(f"{red}Time: {round(answer_end - answer_start, 2)} sec")
    # print(f"{green}")

    ##################
    # refine_template_me = (
    #     "Refine the existing summary by enriching it using information \
    # from the additional context.\n"
    #     "If the additional context is not useful, return the existing text.\n"
    #     "Prioritize information from the summary.\n"
    #     "Do not over compress the refined summary.\n"
    #     "Deprioritized information coming from  incomplete sentences of \
    # the additional context.\n"
    #      "Only return the refined summary produced\n\n"
    #     "The refined summary must keep a formal and mostly descriptive tone. \
    # It cannot include any dialog unless it is used as a citation."
    #     "existing summary: {existing_answer}\n\n"
    #     "additional context: {text}\n\n"
    #     "refined summary:"
    # )
    # refine_prompt_me = PromptTemplate(
    #     input_variables=["existing_answer", "text"],
    #     template=refine_template_me,
    # )
    # print(f"refine_prompt_me: {refine_prompt_me}")
    # print()
    # print()
    llm_chain = LLMChain(prompt=refine_prompt_me, llm=llm)
    llm_collapse = LLMChain(prompt=collapse_prompt, llm=llm)
    """
    start_prompt = PromptTemplate(
        input_variables=["text"],
        template="Write a concist summary of the following text: \n\n \
text: {text}\
\n\n \
If the text is truncated, do not extend it. Only return the summary\n\n \
Summary: ",
    )
    """
    llm_chain_start = LLMChain(prompt=start_prompt_me, llm=llm)
    temp_summary = ""
    answer_start = time.time()
    for index, chunk in enumerate(chunks):
        text = chunk.page_content.replace("\n", " ")
        if index == 0:
            print(f"{yellow}Start Prompt: {start_prompt_me.format(text=text)} {green}")
            nb_token = llm.get_num_tokens(start_prompt_me.format(text=text))
            print(f"{blue}Number of prompt token: {nb_token}{green}")
            print()
            print()
            resu = llm_chain_start.run(text)
            print(f"{red} Resu: {resu} {green}")
            print(f"{blue}Number of result token: {llm.get_num_tokens(resu)}{green}")
            print()
            print()
            temp_summary = resu

        else:
            question = {"existing_answer": temp_summary, "text": text}
            my_prompt = refine_prompt_me.format(
                existing_answer=question["existing_answer"], text=question["text"]
            )
            print(f"{yellow}Prompt: {my_prompt}{green}")
            nb_token = llm.get_num_tokens(
                refine_prompt_me.format(
                    existing_answer=question["existing_answer"], text=question["text"]
                )
            )
            print(f"{blue}Number of prompt token: {nb_token}{green}")
            if (
                llm.get_num_tokens(
                    refine_prompt_me.format(
                        existing_answer=question["existing_answer"],
                        text=question["text"],
                    )
                )
                > llm.n_ctx
            ):
                print(f"{yellow}Warning - Collapsing context{green}")
                my_prompt = collapse_prompt.format(text=question["text"])
                print(f"{yellow}Prompt: {my_prompt}{green}")
                new_context = llm_collapse.run(question["text"])
                print(f"{yellow}New context: {new_context}{green}")
                question["text"] = new_context
                my_prompt = refine_prompt_me.format(
                    existing_answer=question["existing_answer"], text=question["text"]
                )
                print(f"{yellow}Prompt: {my_prompt}{green}")
                nb_token = llm.get_num_tokens(
                    refine_prompt_me.format(
                        existing_answer=question["existing_answer"],
                        text=question["text"],
                    )
                )
                print(f"{blue}Number of prompt token: {nb_token}{green}")
            # print(f"Question: {question}")
            result_me = llm_chain.run(question)
            temp_summary = result_me
            print(f"{red}result_me: {result_me}{green}")
            print(
                f"{blue}Number of result token: {llm.get_num_tokens(result_me)}{green}"
            )
            print()
            print()
            # Collapse
            if llm.get_num_tokens(temp_summary) > 1000:
                print(f"{yellow}Warning - Collapsing the temporary summary{green}")
                temp_summary = llm_collapse.run(temp_summary)
                print(
                    f"{blue}Number of result token: \
{llm.get_num_tokens(temp_summary)}{green}"
                )

        # inpu("Wait")
    answer_end = time.time()
    print(f"{blue}: Final answer: {result_me} {green}")
    print(f"{red}Time: {round(answer_end - answer_start, 2)} sec {green}")

    final_template = (
        "Improve the format of the following summary \
without adding external information or context.\n"
        "You can reformat and rearrange the text, but you cannot add anything new."
        "The desired format is a block of text including multiple paragraphs.\n"
        "The flow of the text must be good.\n\n"
        "text: {text}\n\n"
        "Improved text:"
    )
    final_prompt = PromptTemplate(input_variables=["text"], template=final_template)
    print(f"{yellow}Final Prompt: {final_prompt.format(text=result_me)} {green}")
    llm_chain_final = LLMChain(prompt=final_prompt, llm=llm)
    resu = llm_chain_final.run(result_me)
    answer_end = time.time()
    print(f"{blue}: Enhanced answer: {resu} {green}")
    print(f"{red}Time: {round(answer_end - answer_start, 2)} sec {green}")

    print("Comparison")
    # print(f"{yellow} Langchain: {result[output_text]} {green}")
    print(f"{blue} Custom: {result_me} {green}")
    print(f"{red} Custom++: {resu}")


if __name__ == "__main__":
    typer.run(main)
