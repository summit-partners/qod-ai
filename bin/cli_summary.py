#!/usr/bin/env python3

from typing import Union, Type
from qod.base_data_types import BaseAttributeType
import typer
import time

from langchain.llms import GPT4All, LlamaCpp, OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

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


def main():
    # Get document

    # Choose the LLM to use for Q&A
    # llm = select_llm()

    """
    # tokenizer = AutoTokenizer.from_pretrained("
        models/falcon/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3.ggccv1.q5_1.bin
        ")
    model = AutoModelForCausalLM.from_pretrained(
        "models/falcon/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3.ggccv1.q5_1.bin",
        # device_map=device_map,
        # torch_dtype=torch.float16,
        # max_memory=max_mem,
        # quantization_config=quantization_config,
        local_files_only=True
    )
    pipe = pipeline(
        "text-generation",
        model = model,
        # tokenizer = tokenizer,
        max_length = 512,
        temperature = 0.7,
        top_p = 0.95,
        repetition_penalty = 1.15
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    """

    red = "\033[0;31m"
    yellow = "\033[0;33m"
    green = "\033[0;32m"
    blue = "\033[0;34m"

    print(
        f"{yellow}---------------------------------------------------------------------"
    )
    print("Please enter the text you want to summarize")
    print('Enter "exit" to stop')
    # print('Enter "flush" to flush the history')
    # print('Enter "llm" to change the language model used to ask questions')
    # print('Enter "chain" to change the chain type used to ask questions')
    print("---------------------------------------------------------------------")

    prompt_template = (
        "Write a concise summary of the following:"
        "\n"
        "\n"
        "{text}"
        "\n"
        "\n"
        "Summary:"
    )

    # prompt_template="Summarize the following text: {text}"
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a \
certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary."
        "If the context isn't useful, return the original summary."
        "Only return the refined summary."
        "The refined summary must not contain more than 750 tokens."
        "\n"
        "\n"
        "Refined summary:"
    )
    """
    refine_template = (
            "Your job is to refine an existing summary by including \
some additional context."
            "Only use knowledge from the existing summary and the \
additional context provided."
            "If the additional context is not useful, return the existing \
summary as the refined summary."
            "The refined summary must not contain more than 750 tokens.\n\n"
            "Existing summary: {existing_answer}\n\n"
            "Additional context: {text}\n\n"
            "Refined summary: "

    )
    """
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )
    """
    chain = load_summarize_chain(
        llm,
        chain_type="refine",
        return_intermediate_steps=True,
        question_prompt=PROMPT,
        refine_prompt=refine_prompt,
    )
    """

    chunks, _ = chunk_documents("docs_hobbit", 500, 0)
    chunks = [
        c for c in chunks if (c.metadata["page"] >= 18 and c.metadata["page"] < 33)
    ]  # 33 # 85
    print(chunks)
    print("Text to parse")
    colors = [red, yellow, blue, green]
    for ind, c in enumerate(chunks):
        color = colors[ind % len(colors)]
        chunk_text = c.page_content.replace("\n", " ")
        print(f"{color} {chunk_text}")
    print(f"{green} -------- ")

    resu_map = {}
    for value in [9]:
        # while True:
        # docs = input(
        #     f"{green}Please enter the path of the document to summarize: (FAKE NOW)"
        # )

        # if docs == "exit" or docs == "quit" or docs == "q" or docs == "f":
        #     print("Exiting")
        #     sys.exit()
        # if docs == "":
        #     continue
        # if query == "flush":
        #     print("Erasing the history")
        #     chat_history = []
        #     continue
        # if query == "llm":
        #     llm = select_llm()
        #     continue
        # if query == "chain":
        #     qa = select_chain(llm, vectorstore)
        #     continue

        # chunks, _ = chunk_documents("docs_hobbit", 500, 0)
        # chunks = [
        #     c
        #     for c in chunks
        #     if (c.metadata["page"] >= 18 and c.metadata["page"] < 21)
        # ] # 33 # 85
        # print(chunks)
        # print("Text to parse")
        # colors = [red, yellow, blue, green]
        # for ind, c in enumerate(chunks):
        #     color = colors[ind%len(colors)]
        #     chunk_text = c.page_content.replace("\n", " ")
        #     print(f"{color} {chunk_text}")
        # print(f"{green} -------- ")

        llm_attributes: LLMAttributes = LLMType(value).get_attributes()
        llm = get_llm(attr=llm_attributes)

        chain = load_summarize_chain(
            llm,
            chain_type="refine",
            return_intermediate_steps=True,
            question_prompt=PROMPT,
            refine_prompt=refine_prompt,
        )

        answer_start = time.time()
        # chain = load_summarize_chain(llm, chain_type="refine")

        result = chain({"input_documents": chunks}, return_only_outputs=True)  # 196
        # result = chain.run(chunks[19:26])  # 46])
        answer_end = time.time()
        print()
        print(f"{yellow}Intermediate steps ")
        for inter in result["intermediate_steps"]:
            print(f"{yellow}    {inter}")
        print(f"{blue}Answer: " + result["output_text"])
        print(f"{red}Time: {round(answer_end - answer_start, 2)} sec")
        print(f"{green} -------- ")
        # input("Wait")
        resu_map[value] = (result["output_text"], round(answer_end - answer_start, 2))
        for k in sorted(resu_map.keys()):
            print(k)
            print(resu_map[k][0])
            print(resu_map[k][1])
            print("\n\n")


if __name__ == "__main__":
    typer.run(main)
