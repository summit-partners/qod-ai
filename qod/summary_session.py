from typing import List

from langchain.llms import LlamaCpp
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain import LLMChain

from qod.chain_data_types import ChainType


class SummarySession:
    """Object that captures a summary session"""

    ssid: str
    llm: LlamaCpp
    document_path: str
    documents: List[Document]
    chain_type: ChainType

    def __init__(
        self,
        ssid: str,
        llm,
        document_path: str,
        documents: List[Document],
        chain_type: ChainType,
    ):
        self.ssid = ssid
        self.llm = llm
        self.document_path = document_path
        self.documents = documents
        self.chain_type = chain_type

    def summarize_documents(self) -> str:
        """ """
        if self.chain_type == ChainType.REFINE:
            return self._summarize_documents_refine()
        print("This chain is not supported yet")
        return ""

    def _summarize_documents_refine(self) -> str:
        """ """
        # Setting up the LLM chains
        refine_start_prompt = PromptTemplate.from_template(
            'Summarize this content: "{context}"\n\n' "Summary: "
        )
        llm_start = LLMChain(prompt=refine_start_prompt, llm=self.llm)
        refine_prompt = PromptTemplate.from_template(
            'Here\'s your first summary: "{prev_summary}".\n'
            'Now add to it based on the following context: "{context}"\n'
            "If the additional context is not useful, return the existing \
summary unmodified, else summarize the text obtained by adding the addtional \
context to the first summary.\n\n"
            "Updated summary: "
        )
        llm_refine = LLMChain(prompt=refine_prompt, llm=self.llm)
        collapse_prompt = PromptTemplate.from_template(
            'Collapse this context: "{context}"\n\n' "Updated context: "
        )
        llm_collapse = LLMChain(prompt=collapse_prompt, llm=self.llm)

        # Sequentially summarize the documents
        summary = ""
        summary = ""
        for index, document in enumerate(self.documents):
            # We extract the context from the document
            context: str = document.page_content.replace("\n", " ")

            # We apply the starting chain to the 1st document
            if index == 0:
                nb_prompt_token = self.llm.get_num_tokens(
                    refine_start_prompt.format(context=context)
                )
                # The context is collapsed if the prompt exceed the maximum
                # context size of the LLM
                # TODO - THIS SHOULD BE A WHILE LOOP
                if nb_prompt_token > self.llm.n_ctx:
                    print(
                        f"WARNING  - The prompt exceed the maximum \
context size of the LLM ({nb_prompt_token}/{self.llm.n_ctx}), \
the new context will be collapsed."
                    )
                    context = llm_collapse.run(context=context)
                my_prompt = refine_start_prompt.format(context=context)
                print(f"\n\nStart Prompt: {my_prompt}\n\n")
                summary = llm_start.run(context)
                continue

            # We apply the refine chain to sequentially enhance the summary
            nb_prompt_token = self.llm.get_num_tokens(
                refine_prompt.format(prev_summary=summary, context=context)
            )
            # The context is collapsed if the prompt exceed the maximum
            # context size of the LLM
            # TODO - THIS SHOULD BE A WHILE LOOP
            if nb_prompt_token > self.llm.n_ctx:
                print(
                    f"WARNING  - The prompt exceed the maximum \
context size of the LLM ({nb_prompt_token}/{self.llm.n_ctx}), \
the new context will be collapsed."
                )
                context = llm_collapse.run(context=context)

            my_prompt = refine_prompt.format(prev_summary=summary, context=context)
            print(f"\n\nRefine Prompt: {my_prompt}\n\n")

            summary = llm_refine.run({"prev_summary": summary, "context": context})
        return summary
