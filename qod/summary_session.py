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
        if self.chain_type == ChainType.STUFFED:
            return self._summarize_documents_stuffed()
        elif self.chain_type == ChainType.MAP_REDUCE:
            return self._summarize_documents_reduce()
        elif self.chain_type == ChainType.REFINE:
            return self._summarize_documents_refine()
        print("This chain is not supported yet")
        return ""

    def _summarize_documents_stuffed(self) -> str:
        """ """
        # Settinp up the chain
        prompt = PromptTemplate.from_template(
            'Summarize this content as a text: "{context}"\n\n' "Summary: "
        )
        # Setting up the chain
        chain = LLMChain(prompt=prompt, llm=self.llm)
        # Getting the context
        context = ""
        for document in self.documents:
            context += document.page_content

        # The context is rejected if the prompt exceed the maximum context
        # size of the LLM
        nb_prompt_token = self.llm.get_num_tokens(prompt.format(context=context))
        if nb_prompt_token > self.llm.n_ctx:
            print(
                "The document size exceeds the maximum number of tokens \
supported by the LLM. You must select another chain type that supports \
long documents (e.g., refine)"
            )
            return ""
        return chain.run(context=context)

    def _get_summaries_as_context(
        self, summaries: List[str], prompt: PromptTemplate
    ) -> List[str]:
        """ """
        summaries_as_context = []
        context = ""
        for summary in summaries:
            summary = summary.replace("\n", " ")
            new_context = f'CONTEXT: "{summary}"\n\n'
            temporary_context = context + new_context
            nb_token_with_additional_content = self.llm.get_num_tokens(
                prompt.format(summaries=temporary_context)
            )
            print("\n\n############")
            print(
                f"nb_token_with_additional_content: {nb_token_with_additional_content}"
            )
            print("############\n\n")
            # We add the context as a list of summaries
            if nb_token_with_additional_content > self.llm.n_ctx:
                summaries_as_context.append(context)
                context = ""
            context += new_context
        summaries_as_context.append(context)
        return summaries_as_context

    def _summarize_documents_reduce(self):
        """ """
        max_level = 3
        # Setting up the LLM chains
        reduce_prompt = PromptTemplate.from_template(
            "Write a concise summary of the following: \n\n"
            '"{context}"\n\n'
            "CONCISE SUMMARY: "
        )
        llm_reduce = LLMChain(prompt=reduce_prompt, llm=self.llm)
        # collapse_prompt = PromptTemplate.from_template(
        #     "Collapse the following context: \n\n"
        #     '"{context}"\n\n'
        #     "COLLAPSED CONTEXT: "
        # )
        # llm_collapse = LLMChain(prompt=collapse_prompt, llm=self.llm)
        synthesize_prompt = PromptTemplate.from_template(
            "Given the following pieces of content extracted from \
a long document, write a final summary for the long document:\n\n"
            "{summaries} \n\n"
            "FINAL SUMMARY:"
        )
        llm_synthesize = LLMChain(prompt=synthesize_prompt, llm=self.llm)

        summaries = []
        summaries_as_context = []
        chunks = [document.page_content for document in self.documents]
        level = 0  # TODO - DEFINE THE CONTEXT  OF MAX LEVEL
        while chunks and level < max_level:
            context = chunks.pop(0)
            my_prompt = (
                reduce_prompt.format(context=context)
                if level == 0
                else synthesize_prompt.format(summaries=context)
            )
            print(f"\n\nPrompt: {my_prompt}\n\n")
            summary = (
                llm_reduce.run(context) if level == 0 else llm_synthesize.run(context)
            )
            summaries.append(summary)
            # If all chunks have been processed, we need to that
            #  the summaries do not exceed the LLM context size
            if not chunks:
                print(f"### Level {level}")
                summaries_as_context = self._get_summaries_as_context(
                    summaries=summaries, prompt=synthesize_prompt
                )
                # The summaries are too long for the LLM
                # They must be collated and reprocessed
                if len(summaries_as_context) > 1:
                    print("ADDING A LEVEL OF RECURSION!!!")
                    level += 1
                    for sum_as_ctxt in summaries_as_context:
                        chunks.append(sum_as_ctxt)
                        summaries = []
        # We exceed the maximum level of recursion
        if level >= max_level:
            print(
                "The reduce chain exceeded its maximum level of recursion. \
We recommend using a refine chain instead."
            )
            return ""
        my_prompt = synthesize_prompt.format(summaries=summaries_as_context[0])
        print(f"\n\nPrompt: {my_prompt}\n\n")
        return llm_synthesize.run(summaries_as_context[0])

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
