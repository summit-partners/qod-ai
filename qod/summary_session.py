import re
from typing import List, Optional

from langchain.llms import LlamaCpp
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain import LLMChain

from qod.chain_data_types import SummaryChainType
from qod.langchain_wrappers import chunk_documents
from qod.display_msg import (
    display_error,
    display_cli_notification,
    display_llm_notification,
    display_chain_temp,
    red,
    yellow,
    blue,
    purple,
    white,
    cyan,
)


class SummarySession:
    """Object that captures a summary session"""

    ssid: str
    llm: LlamaCpp
    document_path: str
    summary_chain_type: SummaryChainType
    first_page: Optional[int]
    last_page: Optional[int]
    documents: List[Document] = []

    def __init__(
        self,
        ssid: str,
        llm: LlamaCpp,
        document_path: str,
        summary_chain_type: SummaryChainType,
        first_page: Optional[int],
        last_page: Optional[int],
    ):
        """Constructor for a SummarySession
        :param ssid: Identifier for a summary session
        :param llm: Large language model used by the session
        :param document_path: Path of the document to summarize
        :param sumary_chain_type: Type of chain used to produce the summary
        :param first_page: First page of the document to summarize
        (for pdf only)
        :param last_page: Last page of the document to summarize
        (for pdf only)
        """
        self.ssid = ssid
        self.llm = llm
        self.document_path = document_path
        self.summary_chain_type = summary_chain_type
        self.first_page = first_page
        self.last_page = last_page

    def get_documents(self) -> List[Document]:
        """Import the document to summarize and decompose it into
        chunks based on the chain selected
        """

        # Import the document based on the chain expectations
        # (one chunk for stuffed, multiple chunks for other chains)
        documents: List[Document] = []
        if self.summary_chain_type == SummaryChainType.STUFFED:
            documents, _ = chunk_documents(
                path=self.document_path,
                chunk_size=float("Inf"),
                chunk_overlap=0,
                first_page=self.first_page,
                last_page=self.last_page,
            )
        elif self.summary_chain_type == SummaryChainType.MAP_REDUCE:
            documents, _ = chunk_documents(
                path=self.document_path,
                chunk_size=3000,
                chunk_overlap=500,
                first_page=self.first_page,
                last_page=self.last_page,
            )
        elif self.summary_chain_type == SummaryChainType.REFINE:
            documents, _ = chunk_documents(
                path=self.document_path,
                chunk_size=1500,
                chunk_overlap=250,
                first_page=self.first_page,
                last_page=self.last_page,
            )
        else:
            display_error(
                "The chain selected is not supported. Please select \
another type of chain."
            )
        self.documents = documents

        return documents

    def _find_overlap(self, str_a: str, str_b: str) -> str:
        """ """
        min_length = min(len(str_a), len(str_b))
        for i in range(1, min_length + 1):
            if str_a[-i:] == str_b[:i]:
                return str_a[-i:]
        return ""

    def display_chunked_documents(self):
        """Diplay the document segmented into the chunks
        used by the LLM chain to assemble a summary.
        """
        colors = [red, yellow, blue, purple, cyan]
        overlap_color = white
        nb_docs = len(self.documents)
        chunked_text = ""
        previous_overlap_length = 0
        for ind, doc in enumerate(self.documents):
            text = doc.page_content
            overlap = ""
            if ind < nb_docs - 1:
                next_text = self.documents[ind + 1].page_content
                overlap = self._find_overlap(text, next_text)

            color = colors[ind % len(colors)]
            chunked_text += (
                f"{color} {text[previous_overlap_length:(len(text)-len(overlap))]}"
            )
            chunked_text += f"{overlap_color} {overlap}"
            previous_overlap_length = len(overlap)
            # Replacing \n by a single space
        chunked_text = re.sub("\n{1,}", " ", chunked_text)
        chunked_text = re.sub(" {1,}", " ", chunked_text)
        print(f"{chunked_text}\n")

    def summarize_documents(self) -> str:
        """Use the LLM to compute and return a summary of the document
        :return A summary of the document #TODO - Add intermediary results
        """
        # Call the summarize function associated with the chain type
        # of the session
        if self.summary_chain_type == SummaryChainType.STUFFED:
            display_cli_notification("Summarizing the document using a stuffed chain")
            return self._summarize_documents_stuffed()
        elif self.summary_chain_type == SummaryChainType.MAP_REDUCE:
            display_cli_notification("Summarizing the document using a reduce chain")
            return self._summarize_documents_reduce()
        elif self.summary_chain_type == SummaryChainType.REFINE:
            display_cli_notification("Summarizing the document using a refine chain")
            return self._summarize_documents_refine()
        display_error(
            "The chain selected is not supported. Please select \
another type of chain."
        )
        return ""

    def _summarize_documents_stuffed(self) -> str:
        """Summarize the document using a stuffed chain.
        :return A summary of the document #TODO - Add intermediary results
        """
        documents: List[Document] = self.get_documents()
        # Setting up the chain
        prompt = PromptTemplate.from_template(
            'Summarize this content as a text: "{context}"\n\n' "Summary: "
        )
        chain = LLMChain(prompt=prompt, llm=self.llm)

        # Getting the context as a single blurb
        context = ""
        for document in documents:
            context += document.page_content

        # The context is rejected if the prompt exceed the maximum context
        # size of the LLM
        nb_prompt_token = self.llm.get_num_tokens(prompt.format(context=context))
        if nb_prompt_token > self.llm.n_ctx:
            display_error(
                "The document size exceeds the maximum number of tokens \
supported by the LLM. You must select another chain type that supports \
long documents (e.g., refine)"
            )
            return ""

        display_chain_temp(f"Prompt: \n{prompt.format(context=context)}")
        display_chain_temp(f"Tokens: {nb_prompt_token}")
        display_llm_notification(msg="", reset=False)
        return chain.run(context=context)

    def _get_summaries_as_context(
        self,
        summaries: List[str],
        prompt: PromptTemplate,
        pass_compression_factor: int,
        display_temp: bool,
    ) -> List[str]:
        """Convert a list of summaries into a list of context to be used in a prompt.
        The size of the contexts in the produced list is determined according to the
        maximum context size of the LLM of the session.
        :param summaries: List of summaries (e.g., coming from a reduce chain)
        :param prompt: Prompt template in which the calculated context will be used
        :return A list of string that each capture a sequence of contexts
        that can be inserted
        in a prompt for a synthesis chain.
        """
        summaries_as_context = ""
        segmented_summary = ""
        segmented_summaries_as_context = []
        for index, summary in enumerate(summaries):
            summary = summary.replace("\n", " ")
            new_context = f'CONTEXT: "{summary}"\n\n'
            summaries_as_context += new_context
            segmented_summary += new_context
            if (index + 1) % pass_compression_factor == 0:
                segmented_summaries_as_context.append(segmented_summary)
                segmented_summary = ""
        if segmented_summary != "":
            segmented_summaries_as_context.append(segmented_summary)
        # Validating if the entire summary context exceed the
        #  LLM context size
        nb_token_with_additional_content = self.llm.get_num_tokens(
            prompt.format(summaries=summaries_as_context)
        )
        max_token = self.llm.n_ctx * 0.75
        if display_temp:
            my_prompt = prompt.format(summaries=summaries_as_context)
            nb_tokens = self.llm.get_num_tokens(my_prompt)
            display_chain_temp(f"Intermediary Prompt:\n{my_prompt}")
            display_chain_temp(f"Tokens: {nb_tokens}/{max_token}")

        if nb_token_with_additional_content > self.llm.n_ctx * 0.75:
            return segmented_summaries_as_context
        else:
            return [summaries_as_context]

    def _summarize_documents_reduce(
        self, pass_compression_factor: int = 5, max_pass: int = 5, display_temp=False
    ):
        """Summarize the document using a reduce chain.
        :return A summary of the document #TODO - Add intermediary results"""
        documents: List[Document] = self.get_documents()
        self.display_chunked_documents()

        # The reduce chain is used to extract a summary for each
        # chunk of the document to summarize
        reduce_prompt = PromptTemplate.from_template(
            "Write a concise summary of the following: \n\n"
            '"{context}"\n\n'
            "CONCISE SUMMARY: "
        )
        llm_reduce = LLMChain(prompt=reduce_prompt, llm=self.llm)
        # The colapse chain is used to collapse summaries assembled
        # by the reduce chain or by another collapse chain
        collapse_prompt = PromptTemplate.from_template(
            "Write a summary that combines the following sequence \
of pieces of contexts \
extracted from the same document: :\n\n"
            "{summaries} \n\n"
            "SUMMARY:"
        )
        llm_collapse = LLMChain(prompt=collapse_prompt, llm=self.llm)
        # The synthesize chain is used to assemble a summary from all summaries
        # calculated during the last pass
        synthesize_prompt = PromptTemplate.from_template(
            "Assemble a final summary that combines all the \
information contained in the following pieces of contexts: \n\n"
            "{summaries} \n\n"
            "FINAL SUMMARY:"
        )
        llm_synthesize = LLMChain(prompt=synthesize_prompt, llm=self.llm)

        summaries = []
        summaries_as_context = []
        chunks = [document.page_content for document in documents]
        pass_index = 0
        nb_pass_chunks = len(chunks)
        chunk_index = 0
        while chunks and pass_index < max_pass:
            context = chunks.pop(0)
            display_cli_notification(
                f"Summarizing chunk {chunk_index+1}/{nb_pass_chunks} \
from pass {pass_index+1}"
            )
            # Display prompt
            if display_temp:
                my_prompt = (
                    reduce_prompt.format(context=context)
                    if pass_index == 0
                    else collapse_prompt.format(summaries=context)
                )
                nb_tokens = self.llm.get_num_tokens(my_prompt)
                display_chain_temp(f"Intermediary prompt:\n{my_prompt}")
                display_chain_temp(f"# tokens: {nb_tokens}")

            # Assemble summary for the chunk
            display_llm_notification(msg="", reset=False)
            summary = (
                llm_reduce.run(context)
                if pass_index == 0
                else llm_collapse.run(context)
            )
            print()

            # Display summary for the chunk
            if display_temp:
                display_chain_temp(f"Intermediary summary: {summary}")

            summaries.append(summary)
            chunk_index += 1

            # If all chunks have been processed, we need to that
            #  the summaries do not exceed the LLM context size
            if not chunks:
                summaries_as_context = self._get_summaries_as_context(
                    summaries=summaries,
                    prompt=collapse_prompt,
                    pass_compression_factor=pass_compression_factor,
                    display_temp=display_temp,
                )
                # The summaries are too long for the LLM
                # They must be collated and reprocessed
                if len(summaries_as_context) > 1:
                    display_cli_notification(
                        "Generating the content for an additional pass"
                    )
                    pass_index += 1
                    for sum_as_ctxt in summaries_as_context:
                        chunks.append(sum_as_ctxt)
                        summaries = []
                    nb_pass_chunks = len(chunks)
        # We exceed the maximum level of recursion
        if pass_index >= max_pass:
            display_error(
                "The reduce chain exceeded its maximum level of recursion. \
We recommend using a refine chain instead."
            )
            return ""
        display_cli_notification(
            "Synthesizing the summaries assembled in the last pass"
        )

        # Display final prompt
        if display_temp:
            my_prompt = synthesize_prompt.format(summaries=summaries_as_context[0])
            nb_tokens = self.llm.get_num_tokens(my_prompt)
            display_chain_temp(f"Final prompt:\n{my_prompt}")
            display_chain_temp(f"# tokens: {nb_tokens}")

        display_llm_notification(msg="", reset=False)
        return llm_synthesize.run(summaries_as_context[0])

    def _summarize_documents_refine(self) -> str:
        """Summarize the document using a refine chain.
        :return A summary of the document #TODO - Add intermediary results"""
        documents: List[Document] = self.get_documents()
        self.display_chunked_documents()

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
        for index, document in enumerate(documents):
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
                    display_cli_notification(
                        f"WARNING  - The prompt exceed the maximum \
context size of the LLM ({nb_prompt_token}/{self.llm.n_ctx}), \
the new context will be collapsed."
                    )
                    display_llm_notification(msg="", reset=False)
                    context = llm_collapse.run(context=context)
                my_prompt = refine_start_prompt.format(context=context)
                display_chain_temp(f"Intermediary prompt:\n{my_prompt}")
                display_llm_notification(msg="", reset=False)
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
                display_cli_notification(
                    f"The prompt exceed the maximum \
context size of the LLM ({nb_prompt_token}/{self.llm.n_ctx}), \
the new context will be collapsed."
                )
                display_llm_notification(msg="", reset=False)
                context = llm_collapse.run(context=context)

            my_prompt = refine_prompt.format(prev_summary=summary, context=context)
            display_chain_temp(f"Intermediary prompt:\n{my_prompt}")

            display_llm_notification(msg="", reset=False)
            summary = llm_refine.run({"prev_summary": summary, "context": context})
        return summary
