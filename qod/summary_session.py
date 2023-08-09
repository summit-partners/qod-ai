import re
from typing import List, Optional

from langchain.llms import LlamaCpp
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain import LLMChain

from qod.chain_data_types import SummaryChainType
from qod.langchain_wrappers import chunk_documents


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
            print(
                f"{red}The chain selected is not supported. Please select \
another type of chain.{reset}"
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
        print(f"{chunked_text}{reset}\n")

    def summarize_documents(self) -> str:
        """Use the LLM to compute and return a summary of the document
        :return A summary of the document #TODO - Add intermediary results
        """
        # Call the summarize function associated with the chain type
        # of the session
        if self.summary_chain_type == SummaryChainType.STUFFED:
            print(f"{white}Summarizing the document using a stuffed chain{reset}")
            return self._summarize_documents_stuffed()
        elif self.summary_chain_type == SummaryChainType.MAP_REDUCE:
            print(f"{white}Summarizing the document using a reduce chain{reset}")
            return self._summarize_documents_reduce()
        elif self.summary_chain_type == SummaryChainType.REFINE:
            print(f"{white}Summarizing the document using a refine chain{reset}")
            return self._summarize_documents_refine()
        print(
            f"{red}The chain selected is not supported. Please select \
another type of chain.{reset}"
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
            print(
                f"{red}The document size exceeds the maximum number of tokens \
supported by the LLM. You must select another chain type that supports \
long documents (e.g., refine)"
            )
            return ""

        print(f"{yellow}\n\nPrompt: \n{prompt.format(context=context)}\n{reset}")
        print(f"{yellow}Tokens: {nb_prompt_token}\n\n{reset}")
        print(f"{purple}")
        return chain.run(context=context)

    def _get_summaries_as_context(
        self, summaries: List[str], prompt: PromptTemplate
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
        compression_ratio = (
            5  # int(len(summaries)/10)+1  # TODO - Add that as a parameter
        )
        summaries_as_context = ""
        segmented_summary = ""
        segmented_summaries_as_context = []
        for index, summary in enumerate(summaries):
            summary = summary.replace("\n", " ")
            new_context = f'CONTEXT: "{summary}"\n\n'
            summaries_as_context += new_context
            segmented_summary += new_context
            if (index + 1) % compression_ratio == 0:
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
        print(
            f"{yellow}\n\nIntermediary Prompt: \n \
{prompt.format(summaries=summaries_as_context)}\n{reset}"
        )
        print(
            f"{yellow}Tokens: \
{nb_token_with_additional_content}/{max_token}\n{reset}"
        )

        if nb_token_with_additional_content > self.llm.n_ctx * 0.75:
            return segmented_summaries_as_context
        else:
            return [summaries_as_context]

    def _summarize_documents_reduce(self):
        """Summarize the document using a reduce chain.
        :return A summary of the document #TODO - Add intermediary results"""
        # The max recursion level indicates the maximjm passes we want to
        #  do over the document or its summaries
        max_pass = 5  # TODO - add as an argument to the method

        documents: List[Document] = self.get_documents()

        # The reduce chain is used to extract a summary for each
        # chunk of the document to summarize
        reduce_prompt = PromptTemplate.from_template(
            "Write a concise summary of the following: \n\n"
            '"{context}"\n\n'
            "CONCISE SUMMARY: "
        )
        llm_reduce = LLMChain(prompt=reduce_prompt, llm=self.llm)
        # The synthesize chain is used to collapse summaries assembled
        # by the reduce chain or by another synthesize chain
        # synthesize_prompt = PromptTemplate.from_template(
        #     "Given the following pieces of content extracted
        # from a long document, write a final summary for the long
        # document:\n\n"
        #     "{summaries} \n\n"
        #     "FINAL SUMMARY:"
        # )
        synthesize_prompt = PromptTemplate.from_template(
            "Write a summary that combines the following sequence \
of pieces of contexts \
extracted from the same document: :\n\n"
            "{summaries} \n\n"
            "SUMMARY:"
        )
        llm_synthesize = LLMChain(prompt=synthesize_prompt, llm=self.llm)

        final_prompt = PromptTemplate.from_template(
            "Assemble a final summary that combines all the \
information contained in the following pieces of contexts: \n\n"
            "{summaries} \n\n"
            "FINAL SUMMARY:"
        )
        llm_final = LLMChain(prompt=final_prompt, llm=self.llm)

        summaries = []
        summaries_as_context = []
        chunks = [document.page_content for document in documents]
        nb_pass = 0
        while chunks and nb_pass < max_pass:
            context = chunks.pop(0)
            my_prompt = (
                reduce_prompt.format(context=context)
                if nb_pass == 0
                else synthesize_prompt.format(summaries=context)
            )
            print(f"\n\n{yellow}Intermediary prompt:\n{my_prompt}\n{reset}")
            print(f"{purple}")
            summary = (
                llm_reduce.run(context) if nb_pass == 0 else llm_synthesize.run(context)
            )
            print("f{reset}")
            print(f"\n{yellow}Intermediary summary: {summary}\n{reset}")
            summaries.append(summary)
            # If all chunks have been processed, we need to that
            #  the summaries do not exceed the LLM context size
            if not chunks:
                print(f"{white}Pass #{nb_pass}{reset}\n")
                summaries_as_context = self._get_summaries_as_context(
                    summaries=summaries, prompt=synthesize_prompt
                )
                # The summaries are too long for the LLM
                # They must be collated and reprocessed
                if len(summaries_as_context) > 1:
                    print("{white}Generating the content for an additional pass{reset}")
                    nb_pass += 1
                    for sum_as_ctxt in summaries_as_context:
                        chunks.append(sum_as_ctxt)
                        summaries = []
        # We exceed the maximum level of recursion
        if nb_pass >= max_pass:
            print(
                f"{red}The reduce chain exceeded its maximum level of recursion. \
We recommend using a refine chain instead.{reset}"
            )
            return ""
        my_prompt = final_prompt.format(summaries=summaries_as_context[0])
        print(f"{yellow}\n\nFinal prompt:\n{my_prompt}\n{reset}")
        print(f"{purple}")
        return llm_final.run(summaries_as_context[0])

    def _summarize_documents_refine(self) -> str:
        """Summarize the document using a refine chain.
        :return A summary of the document #TODO - Add intermediary results"""
        documents: List[Document] = self.get_documents()
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
                    print(
                        f"{white}WARNING  - The prompt exceed the maximum \
context size of the LLM ({nb_prompt_token}/{self.llm.n_ctx}), \
the new context will be collapsed.{reset}"
                    )
                    print(f"{purple}")
                    context = llm_collapse.run(context=context)
                    print("reset")
                my_prompt = refine_start_prompt.format(context=context)
                print(f"{yellow}\n\nIntermediary prompt:\n{my_prompt}\n{reset}")
                print(f"{purple}")
                summary = llm_start.run(context)
                print(f"{reset}")
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
                    f"{white}WARNING  - The prompt exceed the maximum \
context size of the LLM ({nb_prompt_token}/{self.llm.n_ctx}), \
the new context will be collapsed.{reset}"
                )
                print(f"{purple}")
                context = llm_collapse.run(context=context)
                print(f"{reset}")

            my_prompt = refine_prompt.format(prev_summary=summary, context=context)
            print(f"{yellow}\n\nIntermediary prompt:\n{my_prompt}\n{reset}")

            print(f"{purple}")
            summary = llm_refine.run({"prev_summary": summary, "context": context})
            print(f"{reset}")
        return summary
