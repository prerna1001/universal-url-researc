from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLLM as LLM
from langchain_core.outputs import LLMResult, Generation
from typing import Optional, List

import os
import requests

class WorkerAILLM(LLM):
    """Custom LangChain LLM wrapper for Cloudflare Worker AI."""

    # Declare endpoint as a Pydantic/LLM field instead of setting it in __init__
    endpoint: str

    @property
    def _llm_type(self) -> str:
        return "worker_ai"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Sends the prompt to the Worker AI endpoint and returns the response.
        """
        import logging
        logging.basicConfig(level=logging.DEBUG)

        headers = {"Content-Type": "application/json"}
        payload = {"prompt": prompt}
        response = requests.post(self.endpoint, json=payload, headers=headers)

        logging.debug(f"Request Payload: {payload}")
        logging.debug(f"Response: {response.text}")


        if response.status_code == 200:
            try:
                # Parse the nested response structure
                return response.json()[0]["response"]["response"]
            except (KeyError, IndexError) as e:
                raise ValueError(f"Unexpected response structure: {response.text}") from e
        else:
            raise ValueError(f"Worker AI API call failed with status {response.status_code}: {response.text}")

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResult:
        """Generate LLMResult for a batch of prompts using underlying _call.

        This satisfies the abstract interface of BaseLLM in langchain-core.
        """
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop)
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)

def create_rag_chain(retriever, model_name="llama-2-7b"):
    """
    Create a Retrieval-Augmented Generation (RAG) chain using LangChain.

    Args:
        retriever: A LangChain retriever for fetching relevant documents.
        model_name (str): Name of the Llama model to use for generation.

    Returns:
        RetrievalQA: A LangChain RetrievalQA chain.
    """
    # Define the prompt template (MVP: serious, context-grounded, no jokes)
    prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a concise, professional financial research assistant.\n"
        "Use ONLY the information in the CONTEXT block to answer the QUESTION.\n"
        "- Do NOT invent facts or speculate beyond the context.\n"
        "- Do NOT make jokes, puns, or casual commentary.\n"
        "- If the answer is not in the context, respond exactly with:\n"
        '  "The answer is not found in the provided sources."\n\n'
        "Format your answer in clear Markdown with this structure:\n"
        "1. **Short Answer** – 2–3 sentences directly answering the question.\n"
        "2. **Key Points** – 3–6 bullet points summarizing the main arguments.\n"
        "3. **Evidence from Sources** – bullets that briefly quote or paraphrase\n"
        "   the most relevant parts of the context.\n"
        "4. **Limitations** – 1–2 bullets if the context is incomplete or partial.\n\n"
        "CONTEXT:\n"
        "{context}\n\n"
        "QUESTION:\n"
        "{question}\n\n"
        "Now write the answer following the structure above:"
    ),
)


    # Initialize the Worker AI LLM
    # Read endpoint from environment for flexible deployment; fall back to default if unset.
    endpoint = os.getenv("WORKER_ENDPOINT")
    llm = WorkerAILLM(endpoint=endpoint)

    # Create the RetrievalQA chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True,
    )

    return rag_chain