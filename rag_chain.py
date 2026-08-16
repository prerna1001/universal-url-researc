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
    # Define a simpler prompt so answers feel like a natural chat reply.
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a grounded research assistant inside a chat app.\n"
            "Answer the user's QUESTION using ONLY the CONTEXT.\n"
            "Rules:\n"
            "- Be direct, clear, and natural.\n"
            "- Do not repeat instructions or mention formatting rules.\n"
            "- Do not invent facts beyond the context.\n"
            "- If the context is incomplete, say so briefly.\n"
            "- If the answer is not supported by the context, say: "
            "\"I couldn't find that in the indexed sources.\"\n"
            "- Prefer a short paragraph or a few bullets only when bullets genuinely help.\n"
            "- Do not add headings like Short Answer, Key Points, Evidence, or Limitations.\n\n"
            "CONTEXT:\n"
            "{context}\n\n"
            "QUESTION:\n"
            "{question}\n\n"
            "Answer:"
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
