from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLLM as LLM
from langchain_core.outputs import LLMResult, Generation
from typing import Optional, List, Sequence, Mapping

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
            raise ValueError(
                f"Worker AI API call failed with status {response.status_code}: {response.text}"
            )

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

def get_worker_llm():
    """Initialize the Worker AI LLM from the configured endpoint."""
    endpoint = os.getenv("WORKER_ENDPOINT")
    return WorkerAILLM(endpoint=endpoint)


def get_rag_prompt_template():
    """Return the grounded chat prompt template."""
    return PromptTemplate(
        input_variables=["context", "question", "chat_history"],
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
            "- If you use that fallback sentence, return exactly that sentence and nothing else.\n"
            "- Keep answers concise unless the user asks for depth.\n"
            "- Keep the final answer under 2000 characters.\n"
            "- Use short paragraphs with breathing room.\n"
            "- If you have 3 or more distinct points, use a compact bullet list.\n"
            "- Never return one long unbroken wall of text.\n"
            "- Do not add headings like Short Answer, Key Points, Evidence, or Limitations.\n\n"
            "- Never add notes, disclaimers, or phrases like 'Note:'.\n\n"
            "Use CHAT HISTORY only to understand follow-up references like 'it', 'that', or "
            "'the previous paper'. Do not treat chat history as factual source material.\n\n"
            "CHAT HISTORY:\n"
            "{chat_history}\n\n"
            "CONTEXT:\n"
            "{context}\n\n"
            "QUESTION:\n"
            "{question}\n\n"
            "Answer:"
        ),
    )
 

def format_chat_history(
    chat_history: Sequence[Mapping[str, str]] | None,
    max_turns: int = 8,
) -> str:
    """Format the recent chat history for the model prompt."""
    if not chat_history:
        return "No prior chat."

    recent_turns = list(chat_history)[-max_turns:]
    lines: list[str] = []

    for turn in recent_turns:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")

    return "\n".join(lines) if lines else "No prior chat."


def build_context_block(source_docs) -> str:
    """Convert retrieved documents into a compact context block."""
    chunks: list[str] = []

    for index, doc in enumerate(source_docs, start=1):
        content = getattr(doc, "page_content", "").strip()
        if not content:
            continue

        metadata = getattr(doc, "metadata", {}) or {}
        url = metadata.get("url", "Unknown source")
        chunks.append(f"[Source {index}] URL: {url}\n{content}")

    return "\n\n".join(chunks).strip()


def generate_grounded_answer(
    llm: WorkerAILLM,
    prompt_template: PromptTemplate,
    question: str,
    source_docs,
    chat_history: Sequence[Mapping[str, str]] | None = None,
) -> str:
    """Generate a grounded answer from retrieved docs and recent chat turns."""
    context = build_context_block(source_docs)
    prompt = prompt_template.format(
        context=context,
        question=question,
        chat_history=format_chat_history(chat_history),
    )
    return llm.invoke(prompt)


def create_rag_chain(retriever, model_name="llama-2-7b"):
    """Backward-compatible helper returning the prompt and LLM pieces."""
    return {
        "retriever": retriever,
        "llm": get_worker_llm(),
        "prompt_template": get_rag_prompt_template(),
    }
