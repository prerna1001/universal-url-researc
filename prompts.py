from langchain.prompts import PromptTemplate

# Prompt template for RAG-based question answering
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. Use the following context to answer the question.\n"
        "If the answer is not in the context, respond with: \"The answer is not found in the provided sources.\"\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)

# Additional prompt templates can be added here as needed.