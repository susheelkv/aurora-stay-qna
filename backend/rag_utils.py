
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    rules = """
    1. Answer ONLY using the Retrieved Policy Information above. Do not use outside knowledge,
       general hotel industry practice, or assumptions.
    2. If the retrieved information does not contain the answer, reply exactly:
       "I could not find this in the AuroraStay policy documents. Please check manually or escalate to the helpdesk."
       If it answers only part of the question, answer that part and clearly state which part is not covered.
    3. Answer the question directly in the first sentence, then add only the supporting details needed
       (e.g., deadlines, fees, conditions, exceptions). Do not include unrelated policies.
    4. Quote numbers exactly as written in the policy: times, dollar amounts, percentages, number of
       nights/hours, and deadlines. Never round, estimate, or convert them.
    5. If the policy differs by rate plan, room type, or property, state which one applies; if the query
       does not specify, list the relevant variations briefly.
    6. If the retrieved chunks conflict, say so and present both rather than choosing one.
    7. Keep the answer concise (under about 150 words), in plain language suitable for hotel staff
       to relay to a guest. Use short bullet points only when listing multiple items or conditions.
    """

    prompt = f"""
    You are an AI assistant answering questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    {rules}

    Answer:
    """
    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
