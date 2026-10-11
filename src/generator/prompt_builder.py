from typing import Dict, List, Any

def build_prompt(query: str, retrieved_chunks: List[Dict]) -> str:

    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        text = chunk['text']
        filepath = chunk.get('filepath', "Unknown")

        block = f"--- Source: {i+1}: {filepath} ---\n{text}"
        context_blocks.append(block)

    context_str = "\n\n".join(context_blocks)

    prompt = f"""You are an expert AI software engineer. 
Your task is to answer the user's question based ONLY on the provided source code and documentation.
If the answer is not in the context, say "I don't have enough context to answer that."

CONTEXT:
{context_str}

USER QUESTION: 
{query}

ANSWER:"""

    return prompt

