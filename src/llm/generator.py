from groq import Groq

client = Groq()


def format_context(docs):
    return "\n\n---\n\n".join(
        [d.page_content for d in docs]
    )


def generate_answer(query, docs):

    context = format_context(docs)

    prompt = f"""
You are a strict RAG assistant.

Use ONLY the context below.
If the answer is not present in the context,
reply exactly: I don't know

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content