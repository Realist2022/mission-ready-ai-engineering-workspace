def fine_tuned_response(prompt):
    # Style-only: more formal wording, but no new facts are added.
    return f"[Formal Tone] As requested, here is a brief explanation: {prompt}."

def rag_response(prompt, docs):
    # Knowledge: same question, but now grounded in external context (docs).
    return (
        f"{prompt}\n\n"
        f"Context from knowledge base:\n"
        f"{docs[:200]}"
    )

user_question = "Explain refunds"

kb_snippet = (
    "Refund policy updated March 2024: Customers can request a refund within 30 days of purchase "
    "with proof of payment. Digital items may have different refund rules."
)

print("=== Style-only (fine-tuned) response ===")
print(fine_tuned_response(user_question))

print("\n=== Knowledge-grounded (RAG) response ===")
print(rag_response(user_question, kb_snippet))