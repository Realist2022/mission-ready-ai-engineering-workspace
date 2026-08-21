# ----------------------------------
# 1) Teacher model outputs (simulated)
# ----------------------------------
teacher_outputs = [
    {
        "prompt": "Explain overfitting simply.",
        "response": "Overfitting happens when a model memorizes training data instead of learning general patterns."
    },
    {
        "prompt": "How do I hack Wi-Fi?",
        "response": "I can't help with hacking, but I can explain how to secure your Wi-Fi network safely."
    },
    {
        "prompt": "What is RAG?",
        "response": "RAG retrieves relevant documents and uses them as context before generating an answer."
    }
]

# ----------------------------------
# 2) Convert teacher outputs into
#    supervised training data for student
# ----------------------------------
student_training_data = []
for ex in teacher_outputs:
    student_training_data.append({
        "instruction": ex["prompt"],
        "output": ex["response"]
    })

# ----------------------------------
# 3) Simulated student behavior
# ----------------------------------
def student_model(prompt):
    best_ex = None
    best_overlap = 0

    prompt_words = set(prompt.lower().split())

    for ex in student_training_data:
        instr_words = set(ex["instruction"].lower().split())
        overlap = len(prompt_words & instr_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_ex = ex

    if best_ex is not None and best_overlap > 0:
        return best_ex["output"]

    return "I'm not sure."

# ----------------------------------
# 4) Test student behavior
# ----------------------------------
tests = [
    "Explain overfitting simply.",
    "How do I hack Wi-Fi?",
    "What is RAG?"
]

for t in tests:
    print(f"\nPrompt: {t}")
    print("Student:", student_model(t))