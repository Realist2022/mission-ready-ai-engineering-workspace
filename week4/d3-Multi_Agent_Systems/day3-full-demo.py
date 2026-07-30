"""
Multi-Agent Demo: Planner â†’ Executor â†’ Verifier

What it shows:
- Multiple agents (Planner, Executor, Verifier) with different roles
- Message passing between agents
- A simple orchestration loop over steps
- A real task: summarise a text + extract 3 key points

Requirements:
    pip install openai python-dotenv

Set your API key in the .env file:
    OPENAI_API_KEY=your-api-key-here

Run:
    python multi_agent_demo.py
"""



import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -------------------------
# OpenAI client setup
# -------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------------------------
# System prompts (roles)
# -------------------------

PLANNER_SYSTEM_PROMPT = """
You are the PLANNER agent.

Your job is to break the user's request into 2â€“4 clear steps.
Each step must be short and action-oriented.

Rules:
- Use a numbered list.
- Each step should be one sentence.
- Focus on text analysis tasks: summarising, extracting key points, checking tone, etc.
"""

EXECUTOR_SYSTEM_PROMPT = """
You are the EXECUTOR agent.

Your job is to perform ONE specific step on the given text.
You do not plan. You just DO the step.

Rules:
- Only focus on the current step.
- Only use the provided text as context.
- Answer clearly and concisely.
"""

VERIFIER_SYSTEM_PROMPT = """
You are the VERIFIER agent.

Your job is to check whether the EXECUTOR's answer correctly followed the step.

Respond in one of two formats ONLY:
1) APPROVE: <short reason>
2) REVISE: <short feedback on what to fix>

Be strict but fair. Approve only if the answer clearly follows the instruction.
"""


# -------------------------
# Helper: call LLM
# -------------------------

def call_llm(system_prompt: str, user_content: str, temperature: float = 0.0) -> str:
    """
    Simple wrapper around OpenAI Chat Completions.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


# -------------------------
# Planner agent
# -------------------------

def planner_agent(task: str, text: str) -> List[str]:
    """
    Asks the PLANNER to break the task into steps.
    Returns a list of step strings.
    """
    prompt = f"""
User task:
{task}

Text to work with:
{text[:500]}  # (Only showing first 500 chars for context preview.)

Break this task into 2â€“4 steps.
"""
    plan_output = call_llm(PLANNER_SYSTEM_PROMPT, prompt)
    print("\n[PLANNER OUTPUT]")
    print(plan_output)
    print("-" * 60)

    # Parse numbered steps: lines starting with "1.", "2.", etc.
    steps: List[str] = []
    for line in plan_output.splitlines():
        line = line.strip()
        if line and line[0].isdigit() and "." in line:
            # Remove "1." prefix
            step_text = line.split(".", 1)[1].strip()
            if step_text:
                steps.append(step_text)

    if not steps:
        # Fallback: if parsing failed, just use the whole output as one step
        steps = [plan_output.strip()]

    return steps


# -------------------------
# Executor agent
# -------------------------

def executor_agent(step: str, text: str) -> str:
    """
    Asks the EXECUTOR to perform one step on the given text.
    """
    prompt = f"""
Current step:
{step}

Text to work with:
{text}

Perform ONLY this step and return the result.
"""
    result = call_llm(EXECUTOR_SYSTEM_PROMPT, prompt, temperature=0.2)
    print("\n[EXECUTOR OUTPUT]")
    print(result)
    print("-" * 60)
    return result


# -------------------------
# Verifier agent
# -------------------------

def verifier_agent(step: str, executor_output: str) -> str:
    """
    Asks the VERIFIER to approve or request revision.
    """
    prompt = f"""
Instruction step:
{step}

Executor's answer:
{executor_output}

Did the executor correctly follow the step?
Remember: respond with either
APPROVE: <reason>
or
REVISE: <what was wrong and how to fix it>
"""
    verdict = call_llm(VERIFIER_SYSTEM_PROMPT, prompt)
    print("\n[VERIFIER OUTPUT]")
    print(verdict)
    print("-" * 60)
    return verdict


def is_approved(verdict: str) -> bool:
    """
    Simple check: does verifier start with 'APPROVE:'?
    """
    return verdict.strip().upper().startswith("APPROVE:")


# -------------------------
# Orchestrator
# -------------------------

def run_multi_agent_pipeline(task: str, text: str) -> Dict:
    """
    Orchestrates Planner â†’ Executor â†’ Verifier for each step.
    Returns a dict with steps, intermediate results, and a combined answer.
    """
    print("=" * 60)
    print("USER TASK:", task)
    print("=" * 60)

    # 1) PLAN
    steps = planner_agent(task, text)
    print("\n[PARSED STEPS]")
    for i, s in enumerate(steps, start=1):
        print(f"Step {i}: {s}")
    print("-" * 60)

    results = []
    for i, step in enumerate(steps, start=1):
        print(f"\n=== PROCESSING STEP {i} ===")
        # 2) EXECUTE
        exec_output = executor_agent(step, text)

        # 3) VERIFY (up to 1 retry for simplicity)
        verdict = verifier_agent(step, exec_output)
        if not is_approved(verdict):
            print("[ORCHESTRATOR] Verifier requested a revision. Asking executor again...")
            # Could add feedback to executor prompt here; keeping simple.
            exec_output = executor_agent(step, text)
            verdict = verifier_agent(step, exec_output)

        if is_approved(verdict):
            print(f"[ORCHESTRATOR] Step {i} approved.")
            results.append(exec_output)
        else:
            print(f"[ORCHESTRATOR] Step {i} not approved after retry. Skipping.")
            results.append(f"[Step {i} failed verification]")

    # Combine results into final answer (simple concatenation)
    final_answer = "\n\n".join(results)

    return {
        "steps": steps,
        "results": results,
        "final_answer": final_answer,
    }


# -------------------------
# Main demo
# -------------------------

if __name__ == "__main__":
    demo_text = """
Large language models (LLMs) are powerful tools for working with text.
They can summarise documents, answer questions, generate code, and help with research.
However, they can also make mistakes or hallucinate information that is not in the source material.
One way to reduce these errors is to use multiple agents with different roles, such as planner, executor, and verifier.
By splitting responsibilities, we can often catch mistakes and improve reliability.
    """.strip()

    user_task = "Summarise this text in 3â€“4 sentences and then list 3 key bullet points."

    result = run_multi_agent_pipeline(user_task, demo_text)

    print("\n" + "=" * 60)
    print("FINAL COMBINED ANSWER:\n")
    print(result["final_answer"])
    print("=" * 60)