import os
from typing import List, Dict, Tuple
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -------------------------
# System Prompts
# -------------------------
PLANNER_SYSTEM_PROMPT = """
You are the PLANNER agent.

Your job is to break the user's request into 2–4 clear steps.
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
# Agent Classes
# -------------------------
class BaseAgent:
    """Base class for all agents handling the core LLM communication."""
    def __init__(self, client: OpenAI, system_prompt: str, temperature: float = 0.0):
        self.client = client
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.model = "gpt-4o-mini"

    def _call_llm(self, user_content: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
        )
        return resp.choices[0].message.content.strip()


class PlannerAgent(BaseAgent):
    def __init__(self, client: OpenAI):
        super().__init__(client, PLANNER_SYSTEM_PROMPT, temperature=0.0)

    def plan(self, task: str, text: str) -> List[str]:
        prompt = f"User task:\n{task}\n\nText to work with:\n{text[:500]}\n\nBreak this task into 2–4 steps."
        output = self._call_llm(prompt)
        
        print("\n[PLANNER OUTPUT]\n", output)
        print("-" * 60)

        steps = [
            line.split(".", 1)[1].strip() 
            for line in output.splitlines() 
            if line.strip() and line.strip()[0].isdigit() and "." in line
        ]
        return steps if steps else [output.strip()]


class ExecutorAgent(BaseAgent):
    def __init__(self, client: OpenAI):
        super().__init__(client, EXECUTOR_SYSTEM_PROMPT, temperature=0.2)

    def execute(self, step: str, text: str) -> str:
        prompt = f"Current step:\n{step}\n\nText to work with:\n{text}\n\nPerform ONLY this step and return the result."
        result = self._call_llm(prompt)
        
        print("\n[EXECUTOR OUTPUT]\n", result)
        print("-" * 60)
        return result


class VerifierAgent(BaseAgent):
    def __init__(self, client: OpenAI):
        super().__init__(client, VERIFIER_SYSTEM_PROMPT, temperature=0.0)

    def verify(self, step: str, executor_output: str) -> Tuple[bool, str]:
        prompt = f"Instruction step:\n{step}\n\nExecutor's answer:\n{executor_output}\n\nDid the executor correctly follow the step?\nRespond with APPROVE: <reason> or REVISE: <reason>"
        verdict = self._call_llm(prompt)
        
        print("\n[VERIFIER OUTPUT]\n", verdict)
        print("-" * 60)
        
        is_approved = verdict.strip().upper().startswith("APPROVE:")
        return is_approved, verdict

# -------------------------
# Pipeline Orchestrator
# -------------------------
class MultiAgentOrchestrator:
    """Manages the lifecycle of planning, executing, and verifying tasks."""
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.planner = PlannerAgent(self.client)
        self.executor = ExecutorAgent(self.client)
        self.verifier = VerifierAgent(self.client)

    def run_pipeline(self, task: str, text: str) -> Dict:
        print("=" * 60)
        print(f"USER TASK: {task}")
        print("=" * 60)

        # 1) PLAN
        steps = self.planner.plan(task, text)
        print("\n[PARSED STEPS]")
        for i, s in enumerate(steps, start=1):
            print(f"Step {i}: {s}")
        print("-" * 60)

        results = []
        
        # 2 & 3) EXECUTE AND VERIFY
        for i, step in enumerate(steps, start=1):
            print(f"\n=== PROCESSING STEP {i} ===")
            exec_output = self.executor.execute(step, text)
            
            is_approved, verdict = self.verifier.verify(step, exec_output)
            
            # Simple retry logic (1 retry)
            if not is_approved:
                print("[ORCHESTRATOR] Verifier requested a revision. Retrying...")
                exec_output = self.executor.execute(step, text)
                is_approved, verdict = self.verifier.verify(step, exec_output)

            if is_approved:
                print(f"[ORCHESTRATOR] Step {i} approved.")
                results.append(exec_output)
            else:
                print(f"[ORCHESTRATOR] Step {i} not approved after retry. Skipping.")
                results.append(f"[Step {i} failed verification]")

        return {
            "steps": steps,
            "results": results,
            "final_answer": "\n\n".join(results),
        }

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    demo_text = """
    Large language models (LLMs) are powerful tools for working with text.
    They can summarise documents, answer questions, generate code, and help with research.
    However, they can also make mistakes or hallucinate information that is not in the source material.
    One way to reduce these errors is to use multiple agents with different roles, such as planner, executor, and verifier.
    By splitting responsibilities, we can often catch mistakes and improve reliability.
    """.strip()

    user_task = "Summarise this text in 3–4 sentences and then list 3 key bullet points."

    # Initialize orchestrator with the API key
    orchestrator = MultiAgentOrchestrator(api_key=os.getenv("OPENAI_API_KEY"))
    result = orchestrator.run_pipeline(user_task, demo_text)

    print("\n" + "=" * 60)
    print("FINAL COMBINED ANSWER:\n")
    print(result["final_answer"])
    print("=" * 60)