import os
import json
from typing import List, Dict, Tuple
from openai import OpenAI
from pypdf import PdfReader
from jsonschema import validate, ValidationError

# -------------------------
# Local SLM Config Prompts
# -------------------------
# We direct the system prompts to focus on analytical recruitment extraction
PLANNER_SYSTEM_PROMPT = """
You are the Recruitment Planner agent.
Your job is to look at a Job Description and identify the 3 most critical categories of requirements needed for the role.
(e.g., 1. Core Technical Programming Skills, 2. Target Years of Experience, 3. Required Frameworks & Tools).

Rules:
- Output a numbered list of exactly 3 items.
- Keep each step short and action-oriented.
"""

EXECUTOR_SYSTEM_PROMPT = """
You are the Recruitment Data Executor agent.
Your job is to evaluate a candidate's CV against one specific category of job requirements provided by the Planner.

You must output your findings as a strict JSON object with this exact schema:
{
  "requirement_category": "Name of the current category being evaluated",
  "total_requirements_in_job": <int: total number of criteria the job demands in this category>,
  "matched_requirements_in_cv": <int: number of criteria the candidate actually possesses>,
  "candidate_years_extracted": <float: years of experience found, default to 0.0 if not an experience category>,
  "target_years_required": <float: target years required by job listing, default to 0.0 if not an experience category>,
  "rationale": "A concise 1-sentence explanation of the match status"
}
Output ONLY raw JSON. Do not write any conversational text or markdown code block syntax.
"""

VERIFIER_SYSTEM_PROMPT = """
You are the Recruitment Data Verifier agent.
Your job is to check whether the EXECUTOR's extracted metrics accurately reflect the raw data found in the CV.
You are ensuring that if the Executor claims a skill matched, it is explicitly present in the candidate's text.

Respond in one of two formats ONLY:
1) APPROVE: The data is accurate and fits the required JSON structure.
2) REVISE: The executor made an extraction error or failed to output clean JSON. Provide short feedback.
"""

# -------------------------
# Document Extraction
# -------------------------
class DocumentParser:
    """Handles extracting raw text from specialized file types like PDFs."""
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            return "".join([page.extract_text() for page in reader.pages])
        except Exception as e:
            raise IOError(f"Failed to read or parse PDF at {pdf_path}: {e}")

# -------------------------
# Agent Classes
# -------------------------
class BaseAgent:
    """Base class handling local server connectivity to the Ollama engine."""
    def __init__(self, client: OpenAI, system_prompt: str, temperature: float = 0.0):
        self.client = client
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.model = "llama3.2:latest" # Configured for your local Ollama model

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

    def plan(self, job_desc: str) -> List[str]:
        prompt = f"Analyze this Job Description and establish the 3 core criteria pillars:\n\n{job_desc[:2000]}"
        output = self._call_llm(prompt)
        
        print("\n[PLANNER STATUS: CRITERIA EXTRACTION]\n", output)
        print("-" * 60)

        steps = [
            line.split(".", 1)[1].strip() 
            for line in output.splitlines() 
            if line.strip() and line.strip()[0].isdigit() and "." in line
        ]
        return steps if steps else [output.strip()]


class ExecutorAgent(BaseAgent):
    def __init__(self, client: OpenAI):
        super().__init__(client, EXECUTOR_SYSTEM_PROMPT, temperature=0.1)

    def execute(self, category: str, job_desc: str, cv_text: str) -> str:
        prompt = f"""
        Target Category to Evaluate: {category}
        
        Job Description Context:
        {job_desc}
        
        Candidate CV Text:
        {cv_text}
        
        Execute the analysis for this single category and output the structured JSON object.
        """
        result = self._call_llm(prompt)
        print(f"\n[EXECUTOR STATUS: EVALUATING MODULE - {category[:30]}...]\n", result)
        print("-" * 60)
        return result


class VerifierAgent(BaseAgent):
    def __init__(self, client: OpenAI):
        super().__init__(client, VERIFIER_SYSTEM_PROMPT, temperature=0.0)

    def verify(self, category: str, executor_output: str) -> Tuple[bool, str]:
        prompt = f"Category:\n{category}\n\nExecutor JSON:\n{executor_output}\n\nVerify consistency and respond with APPROVE or REVISE."
        verdict = self._call_llm(prompt)
        print("\n[VERIFIER AUDIT OUTCOME]\n", verdict)
        print("-" * 60)
        return verdict.strip().upper().startswith("APPROVE:"), verdict

# -------------------------
# Pipeline Orchestrator
# -------------------------
class MultiAgentJobMatcher:
    """Manages the lifecycle of planning, executing, verifying, and final score calculations."""
    
    EXECUTOR_SCHEMA = {
        "type": "object",
        "properties": {
            "requirement_category": {"type": "string"},
            "total_requirements_in_job": {"type": "integer"},
            "matched_requirements_in_cv": {"type": "integer"},
            "candidate_years_extracted": {"type": "number"},
            "target_years_required": {"type": "number"},
            "rationale": {"type": "string"}
        },
        "required": ["requirement_category", "total_requirements_in_job", "matched_requirements_in_cv", "rationale"]
    }

    def __init__(self):
        # Pointing client to your validated local Ollama instance port
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.planner = PlannerAgent(self.client)
        self.executor = ExecutorAgent(self.client)
        self.verifier = VerifierAgent(self.client)

    def run_matching_pipeline(self, job_path: str, cv_path: str):
        print("=" * 60)
        print("STARTING OLLAMA MULTI-AGENT RELEVANCE PIPELINE")
        print("=" * 60)

        # 1. Parse PDFs
        job_description = DocumentParser.extract_text_from_pdf(job_path)
        cv_text = DocumentParser.extract_text_from_pdf(cv_path)

        # 2. Planner agent breaks down requirements pillars
        categories = self.planner.plan(job_description)
        
        validated_metrics = []

        # 3. Distributed execution loop per requirement pillar
        for i, category in enumerate(categories, start=1):
            print(f"\n>>> PROCESSING STRUCTURAL COMPONENT {i}/{len(categories)} <<<")
            exec_output = self.executor.execute(category, job_description, cv_text)
            is_approved, verdict = self.verifier.verify(category, exec_output)
            
            # Agent self-correction loop (1 retry)
            if not is_approved:
                print("[ORCHESTRATOR] Audit failed validation schema. Triggering agent revision cycle...")
                exec_output = self.executor.execute(f"REVISION INSTRUCTION: Fix syntax. {category}", job_description, cv_text)
                is_approved, verdict = self.verifier.verify(category, exec_output)

            if is_approved:
                try:
                    parsed_json = json.loads(exec_output)
                    validate(instance=parsed_json, schema=self.EXECUTOR_SCHEMA)
                    validated_metrics.append(parsed_json)
                    print(f"[ORCHESTRATOR] Component {i} securely validated and added.")
                except (json.JSONDecodeError, ValidationError) as err:
                    print(f"[ORCHESTRATOR] Fallback rejection. Structuring failed schema constraints: {err}")

        # 4. Deterministic Python Mathematical Reduction Engine
        if not validated_metrics:
            print("❌ Failure: Agents were unable to isolate clean metrics data fields.")
            return

        total_skills_job = 0
        total_skills_cv = 0
        exp_score = 100.0 # Default fallback if no numerical experience category isolated

        for metric in validated_metrics:
            # Aggregate standard matching metrics
            if metric.get("total_requirements_in_job", 0) > 0:
                total_skills_job += metric["total_requirements_in_job"]
                total_skills_cv += metric["matched_requirements_in_cv"]
            
            # Capture experience specific sub-weights
            if metric.get("target_years_required", 0) > 0:
                target = metric["target_years_required"]
                actual = metric["candidate_years_extracted"]
                exp_score = min((actual / target) * 100, 100)

        skills_score = (total_skills_cv / total_skills_job * 100) if total_skills_job > 0 else 70.0
        
        # Linear Combination Formula: 60% Domain Requirements Alignment + 40% Experience Pillar
        final_relevance = (0.60 * skills_score) + (0.40 * exp_score)

        print("\n" + "=" * 60)
        print("AGENT PIPELINE FINAL COMPILED SCORECARD")
        print("=" * 60)
        print(f"Calculated Relevance Alignment: {final_relevance:.1f}%")
        print(f"• Combined Domain Attribute Score: {total_skills_cv}/{total_skills_job} ({skills_score:.1f}%)")
        print(f"• Isolated Experience Metric Profile Score: {exp_score:.1f}%")
        print("\nComponent Rationales:")
        for m in validated_metrics:
            print(f"- [{m['requirement_category']}]: {m['rationale']}")
        print("=" * 60)

# -------------------------
# Main Execution
# -------------------------
# if __name__ == "__main__":
#     # Point paths directly to your environment folders
#     job_listing_pdf = "dataSet/tradeMeJobListing/job_listing.pdf"
#     candidate_cv_pdf = "dataSet/trademeCV/Sonny H Tapara CV.pdf"

#     matcher_engine = MultiAgentJobMatcher()
#     matcher_engine.run_matching_pipeline(job_listing_pdf, candidate_cv_pdf)


# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    # Get the directory where this script actually lives (mission1 folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go one level up to find the project root, then point to the dataSet folders
    project_root = os.path.dirname(script_dir)
    job_listing_pdf = os.path.join(project_root, "dataSet", "tradeMeJobListing", "job_listing.pdf")
    candidate_cv_pdf = os.path.join(project_root, "dataSet", "tradeMeCV", "Sonny H Tapara CV.pdf")

    matcher_engine = MultiAgentJobMatcher()
    matcher_engine.run_matching_pipeline(job_listing_pdf, candidate_cv_pdf)