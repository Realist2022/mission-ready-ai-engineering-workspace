import os
import json
from typing import List, Dict, Tuple
from openai import OpenAI
from pypdf import PdfReader
from jsonschema import validate, ValidationError

# ----------------------------------------------------------------------
# 1. STRUCTURAL SCHEMAS & AGENT PROMPTS (OPTIMIZED FOR LOCAL LLMS)
# ----------------------------------------------------------------------
PLANNER_SYSTEM_PROMPT = """
You are a Recruitment Planner agent.
Analyze the provided Job Description and split the evaluation framework into exactly two clear categories.
Return exactly two plain text lines matching these exact string names:
1. Core Technical Skills
2. Seniority & Experience

Rules:
- Output exactly those two names on separate lines.
- Do not provide any conversational introduction, markdown formatting, or extra text.
"""

# Specialized prompts targeting exact structural fields to prevent model confusion
SKILLS_EXECUTOR_PROMPT = """
You are the Recruitment Data Executor specializing in Technical Capabilities.
Analyze the Job Description and the CV to evaluate technical stacks.

You MUST return ONLY a single JSON object matching this exact schema:
{
  "requirement_category": "Core Technical Skills",
  "total_requirements_in_job": <int: count of unique core tech/tools/frameworks required in the job description>,
  "matched_requirements_in_cv": <int: count of those unique required tech tools found explicitly in the CV>,
  "rationale": "A concise 1-sentence explanation of the matched skills."
}

Rules:
- Return valid JSON only. Do not wrap in markdown code blocks. 
- Set numeric defaults for years to 0.0 if required by the wrapper engine.
"""

EXPERIENCE_EXECUTOR_PROMPT = """
You are the Recruitment Data Executor specializing in Seniority and Tenure.
Analyze the Job Description and the CV to extract explicit experience durations.

You MUST return ONLY a single JSON object matching this exact schema:
{
  "requirement_category": "Seniority & Experience",
  "candidate_years_extracted": <float: total years of relevant software development experience extracted from the CV>,
  "target_years_required": <float: target years of experience explicitly demanded by the job listing>,
  "rationale": "A concise 1-sentence explanation of the computed timeline match."
}

Rules:
- Return valid JSON only. Do not wrap in markdown code blocks.
- If a target range is specified (e.g., 2-5 years), use the lower boundary (2.0) as the target requirement.
"""

VERIFIER_SYSTEM_PROMPT = """
You are the Recruitment Data Verifier agent.
Verify if the EXECUTOR's extracted metrics structurally make sense relative to the category.

Respond in one of two formats ONLY:
1) APPROVE
2) REVISE
"""

# ----------------------------------------------------------------------
# 2. SEPARATED PROCESSING CLASSES
# ----------------------------------------------------------------------
class DocumentParser:
    """Handles extracting raw text from specialized file types like PDFs."""

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            return "".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            raise IOError(f"Failed to read or parse PDF at {pdf_path}: {e}")


class BaseAgent:
    """Base class handling local server connectivity to the Ollama engine using JSON mode."""

    def __init__(self, client: OpenAI, system_prompt: str, temperature: float = 0.0, force_json: bool = False):
        self.client = client
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.model = "llama3.2:latest"
        self.force_json = force_json

    def _call_llm(self, user_content: str) -> str:
        # Enforce json_object formatting natively if specified
        extra_args = {"response_format": {"type": "json_object"}} if self.force_json else {}
        
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
            **extra_args
        )
        return resp.choices[0].message.content.strip()


class PlannerAgent(BaseAgent):
    def plan(self, job_desc: str) -> List[str]:
        output = self._call_llm(f"Analyze this Job Description:\n\n{job_desc[:4000]}")
        steps = []
        for line in output.splitlines():
            cleaned = line.replace("1.", "").replace("2.", "").strip()
            if cleaned in ["Core Technical Skills", "Seniority & Experience"]:
                steps.append(cleaned)
        
        # Fallback safeguard
        if len(steps) < 2:
            return ["Core Technical Skills", "Seniority & Experience"]
        return steps[:2]


class ExecutorAgent:
    """Orchestrates structured task execution by dynamically switching target systems."""
    
    def __init__(self, client: OpenAI):
        self.skills_agent = BaseAgent(client, SKILLS_EXECUTOR_PROMPT, temperature=0.0, force_json=True)
        self.exp_agent = BaseAgent(client, EXPERIENCE_EXECUTOR_PROMPT, temperature=0.0, force_json=True)

    def execute(self, category: str, job_desc: str, cv_text: str) -> str:
        prompt = f"Job Context:\n{job_desc}\n\nCandidate CV:\n{cv_text}"
        if "skill" in category.lower():
            return self.skills_agent._call_llm(prompt)
        else:
            return self.exp_agent._call_llm(prompt)


class VerifierAgent(BaseAgent):
    def verify(self, category: str, executor_output: str) -> Tuple[bool, str]:
        prompt = f"Category:\n{category}\n\nExecutor JSON Data:\n{executor_output}"
        verdict = self._call_llm(prompt)
        return verdict.strip().upper().startswith("APPROVE"), verdict


# ----------------------------------------------------------------------
# 3. DEDICATED MATHEMATICAL ENGINE CLASS
# ----------------------------------------------------------------------
class RelevanceScoringEngine:
    """Natively processes validated semantic facts using deterministic code equations."""

    def __init__(self, skills_weight: float = 0.60, experience_weight: float = 0.40):
        self.skills_weight = skills_weight
        self.experience_weight = experience_weight

    def calculate_scorecard(self, validated_metrics: List[Dict]) -> Dict:
        total_skills_job = 0
        total_skills_cv = 0
        candidate_years = 0.0
        target_years = 0.0

        for metric in validated_metrics:
            category = metric.get("requirement_category", "").lower()

            if "skill" in category:
                total_skills_job = metric.get("total_requirements_in_job", 0)
                total_skills_cv = metric.get("matched_requirements_in_cv", 0)

            if "experience" in category or "senior" in category:
                candidate_years = float(metric.get("candidate_years_extracted", 0.0))
                target_years = float(metric.get("target_years_required", 0.0))

        skills_score = ((total_skills_cv / total_skills_job * 100) if total_skills_job > 0 else 0.0)
        exp_score = (min((candidate_years / target_years) * 100, 100.0) if target_years > 0 else 0.0)

        final_relevance = (self.skills_weight * skills_score) + (self.experience_weight * exp_score)

        return {
            "final_relevance": round(final_relevance, 1),
            "skills_percentage": round(skills_score, 1),
            "skills_raw": f"{total_skills_cv}/{total_skills_job}",
            "experience_percentage": round(exp_score, 1),
            "exp_raw": f"{candidate_years} yrs vs {target_years} yrs",
            "candidate_years": candidate_years,
            "target_years": target_years,
        }


# ----------------------------------------------------------------------
# 4. ORCHESTRATOR CLASS
# ----------------------------------------------------------------------
class MultiAgentJobMatcher:
    """Manages the agent workflow to gather validated metrics data."""

    EXECUTOR_SCHEMA = {
        "type": "object",
        "properties": {
            "requirement_category": {"type": "string"},
            "total_requirements_in_job": {"type": "integer"},
            "matched_requirements_in_cv": {"type": "integer"},
            "candidate_years_extracted": {"type": "number"},
            "target_years_required": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["requirement_category", "rationale"],
    }

    def __init__(self):
        # Connected using Ollama API mapping layer compatibility parameters
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.planner = PlannerAgent(self.client, PLANNER_SYSTEM_PROMPT)
        self.executor = ExecutorAgent(self.client)
        self.verifier = VerifierAgent(self.client, VERIFIER_SYSTEM_PROMPT)

    def extract_metrics(self, job_text: str, cv_text: str) -> List[Dict]:
        categories = self.planner.plan(job_text)
        validated_metrics = []

        for category in categories:
            exec_output = self.executor.execute(category, job_text, cv_text)
            is_approved, verdict_text = self.verifier.verify(category, exec_output)

            if not is_approved:
                # One retry optimization attempt
                exec_output = self.executor.execute(category, job_text + "\n(Verify layout schema compatibility rules)", cv_text)
                is_approved, _ = self.verifier.verify(category, exec_output)

            if is_approved:
                try:
                    parsed = json.loads(exec_output)
                    validate(instance=parsed, schema=self.EXECUTOR_SCHEMA)
                    
                    # Normalize defaults so scoring logic stays deterministic
                    parsed.setdefault("candidate_years_extracted", 0.0)
                    parsed.setdefault("target_years_required", 0.0)
                    parsed.setdefault("total_requirements_in_job", 0)
                    parsed.setdefault("matched_requirements_in_cv", 0)
                    
                    validated_metrics.append(parsed)
                except (json.JSONDecodeError, ValidationError) as e:
                    print(f"\n[DEBUG WARNING]: Validation Failed for {category}. Error: {e}")
                    print(f"[DEBUG WARNING]: Raw payload received: {exec_output}")
                    
        return validated_metrics


# ----------------------------------------------------------------------
# 5. DYNAMIC SYSTEM EXECUTION RUNTIME
# ----------------------------------------------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Resolve direct pipeline paths
    job_pdf = os.path.join(project_root, "dataSet", "tradeMeJobListing", "job_listing.pdf")
    cv_pdf = os.path.join(project_root, "dataSet", "trademeCV", "Sonny H Tapara CV.pdf")

    if not os.path.exists(job_pdf) or not os.path.exists(cv_pdf):
        print("Error: Could not find job_listing.pdf or Sonny H Tapara CV.pdf. Check your dataSet path setup.")
        exit(1)

    # Step 1: Extract Texts
    job_desc = DocumentParser.extract_text_from_pdf(job_pdf)
    cv_text = DocumentParser.extract_text_from_pdf(cv_pdf)

    # Step 2: Extract Metrics using Multi-Agent Pipeline
    matcher = MultiAgentJobMatcher()
    print("Running Semantic Agent Extraction Pipeline...")
    extracted_data = matcher.extract_metrics(job_desc, cv_text)

    # Print extracted validated metrics for debugging
    print("\nValidated Metrics (raw JSON arrays collected):")
    print("-" * 50)
    for m in extracted_data:
        print(json.dumps(m, ensure_ascii=False, indent=2))
    print("-" * 50)

    # Step 3: Run Deterministic Calculations via the Decoupled Calculator Class
    scoring_engine = RelevanceScoringEngine(skills_weight=0.60, experience_weight=0.40)
    report = scoring_engine.calculate_scorecard(extracted_data)

    # Print explicit scorecard output details
    print("\n" + "=" * 50)
    print("OOP COMPUTED RELEVANCE SCORECARD REPORT")
    print("=" * 50)
    print(f"Overall Chance of Getting the Job: {report['final_relevance']}%")
    print(f"• Pillar A (Skills Profile Match): {report['skills_raw']} ({report['skills_percentage']}%)")
    print(f"• Pillar B (Seniority Alignment): {report['exp_raw']} ({report['experience_percentage']}%)")
    print("-" * 50)
    print(f"• Candidate years extracted: {report['candidate_years']}")
    print(f"• Target years required (job): {report['target_years']}")
    print("=" * 50)