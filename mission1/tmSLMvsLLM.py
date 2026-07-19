import os
import json
from typing import List, Dict
from openai import OpenAI
from pypdf import PdfReader
from jsonschema import validate, ValidationError
from typer import prompt


# ----------------------------------------------------------------------
# 0. GLOBAL CONFIGURATION
# ----------------------------------------------------------------------

from dotenv import load_dotenv
load_dotenv(override=True)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv('GOOGLE_API_KEY')

# GPT Model Engine Settings
# MODEL_NAME = "gpt-4o"
# MODEL_BASE_URL = "https://api.openai.com/v1"
# MODEL_API_KEY = OPENAI_API_KEY
# MODEL_TEMPERATURE = float(0.0)

# OLLAMA Model Engine Settings
# MODEL_NAME = "llama3.2:latest"
# MODEL_BASE_URL = "http://localhost:11434/v1"
# MODEL_API_KEY = "ollama"
# MODEL_TEMPERATURE = float(0.0)

# GOOGLE Model Engine Settings
MODEL_NAME = "gemini-3.1-flash-lite"
MODEL_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_API_KEY = google_api_key
MODEL_TEMPERATURE = float(0.0)


# Weight Balancing (Ensure they add up to 1.0)
SKILLS_WEIGHT = 0.60
EXPERIENCE_WEIGHT = 0.40

# (Validate they add up to 1.0)
if (SKILLS_WEIGHT + EXPERIENCE_WEIGHT) != 1.0:
    print("WARNING: Your weights do not add up to 1.0!")

# Default Test Files
DEFAULT_JOB_DIR = "tradeMeJobListing"
DEFAULT_JOB_FILE = "Job_listing.pdf"
DEFAULT_CV_DIR = "tradeMeCV"
DEFAULT_CV_FILE = "Sonny H Tapara CV.pdf"

# ----------------------------------------------------------------------
# 1. STRUCTURAL SCHEMAS & AGENT PROMPTS (OPTIMIZED FOR LOCAL LLMS)
# ----------------------------------------------------------------------

SKILLS_EXECUTOR_PROMPT = """
You are an Expert Recruitment Assessor specializing in Core Competency Matching.
Your task is to strictly evaluate the operational skill and tool alignment between a Job Description and a Candidate's CV.

CONTEXT ENGINEERING INSTRUCTIONS:
1. Identify Core Requirements: Extract specific domain tools, machinery, software, methodologies, strict frameworks, certifications, or technical capabilities required in the Job Description.
2. Exclude baseline behavior: Do NOT count generic soft skills (e.g., "hard worker," "good communication," "punctual") as core operational requirements.
3. Functional Matching: If the job asks for a broad capability standard or category and the CV lists a specific tool or certification that directly fulfills it, count it as a successful match.
4. Tally the Metrics: Count the absolute total of unique core requirements discovered in the job. Then, count how many of those are explicitly satisfied by the CV.

You MUST return ONLY a single JSON object matching this exact schema:
{
  "requirement_category": "Core Competencies & Skills",
  "total_requirements_in_job": <int: count of unique core requirements found in the job>,
  "matched_requirements_in_cv": <int: count of those required items found in the CV>,
  "rationale": "A concise 1-sentence explanation listing the matched competencies or tools."
}

Rules:
- Return valid JSON only. Do not wrap in markdown code blocks.
"""

EXPERIENCE_EXECUTOR_PROMPT = """
You are an Expert Recruitment Assessor specializing in Professional Seniority and Tenure.
Your task is to calculate the explicit years of relevant target-domain experience from the CV against the Job Description requirements.

CONTEXT ENGINEERING INSTRUCTIONS:
1. Identify Target Demands: Scan the Job Description for explicit experience duration demands. If a range is given (e.g., "X to Y years"), extract the lower boundary (X) as the target requirement. Do NOT use this abstract example as your answer.
2. Filter for Domain Relevance: Scan the CV specifically for past roles, industries, or responsibilities that match the domain of the target job listing. 
3. Perform Date Math: For every relevant role, extract the start date and end date. Calculate the duration in months. Divide the total months by 12 to get the decimal years. 
4. Strict irrelevance exclusion: Do NOT include historical roles that bear zero translatable relationship to the target position's operational domain.

You MUST return ONLY a single JSON object matching this exact schema:
{
  "requirement_category": "Seniority & Experience",
  "calculation_scratchpad": "<string: You MUST write out the math here. List the start and end dates of the roles, calculate the months, and show the division. DO NOT copy this sentence.>",
  "candidate_years_extracted": <float: total years of relevant domain experience found in the CV, rounded to one decimal>,
  "target_years_required": <float: target minimum years of experience demanded by the job>,
  "rationale": "A concise 1-sentence explanation of the specific roles and timelines calculated."
}

Rules:
- Return valid JSON only. Do not wrap in markdown code blocks.
- Set numeric defaults to 0.0 if no explicit timelines are found.
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

    def __init__(
        self,
        client: OpenAI,
        system_prompt: str,
        temperature: float = MODEL_TEMPERATURE,
        force_json: bool = False,
        model: str = MODEL_NAME,
    ):
        self.client = client
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.model = model
        self.force_json = force_json

    def _call_llm(self, user_content: str) -> str:
        extra_args = (
            {"response_format": {"type": "json_object"}} if self.force_json else {}
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
            **extra_args,
        )
        return resp.choices[0].message.content.strip()

# Composition: used BaseAgent as a component within ExecutorAgent to handle different categories of tasks.
class ExecutorAgent:
    """Orchestrates structured task execution by dynamically switching target systems."""

    def __init__(self, client: OpenAI):
        self.skills_agent = BaseAgent(
            client, SKILLS_EXECUTOR_PROMPT, temperature=MODEL_TEMPERATURE, force_json=True
        )
        self.exp_agent = BaseAgent(
            client, EXPERIENCE_EXECUTOR_PROMPT, temperature=MODEL_TEMPERATURE, force_json=True
        )

    def execute(self, category: str, job_desc: str, cv_text: str) -> str:
        prompt = f"Job Context:\n{job_desc}\n\nCandidate CV:\n{cv_text}"
        if "skill" in category.lower() or "competency" in category.lower():
            return self.skills_agent._call_llm(prompt)
        else:
            return self.exp_agent._call_llm(prompt)


# ----------------------------------------------------------------------
# 3. DEDICATED MATHEMATICAL ENGINE CLASS
# ----------------------------------------------------------------------
class RelevanceScoringEngine:
    """Natively processes validated semantic facts using deterministic code equations."""

    def __init__(self, skills_weight: float = SKILLS_WEIGHT, experience_weight: float = EXPERIENCE_WEIGHT):
        self.skills_weight = skills_weight
        self.experience_weight = experience_weight

    def _calculate_skills_score(self, matched: int, total: int) -> float:
        return (matched / total * 100) if total > 0 else 0.0

    def _calculate_experience_score(self, candidate_yrs: float, target_yrs: float) -> float:
        if target_yrs <= 0:
            return 0.0
        return min((candidate_yrs / target_yrs) * 100, 100.0)

    def calculate_scorecard(self, validated_metrics: List[Dict]) -> Dict:
        total_skills_job = 0
        total_skills_cv = 0
        candidate_years = 0.0
        target_years = 0.0

        for metric in validated_metrics:
            category = metric.get("requirement_category", "").lower()

            if "skill" in category or "competency" in category or "tool" in category or "framework" in category or "certification" in category or "methodology" in category:
                total_skills_job = metric.get("total_requirements_in_job", 0)
                total_skills_cv = metric.get("matched_requirements_in_cv", 0)

            if "experience" in category or "senior" in category or "tenure" in category or "years" in category or "domain" in category or "relevant" in category:
                candidate_years = float(metric.get("candidate_years_extracted", 0.0))
                target_years = float(metric.get("target_years_required", 0.0))

        skills_score = self._calculate_skills_score(total_skills_cv, total_skills_job)
        exp_score = self._calculate_experience_score(candidate_years, target_years)

        final_relevance = (self.skills_weight * skills_score) + (
            self.experience_weight * exp_score
        )

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
# 4. ORCHESTRATOR CLASS FOR MULTI-AGENT PIPELINE
# ----------------------------------------------------------------------
class MultiAgentJobMatcher:
    """Manages the agent workflow to gather validated metrics data using targeted schemas."""

    # Explicit schema for the Core Technical Skills agent output verification
    SKILLS_SCHEMA = {
        "type": "object",
        "properties": {
            "requirement_category": {"type": "string"},
            "total_requirements_in_job": {"type": "integer"},
            "matched_requirements_in_cv": {"type": "integer"},
            "rationale": {"type": "string"},
        },
        "required": [
            "requirement_category",
            "total_requirements_in_job",
            "matched_requirements_in_cv",
            "rationale",
        ],
        "additionalProperties": False,
    }

    # Explicit schema for the Seniority & Experience agent output verification
    EXP_SCHEMA = {
        "type": "object",
        "properties": {
            "requirement_category": {"type": "string"},
            "calculation_scratchpad": {"type": "string"},
            "candidate_years_extracted": {"type": "number"},
            "target_years_required": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": [
            "requirement_category",
            "calculation_scratchpad",
            "candidate_years_extracted",
            "target_years_required",
            "rationale",
        ],
        "additionalProperties": False,
    }

    def __init__(self):
        self.client = OpenAI(base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY)
        self.executor = ExecutorAgent(self.client)

    def extract_metrics(self, job_text: str, cv_text: str) -> List[Dict]:
        categories = ["Core Technical Skills", "Seniority & Experience"]
        validated_metrics = []

        for category in categories:
            exec_output = self.executor.execute(category, job_text, cv_text)

            try:
                parsed = json.loads(exec_output)

                # --- OPTION B: CONDITIONAL SCHEMA VALIDATION ---
                if "skill" in category.lower():
                    validate(instance=parsed, schema=self.SKILLS_SCHEMA)
                else:
                    validate(instance=parsed, schema=self.EXP_SCHEMA)

                validated_metrics.append(parsed)

            except (json.JSONDecodeError, ValidationError) as e:
                print(
                    f"\n[DEBUG WARNING]: Validation Failed for {category}. Error: {e}"
                )
                print(f"[DEBUG WARNING]: Raw payload received: {exec_output}")

        return validated_metrics


# ----------------------------------------------------------------------
# 5. DYNAMIC SYSTEM EXECUTION RUNTIME
# ----------------------------------------------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    job_pdf = os.path.join(
        project_root, "dataSet", DEFAULT_JOB_DIR, DEFAULT_JOB_FILE
    )
    cv_pdf = os.path.join(project_root, "dataSet", DEFAULT_CV_DIR, DEFAULT_CV_FILE)

    if not os.path.exists(job_pdf) or not os.path.exists(cv_pdf):
        print(
            f"Error: Could not find {DEFAULT_JOB_FILE} or {DEFAULT_CV_FILE}. Check your dataSet path setup."
        )
        exit(1)

    # Step 1: Extract Texts
    job_desc = DocumentParser.extract_text_from_pdf(job_pdf)
    cv_text = DocumentParser.extract_text_from_pdf(cv_pdf)

    # Print model engine details for debugging
    print(f"Using Model Engine: {MODEL_NAME} | Temperature: {MODEL_TEMPERATURE}")

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
    scoring_engine = RelevanceScoringEngine(skills_weight=SKILLS_WEIGHT, experience_weight=EXPERIENCE_WEIGHT)
    report = scoring_engine.calculate_scorecard(extracted_data)

    # Print explicit scorecard output details
    print("\n" + "=" * 50)
    print("OOP COMPUTED RELEVANCE SCORECARD REPORT")
    print("=" * 50)
    print(f"Overall Chance of Getting the Job: {report['final_relevance']}%")
    print(
        f"• Pillar A (Skills Profile Match): {report['skills_raw']} ({report['skills_percentage']}%)"
    )
    print(
        f"• Pillar B (Seniority Alignment): {report['exp_raw']} ({report['experience_percentage']}%)"
    )
    print("-" * 50)
    print(f"• Candidate years extracted: {report['candidate_years']}")
    print(f"• Target years required (job): {report['target_years']}")
    print("=" * 50)