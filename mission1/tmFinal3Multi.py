import os
import json
from typing import List, Dict, Tuple
from openai import OpenAI
from pypdf import PdfReader
from jsonschema import validate, ValidationError

# ----------------------------------------------------------------------
# 0. GLOBAL CONFIGURATION & SCHEMAS
# ----------------------------------------------------------------------

from dotenv import load_dotenv
load_dotenv(override=True)
google_api_key = os.getenv('GOOGLE_API_KEY')

# PRIMARY Executor Engine Settings (e.g., Local Ollama)
MODEL_NAME = "llama3.2:latest"
MODEL_BASE_URL = "http://localhost:11434/v1"
MODEL_API_KEY = "ollama"
MODEL_TEMPERATURE = float(0.0)

# SECONDARY Verifier Engine Settings (e.g., Cloud Gemini or GPT)
VERIFIER_MODEL_NAME = "gemini-3.1-flash-lite"
VERIFIER_MODEL_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
VERIFIER_MODEL_API_KEY = google_api_key

SKILLS_WEIGHT = 0.60
EXPERIENCE_WEIGHT = 0.40

DEFAULT_JOB_DIR = "tradeMeJobListing"
DEFAULT_JOB_FILE = "Job_listing.pdf"
DEFAULT_CV_DIR = "tradeMeCV"
DEFAULT_CV_FILE = "Sonny H Tapara CV.pdf"

# ----------------------------------------------------------------------
# 1. STRUCTURAL SCHEMAS & AGENT PROMPTS 
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
1. Identify Target Demands: Scan the Job Description for explicit experience duration demands. 
2. Filter for Domain Relevance: Scan the CV specifically for past roles, industries, or responsibilities that match the domain of the target job listing.
3. Perform Date Math: For every relevant role, extract the start date and end date. Calculate the duration in months. Divide the total months by 12 to get the decimal years.
4. Strict irrelevance exclusion: Do NOT include historical roles that bear zero translatable relationship to the target position's operational domain.
5. VALID TIME CONTEXT: Treat all dates listed on the candidate's CV as valid, finalized past experience, regardless of the current actual year. Do NOT reject dates as "future-dated."

You MUST return ONLY a single JSON object matching this exact schema:
{
  "requirement_category": "Seniority & Experience",
  "calculation_scratchpad": "<string: You MUST write out the math here. List the start and end dates of the roles, calculate the months, and show the division. Ensure your final written result matches the single-decimal rounding used below (e.g., 0.3 years). DO NOT copy this sentence.>",
  "candidate_years_extracted": <float: total years of relevant domain experience found in the CV, rounded to one decimal place>,
  "target_years_required": <float: target minimum years of experience demanded by the job>,
  "rationale": "A concise 1-sentence explanation of the specific roles and timelines calculated."
}

Rules:
- Return valid JSON only. Do not wrap in markdown code blocks.
- Set numeric defaults to 0.0 if no explicit timelines are found.
"""

VERIFIER_SYSTEM_PROMPT_TEMPLATE = """
You are the Recruitment Data Verifier Agent. Your single role is to critically audit the EXECUTOR's JSON payload against the raw text contents using the structural RABCC framework standard.

TARGET INDUSTRY/DOMAIN CONTEXT: {domain_context}

Evaluate the content payload based on these strict framework parameters:
1. RELEVANCE & ACCURACY: Ensure the isolated metrics exactly match the raw text. For experience processing, explicitly check if 'candidate_years_extracted' matches the timeline shown in 'calculation_scratchpad'.
2. RATIONALE AUDIT (BIAS): Inspect the 'rationale' text string. If the executor exhibited systemic processing bias by penalizing non-traditional CV layouts or international terminology variations acceptable under this industry context, flag it for revision.
3. OUTPUT COMPLETENESS: Ensure all required fields are fully populated and free of raw boilerplate sentences.

STRICT OUTPUT FORMAT GAUNTLET RULES:
- If the payload passes all content audits perfectly, reply with exactly one word: APPROVE
- If it fails any metric or shows calculation logic flaws, you MUST write your entire response on a single line using this exact pattern: 
REVISE: <Write a concise, 1-sentence engineering correction detailing exactly what numeric field did not match the text context or scratchpad layout.>
- CRITICAL EXCEPTION: Do NOT fail verification based on the chronological dates of the candidate's experience. Assume all dates on the CV are valid past events.

CRITICAL: Do NOT print out headers like "RELEVANCE & ACCURACY:", "RATIONALE AUDIT", or "OUTPUT COMPLETENESS". Do NOT use bullet points or newline breaks. Your entire output must be either the single word APPROVE or a single line starting with REVISE:.
"""

# ----------------------------------------------------------------------
# 2. CORE PROCESSING & AGENT CLASSES
# ----------------------------------------------------------------------
class DocumentParser:
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            return "".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            raise IOError(f"Failed to read or parse PDF at {pdf_path}: {e}")


class BaseAgent:
    def __init__(self, client: OpenAI, system_prompt: str, model_name: str, force_json: bool = False):
        self.client = client
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.force_json = force_json

    def _call_llm(self, user_content: str) -> str:
        extra_args = {"response_format": {"type": "json_object"}} if self.force_json else {}
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=MODEL_TEMPERATURE,
            **extra_args,
        )
        return resp.choices[0].message.content.strip()


class ExecutorAgent:
    def __init__(self, client: OpenAI):
        # The Executor defaults to the primary model (e.g., local Llama 3.2)
        self.skills_agent = BaseAgent(client, SKILLS_EXECUTOR_PROMPT, model_name=MODEL_NAME, force_json=True)
        self.exp_agent = BaseAgent(client, EXPERIENCE_EXECUTOR_PROMPT, model_name=MODEL_NAME, force_json=True)


class VerifierAgent(BaseAgent):
    def __init__(self, client: OpenAI, system_prompt: str, model_name: str):
        # Pass the specific verifier model name up to the BaseAgent
        super().__init__(client=client, system_prompt=system_prompt, model_name=model_name, force_json=False)

    def verify(self, category: str, executor_output: str) -> Tuple[bool, str]:
        prompt = f"Category:\n{category}\n\nExecutor JSON Data:\n{executor_output}"
        verdict = self._call_llm(prompt)
        return verdict.strip().upper().startswith("APPROVE"), verdict


# ----------------------------------------------------------------------
# 3. DECOUPLED INFRASTRUCTURE CLASSES (SRP Refactoring)
# ----------------------------------------------------------------------
class DomainContextRegistry:
    """Responsibility: Centrally maps and resolves localized industry hints."""
    def __init__(self):
        self._mappings = {
            "Tech": "Look for explicit programming frameworks, cloud patterns, tooling compliance, and software architecture.",
            "Trades": "Look for strict machinery tickets, specialized trade licenses, apprenticeship hours, and Site Safe passes.",
            "Medical": "Look for specific clinical practicing registrations, healthcare certifications, and nursing/medical shift tenures.",
            "General": "Evaluate standard organizational corporate requirements, software operations, and baseline career timelines."
        }

    def resolve_hint(self, industry_type: str) -> str:
        return self._mappings.get(industry_type, self._mappings["General"])


class AgentSelfCorrectionLoop:
    """Responsibility: Controls execution iterations, error parsing, and correction tracking."""
    def __init__(self, executor: ExecutorAgent, verifier: VerifierAgent, max_retries: int = 3):
        self.executor = executor
        self.verifier = verifier
        self.max_retries = max_retries

    def run_bounded_loop(self, category: str, base_prompt: str, schema: Dict) -> Dict:
        current_user_prompt = base_prompt
        attempt = 0
        parsed_payload = None

        while attempt < self.max_retries:
            attempt += 1
            
            # Route execution based on component context
            if "skill" in category.lower() or "competency" in category.lower():
                exec_output = self.executor.skills_agent._call_llm(current_user_prompt)
            else:
                exec_output = self.executor.exp_agent._call_llm(current_user_prompt)

            try:
                parsed_payload = json.loads(exec_output)
                validate(instance=parsed_payload, schema=schema)

                # Supervisor framework semantic check
                is_approved, verdict_text = self.verifier.verify(category, exec_output)
                
                if is_approved:
                    print(f"[SUCCESS] '{category}' passed compliance on loop iteration {attempt}.")
                    return parsed_payload
                
                print(f"[RETRY PROTOCOL] Attempt {attempt} failed verification check: {verdict_text}")
                current_user_prompt = (
                    f"{base_prompt}\n\n"
                    f"CRITICAL REPAIR TASK FROM RUN ENVIRONMENT:\n"
                    f"Your last structural JSON string was rejected by the supervisor engine with this exact log fault:\n"
                    f"\"{verdict_text}\"\n"
                    f"Please re-analyze the original texts, fix the structural error/rounding math inconsistency, and output clean JSON matching the schema rules."
                )
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"[RETRY PROTOCOL] Hard compilation error caught on loop iteration {attempt}: {e}")
                current_user_prompt = f"{base_prompt}\n\nCRITICAL SCHEMATIC RESET:\nYour previous payload structure caused schema or token layout parsing faults. Re-verify the JSON syntax constraints."

        # Fallback safeguard backstop to keep application thread alive
        return parsed_payload


# ----------------------------------------------------------------------
# 4. CLEAN COMPOSED ORCHESTRATOR CLASS
# ----------------------------------------------------------------------
class MultiAgentJobMatcher:
    """Manages the cohesive workflow by delegating tasks to dedicated sub-systems."""
    
    # Schemas remain properties of the contract interface
    SKILLS_SCHEMA = {"type": "object", "properties": {"requirement_category": {"type": "string"}, "total_requirements_in_job": {"type": "integer"}, "matched_requirements_in_cv": {"type": "integer"}, "rationale": {"type": "string"}}, "required": ["requirement_category", "total_requirements_in_job", "matched_requirements_in_cv", "rationale"], "additionalProperties": False}
    EXP_SCHEMA = {"type": "object", "properties": {"requirement_category": {"type": "string"}, "calculation_scratchpad": {"type": "string"}, "candidate_years_extracted": {"type": "number"}, "target_years_required": {"type": "number"}, "rationale": {"type": "string"}}, "required": ["requirement_category", "calculation_scratchpad", "candidate_years_extracted", "target_years_required", "rationale"], "additionalProperties": False}

    def __init__(self):
        # 1. Main Client for the Executor (Local Model)
        self.executor_client = OpenAI(
            base_url=MODEL_BASE_URL, 
            api_key=MODEL_API_KEY
        )
        self.executor = ExecutorAgent(self.executor_client)
        
        # 2. Secondary Client for the Verifier (Cloud Model or Different Local Model)
        self.verifier_client = OpenAI(
            base_url=VERIFIER_MODEL_BASE_URL, 
            api_key=VERIFIER_MODEL_API_KEY
        )
        self.verifier = VerifierAgent(
            client=self.verifier_client, 
            system_prompt="", 
            model_name=VERIFIER_MODEL_NAME
        )
        
        # Compose SRP sub-classes instead of long methods
        self.domain_registry = DomainContextRegistry()
        self.correction_pipeline = AgentSelfCorrectionLoop(self.executor, self.verifier, max_retries=3)

    def extract_metrics(self, job_text: str, cv_text: str, industry_type: str = "Tech") -> List[Dict]:
        """Now entirely decoupled, highly readable coordinator method."""
        validated_metrics = []
        
        # 1. Resolve Industry Lens Context via Registry
        context_hint = self.domain_registry.resolve_hint(industry_type)
        self.verifier.system_prompt = VERIFIER_SYSTEM_PROMPT_TEMPLATE.format(domain_context=context_hint)

        # 2. Execute Orchestrated Sub-loop Processing Pipelines
        base_prompt = f"Job Context:\n{job_text}\n\nCandidate CV:\n{cv_text}"
        
        # Core Technical Skills Pipeline Target
        skills_payload = self.correction_pipeline.run_bounded_loop("Core Technical Skills", base_prompt, self.SKILLS_SCHEMA)
        if skills_payload: validated_metrics.append(skills_payload)
        
        # Experience Timeline Pipeline Target
        exp_payload = self.correction_pipeline.run_bounded_loop("Seniority & Experience", base_prompt, self.EXP_SCHEMA)
        if exp_payload: validated_metrics.append(exp_payload)

        return validated_metrics


# ----------------------------------------------------------------------
# 5. SCORING ENGINE & RUNTIME PLATFORM
# ----------------------------------------------------------------------
class RelevanceScoringEngine:
    def __init__(self, skills_weight: float = SKILLS_WEIGHT, experience_weight: float = EXPERIENCE_WEIGHT):
        self.skills_weight = skills_weight
        self.experience_weight = experience_weight

    def _calculate_skills_score(self, matched: int, total: int) -> float:
        return (matched / total * 100) if total > 0 else 0.0

    def _calculate_experience_score(self, candidate_yrs: float, target_yrs: float) -> float:
        if target_yrs <= 0: return 0.0
        return min((candidate_yrs / target_yrs) * 100, 100.0)

    def calculate_scorecard(self, validated_metrics: List[Dict]) -> Dict:
        total_skills_job, total_skills_cv, candidate_years, target_years = 0, 0, 0.0, 0.0
        for metric in validated_metrics:
            category = metric.get("requirement_category", "").lower()
            if any(k in category for k in ["skill", "competency", "tool"]):
                total_skills_job = metric.get("total_requirements_in_job", 0)
                total_skills_cv = metric.get("matched_requirements_in_cv", 0)
            if any(k in category for k in ["experience", "senior", "years"]):
                candidate_years = float(metric.get("candidate_years_extracted", 0.0))
                target_years = float(metric.get("target_years_required", 0.0))

        skills_score = self._calculate_skills_score(total_skills_cv, total_skills_job)
        exp_score = self._calculate_experience_score(candidate_years, target_years)
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

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    job_pdf = os.path.join(project_root, "dataSet", DEFAULT_JOB_DIR, DEFAULT_JOB_FILE)
    cv_pdf = os.path.join(project_root, "dataSet", DEFAULT_CV_DIR, DEFAULT_CV_FILE)

    if not os.path.exists(job_pdf) or not os.path.exists(cv_pdf):
        print("Error checking directory datasets.")
        exit(1)

    job_desc = DocumentParser.extract_text_from_pdf(job_pdf)
    cv_text = DocumentParser.extract_text_from_pdf(cv_pdf)

    matcher = MultiAgentJobMatcher()
    
    # Print model engine details for debugging
    print(f"Using Executor: {MODEL_NAME} | Verifier: {VERIFIER_MODEL_NAME} | Temperature: {MODEL_TEMPERATURE}")
    print("Running Agent Extraction Loop Pipeline...")
    extracted_data = matcher.extract_metrics(job_desc, cv_text, industry_type="Tech")

    print("\nValidated Metrics (raw JSON arrays collected):")
    print("-" * 50)
    for m in extracted_data:
        print(json.dumps(m, ensure_ascii=False, indent=2))
    print("-" * 50)

    scoring_engine = RelevanceScoringEngine()
    report = scoring_engine.calculate_scorecard(extracted_data)
    
    print(f"\nOverall Scorecard Result: {report['final_relevance']}%")