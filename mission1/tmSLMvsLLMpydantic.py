import os
import json
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv(override=True)

# ----------------------------------------------------------------------
# 0. GLOBAL CONFIGURATION
# ----------------------------------------------------------------------

# GOOGLE Model Engine Settings (Using OpenAI SDK Compatibility)
# MODEL_NAME = "gemini-3.1-flash-lite"
# MODEL_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# MODEL_API_KEY = os.getenv('GOOGLE_API_KEY')
# MODEL_TEMPERATURE = 0.0

# OLLAMA Model Engine Settings
MODEL_NAME = "llama3.2:latest"
MODEL_BASE_URL = "http://localhost:11434/v1"
MODEL_API_KEY = "ollama"
MODEL_TEMPERATURE = float(0.0)

# Weight Balancing
SKILLS_WEIGHT = 0.60
EXPERIENCE_WEIGHT = 0.40

# ----------------------------------------------------------------------
# 1. PYDANTIC SCHEMAS (STRUCTURED OUTPUTS)
# ----------------------------------------------------------------------
# These schemas force the LLM to return strict data types.
# Notice we ask for LISTS of strings, not integer counts.

class SkillsExtraction(BaseModel):
    requirement_category: str = Field(default="Core Competencies & Skills")
    job_core_requirements: List[str] = Field(description="A list of specific domain tools, software, or methodologies strictly required in the job description.")
    matched_skills_in_cv: List[str] = Field(description="A list of the job's core requirements that were explicitly found in the CV.")
    rationale: str = Field(description="A 1-sentence explanation of the match quality.")

class ExperienceExtraction(BaseModel):
    requirement_category: str = Field(default="Seniority & Experience")
    relevant_roles_found: List[str] = Field(description="A list of roles in the CV that are highly relevant to the job domain, including their start and end dates.")
    candidate_years_extracted: float = Field(description="The total decimal years of relevant domain experience found in the CV. Set to 0.0 if none.")
    target_years_required: float = Field(description="The minimum decimal years of experience explicitly demanded by the job. Set to 0.0 if not stated.")
    rationale: str = Field(description="A 1-sentence explanation of the specific roles and timelines calculated.")

# ----------------------------------------------------------------------
# 2. AGENT PROMPTS 
# ----------------------------------------------------------------------

SKILLS_PROMPT = """
You are an Expert Recruitment Assessor specializing in Core Competency Matching.
1. Extract specific domain tools, software, methodologies, and technical capabilities required in the Job Description.
2. Exclude generic soft skills (e.g., "hard worker," "good communication").
3. Scan the Candidate's CV and extract the exact tools and skills that match the job requirements.
"""

EXPERIENCE_PROMPT = """
You are an Expert Recruitment Assessor specializing in Professional Seniority and Tenure.
1. Scan the Job Description for explicit experience duration demands (extract the lower boundary if a range is given).
2. Scan the CV specifically for past roles that match the domain of the target job listing. 
3. Calculate the total duration of these relevant roles in years. Do NOT include completely irrelevant historical roles.
"""

# ----------------------------------------------------------------------
# 3. PROCESSING CLASSES
# ----------------------------------------------------------------------

class DocumentParser:
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            return "".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            raise IOError(f"Failed to read or parse PDF at {pdf_path}: {e}")

class MultiAgentJobMatcher:
    def __init__(self):
        self.client = OpenAI(base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY)

    def _call_structured_llm(self, system_prompt: str, user_content: str, response_model):
        """Uses the new beta.chat.completions.parse method for guaranteed schema adherence."""
        response = self.client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=MODEL_TEMPERATURE,
            response_format=response_model,
        )
        # Returns the cleanly parsed Pydantic object
        return response.choices[0].message.parsed

    def extract_metrics(self, job_text: str, cv_text: str) -> dict:
        prompt = f"Job Context:\n{job_text}\n\nCandidate CV:\n{cv_text}"
        
        # Run both extractions using their strict schemas
        skills_data = self._call_structured_llm(SKILLS_PROMPT, prompt, SkillsExtraction)
        exp_data = self._call_structured_llm(EXPERIENCE_PROMPT, prompt, ExperienceExtraction)
        
        return {
            "skills": skills_data,
            "experience": exp_data
        }

# ----------------------------------------------------------------------
# 4. DETERMINISTIC SCORING ENGINE
# ----------------------------------------------------------------------
class RelevanceScoringEngine:
    """Natively processes validated semantic facts using deterministic code equations."""

    def __init__(self, skills_weight: float = SKILLS_WEIGHT, experience_weight: float = EXPERIENCE_WEIGHT):
        self.skills_weight = skills_weight
        self.experience_weight = experience_weight

    @staticmethod
    def calculate_skills_score(matched: int, total: int) -> float:
        return (matched / total * 100) if total > 0 else 0.0

    @staticmethod
    def calculate_experience_score(candidate_yrs: float, target_yrs: float) -> float:
        if target_yrs <= 0:
            return 0.0
        return min((candidate_yrs / target_yrs) * 100, 100.0)

    def calculate_scorecard(self, extracted_data: dict) -> dict:
        skills = extracted_data["skills"]
        exp = extracted_data["experience"]

        total_skills_job = len(skills.job_core_requirements)
        total_skills_cv = len(skills.matched_skills_in_cv)
        
        candidate_years = exp.candidate_years_extracted
        target_years = exp.target_years_required

        # Calling the static methods 
        skills_score = self.calculate_skills_score(total_skills_cv, total_skills_job)
        exp_score = self.calculate_experience_score(candidate_years, target_years)

        final_relevance = (self.skills_weight * skills_score) + (self.experience_weight * exp_score)

        return {
            "final_relevance": round(final_relevance, 1),
            "skills_percentage": round(skills_score, 1),
            "skills_raw": f"{total_skills_cv}/{total_skills_job}",
            "experience_percentage": round(exp_score, 1),
            "exp_raw": f"{candidate_years} yrs vs {target_years} yrs",
            "extracted_skills": skills.matched_skills_in_cv,
            "relevant_roles": exp.relevant_roles_found
        }

# ----------------------------------------------------------------------
# 5. DYNAMIC SYSTEM EXECUTION RUNTIME
# ----------------------------------------------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 1. Point to your actual PDF paths just like the original script
    job_pdf = os.path.join(project_root, "dataSet", "tradeMeJobListing", "Job_listing.pdf")
    cv_pdf = os.path.join(project_root, "dataSet", "tradeMeCV", "Sonny H Tapara CV.pdf")

    if not os.path.exists(job_pdf) or not os.path.exists(cv_pdf):
        print("Error: Could not find PDF files. Check your path setup.")
        exit(1)

    # 2. Extract the actual text from the PDFs
    job_desc = DocumentParser.extract_text_from_pdf(job_pdf)
    cv_text = DocumentParser.extract_text_from_pdf(cv_pdf)

    print(f"Using Model Engine: {MODEL_NAME} | Temperature: {MODEL_TEMPERATURE}")
    print("Running Semantic Agent Extraction Pipeline...")
    
    # 3. Pass the real text into the matcher
    matcher = MultiAgentJobMatcher()
    extracted_data = matcher.extract_metrics(job_desc, cv_text)

    # 4. Run the calculations
    scoring_engine = RelevanceScoringEngine(skills_weight=SKILLS_WEIGHT, experience_weight=EXPERIENCE_WEIGHT)
    report = scoring_engine.calculate_scorecard(extracted_data)

    print("\n" + "=" * 50)
    print("OOP COMPUTED RELEVANCE SCORECARD REPORT")
    print("=" * 50)
    print(f"Overall Chance of Getting the Job: {report['final_relevance']}%")
    print(f"• Pillar A (Skills Match): {report['skills_raw']} ({report['skills_percentage']}%)")
    print(f"  -> Found: {', '.join(report['extracted_skills'])}")
    print(f"• Pillar B (Seniority Alignment): {report['exp_raw']} ({report['experience_percentage']}%)")
    print(f"  -> Roles applied: {', '.join(report['relevant_roles'])}")
    print("=" * 50)