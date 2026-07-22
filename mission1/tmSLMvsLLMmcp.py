import os
import json
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

# 1. IMPORT FASTMCP
from mcp.server.fastmcp import FastMCP

load_dotenv(override=True)

# ----------------------------------------------------------------------
# 0. GLOBAL CONFIGURATION
# ----------------------------------------------------------------------
MODEL_NAME = "llama3.2:latest"
MODEL_BASE_URL = "http://localhost:11434/v1"
MODEL_API_KEY = "ollama"
MODEL_TEMPERATURE = float(0.0)

SKILLS_WEIGHT = 0.60
EXPERIENCE_WEIGHT = 0.40

# 2. INITIALIZE FASTMCP SERVER
# We define the dependencies here so FastMCP can manage the environment using uv.
mcp = FastMCP("Job Matcher", dependencies=["openai>=1.40.0", "pydantic", "pypdf", "python-dotenv"])

# ----------------------------------------------------------------------
# 1. PYDANTIC SCHEMAS (STRUCTURED OUTPUTS)
# ----------------------------------------------------------------------
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
        response = self.client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=MODEL_TEMPERATURE,
            response_format=response_model,
        )
        return response.choices[0].message.parsed

    def extract_metrics(self, job_text: str, cv_text: str) -> dict:
        prompt = f"Job Context:\n{job_text}\n\nCandidate CV:\n{cv_text}"
        
        skills_data = self._call_structured_llm(SKILLS_PROMPT, prompt, SkillsExtraction)
        exp_data = self._call_structured_llm(EXPERIENCE_PROMPT, prompt, ExperienceExtraction)
        
        return {
            "skills": skills_data,
            "experience": exp_data
        }

class RelevanceScoringEngine:
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
# 4. EXPOSE THE TOOL TO THE AI CLIENT
# ----------------------------------------------------------------------
@mcp.tool()
def evaluate_candidate(job_pdf_path: str, cv_pdf_path: str) -> dict:
    """
    Analyzes a candidate's CV against a job listing PDF and returns a structured match scorecard.
    
    Args:
        job_pdf_path: The absolute path to the job description PDF on the local machine.
        cv_pdf_path: The absolute path to the candidate's CV PDF on the local machine.
    """
    
    # 1. Parse the PDFs
    job_desc = DocumentParser.extract_text_from_pdf(job_pdf_path)
    cv_text = DocumentParser.extract_text_from_pdf(cv_pdf_path)
    
    # 2. Run the agent extraction
    matcher = MultiAgentJobMatcher()
    extracted_data = matcher.extract_metrics(job_desc, cv_text)
    
    # 3. Calculate the scorecard
    scoring_engine = RelevanceScoringEngine(skills_weight=SKILLS_WEIGHT, experience_weight=EXPERIENCE_WEIGHT)
    report = scoring_engine.calculate_scorecard(extracted_data)
    
    # FastMCP automatically converts dictionaries to JSON for the AI client
    return report

# ----------------------------------------------------------------------
# 5. START THE SERVER
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Start the FastMCP server so an AI client can connect to it
    mcp.run()