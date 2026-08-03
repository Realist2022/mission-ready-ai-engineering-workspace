import os
import time
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv
from datetime import datetime

load_dotenv(override=True)

# ----------------------------------------------------------------------
# 0. GLOBAL CONFIGURATION
# ----------------------------------------------------------------------

start = time.time()

# OLLAMA Model Engine Settings
MODEL_NAME = "llama3.2:latest"  # Consider "llama3.1:8b" if hardware permits
MODEL_BASE_URL = "http://localhost:11434/v1"
MODEL_API_KEY = "ollama"
MODEL_TEMPERATURE = float(0.0)

# The Three-Pillar Weights (Must equal 1.0)
SKILLS_MATCH_WEIGHT = 0.45
SKILL_TENURE_WEIGHT = 0.35
WORK_EXPERIENCE_WEIGHT = 0.20

# ----------------------------------------------------------------------
# 1. PYDANTIC SCHEMAS (STRUCTURED OUTPUTS)
# ----------------------------------------------------------------------

class JobRequirement(BaseModel):
    skill_name: str = Field(description="The exact name of the tool or technology.")
    target_years: float = Field(
        description="The minimum years of experience required for this specific skill. Default to 1.0 if not explicitly stated."
    )

class CandidateSkill(BaseModel):
    skill_name: str = Field(description="The exact name of the tool, matching the job requirement.")
    start_date: str = Field(description="YYYY-MM when the candidate first used this skill.")
    end_date: str = Field(description="YYYY-MM when the candidate last used it, or 'Present'.")

class SkillsExtraction(BaseModel):
    requirement_category: str = Field(default="Core Competencies & Skill Tenure")
    job_core_requirements: List[JobRequirement] = Field(
        description="List of specific domain tools explicitly required by the job. Minimum 5 items."
    )
    matched_skills_in_cv: List[CandidateSkill] = Field(
        description="CRITICAL: This must ONLY contain skills from the 'job_core_requirements' list that are found in the CV."
    )

class Role(BaseModel):
    role_title: str = Field(description="The job title held by the candidate.")
    start_date: str = Field(description="Format strictly as YYYY-MM (e.g., '2020-01').")
    end_date: str = Field(description="Format strictly as YYYY-MM or 'Present'.")

class ExperienceExtraction(BaseModel):
    requirement_category: str = Field(default="Overall Seniority & Experience")
    relevant_roles_found: List[Role] = Field(
        description="A list of all professional roles in the CV to establish overall work maturity."
    )
    target_years_required: float = Field(
        description="The minimum decimal years of OVERALL experience explicitly demanded by the job. Set to 1.0 if not stated."
    )

# ----------------------------------------------------------------------
# 2. AGENT PROMPTS 
# ----------------------------------------------------------------------

# SKILLS_PROMPT = """
# You are an Expert Recruitment Assessor. Follow these steps exactly:
# STEP 1: Analyze the Job Description. Extract a list of the core technical tools required and their specific experience demands (default to 1.0 year if unstated).
# STEP 2: Read the Candidate's CV.
# STEP 3: For each tool from STEP 1 found in the CV, extract the date range (YYYY-MM to YYYY-MM) the candidate actively used it across their projects or roles.
# STEP 4: Output ONLY the skills from STEP 1 that explicitly appear in the CV. Do not invent skills.
# """

SKILLS_PROMPT = """
You are an Expert Recruitment Assessor performing an EXHAUSTIVE extraction. Follow these steps exactly:
STEP 1: Deeply analyze the ENTIRE Job Description from start to finish. 
STEP 2: Extract a COMPREHENSIVE list of EVERY SINGLE technical tool, language, and methodology required. Do not stop early. Do not summarize. Find them all.
STEP 3: Read the Candidate's CV completely.
STEP 4: For each tool from STEP 2 found in the CV, extract the date range (YYYY-MM to YYYY-MM).
STEP 5: Output ONLY the skills from STEP 2 that explicitly appear in the CV.
"""

EXPERIENCE_PROMPT = """
You are an Expert Recruitment Assessor evaluating overall professional maturity. Follow these steps exactly:
STEP 1: Scan the Job Description for explicit OVERALL experience duration demands.
STEP 2: Scan the CV and extract the job title, start date (YYYY-MM), and end date (YYYY-MM or 'Present') for all substantial professional roles to establish their total work history duration.
Do NOT calculate the total duration yourself. Just extract the raw role data.
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
        # Uses the Beta Parse API for strict schema adherence 
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

# ----------------------------------------------------------------------
# 4. DETERMINISTIC SCORING ENGINE
# ----------------------------------------------------------------------

class RelevanceScoringEngine:
    def __init__(self):
        self.weights = {
            "skills_match": SKILLS_MATCH_WEIGHT,
            "skill_tenure": SKILL_TENURE_WEIGHT,
            "work_exp": WORK_EXPERIENCE_WEIGHT
        }

    @staticmethod
    def calculate_duration_in_years(start_str: str, end_str: str) -> float:
        """Determines decimal years safely using Python datetime."""
        try:
            start = datetime.strptime(start_str.strip(), "%Y-%m")
            end_clean = end_str.strip().lower()
            if end_clean in ["present", "current", "now"]:
                end = datetime.now()
            else:
                end = datetime.strptime(end_clean, "%Y-%m")
            
            days = (end - start).days
            return max(round(days / 365.25, 2), 0.0)
        except ValueError:
            return 0.0 # Fail gracefully if SLM hallucinates date formats

    def calculate_scorecard(self, extracted_data: dict) -> dict:
        skills_data: SkillsExtraction = extracted_data["skills"]
        exp_data: ExperienceExtraction = extracted_data["experience"]

        # --- PILLAR A: SKILLS MATCH (Binary Existence) ---
        job_req_dict = {req.skill_name.lower(): req.target_years for req in skills_data.job_core_requirements}
        total_skills_job = len(job_req_dict)
        
        valid_matched_skills = []
        for c_skill in skills_data.matched_skills_in_cv:
            if c_skill.skill_name.lower() in job_req_dict:
                valid_matched_skills.append(c_skill)

        total_skills_cv = len(valid_matched_skills)
        skills_match_score = (total_skills_cv / total_skills_job * 100) if total_skills_job > 0 else 0.0

        # --- PILLAR B: SKILL TENURE (Duration of specific tools) ---
        tenure_percentages = []
        for skill in valid_matched_skills:
            target_yrs = job_req_dict.get(skill.skill_name.lower(), 1.0)
            target_yrs = max(target_yrs, 0.1) # Prevent division by zero
            
            candidate_yrs = self.calculate_duration_in_years(skill.start_date, skill.end_date)
            skill_score = min((candidate_yrs / target_yrs) * 100, 100.0)
            tenure_percentages.append(skill_score)

        skill_tenure_score = (sum(tenure_percentages) / len(tenure_percentages)) if tenure_percentages else 0.0

        # --- PILLAR C: OVERALL WORK EXPERIENCE ---
        total_career_years = sum([
            self.calculate_duration_in_years(r.start_date, r.end_date) 
            for r in exp_data.relevant_roles_found
        ])
        
        target_career_years = max(exp_data.target_years_required, 0.1)
        work_exp_score = min((total_career_years / target_career_years) * 100, 100.0)

        # --- FINAL CALCULATION ---
        final_relevance = (
            (self.weights["skills_match"] * skills_match_score) +
            (self.weights["skill_tenure"] * skill_tenure_score) +
            (self.weights["work_exp"] * work_exp_score)
        )

        return {
            "final_relevance": round(final_relevance, 1),
            "pillar_a": {"score": round(skills_match_score, 1), "raw": f"{total_skills_cv}/{total_skills_job} skills"},
            "pillar_b": {"score": round(skill_tenure_score, 1), "raw": f"Avg alignment across {len(valid_matched_skills)} tools"},
            "pillar_c": {"score": round(work_exp_score, 1), "raw": f"{round(total_career_years, 1)} yrs vs {target_career_years} yrs"},
            "extracted_skills": [s.skill_name for s in valid_matched_skills],
        }

# ----------------------------------------------------------------------
# 5. DYNAMIC SYSTEM EXECUTION RUNTIME
# ----------------------------------------------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    job_pdf = os.path.join(project_root, "dataSet", "tradeMeJobListing", "Job_listing.pdf")
    cv_pdf = os.path.join(project_root, "dataSet", "tradeMeCV", "Sonny H Tapara CV.pdf")

    if not os.path.exists(job_pdf) or not os.path.exists(cv_pdf):
        print("Error: Could not find PDF files. Check your path setup.")
        exit(1)

    job_desc = DocumentParser.extract_text_from_pdf(job_pdf)
    cv_text = DocumentParser.extract_text_from_pdf(cv_pdf)

    print(f"Using Model Engine: {MODEL_NAME} | Temperature: {MODEL_TEMPERATURE}")
    print("Running Semantic Agent Extraction Pipeline...")
    
    matcher = MultiAgentJobMatcher()
    extracted_data = matcher.extract_metrics(job_desc, cv_text)

    scoring_engine = RelevanceScoringEngine()
    report = scoring_engine.calculate_scorecard(extracted_data)

    print("\n" + "=" * 60)
    print("THREE-PILLAR COMPUTED RELEVANCE SCORECARD REPORT")
    print("=" * 60)
    print(f"Overall Chance of Getting the Job: {report['final_relevance']}%")
    print("-" * 60)
    print(f"• Pillar A (Skills Match)   [{int(SKILLS_MATCH_WEIGHT*100)}% weight]: {report['pillar_a']['raw']} ({report['pillar_a']['score']}%)")
    print(f"• Pillar B (Skill Tenure)   [{int(SKILL_TENURE_WEIGHT*100)}% weight]: {report['pillar_b']['raw']} ({report['pillar_b']['score']}%)")
    print(f"• Pillar C (Overall Tenure) [{int(WORK_EXPERIENCE_WEIGHT*100)}% weight]: {report['pillar_c']['raw']} ({report['pillar_c']['score']}%)")
    print("-" * 60)
    print(f"-> Validated Tools: {', '.join(report['extracted_skills'])}")
    print("=" * 60)
    print("time:", round(time.time() - start, 2), "seconds")