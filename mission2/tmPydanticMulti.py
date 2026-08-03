import os
import time
from typing import List
from datetime import datetime
from pydantic import BaseModel, Field
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv
from dateutil import parser 
import json

load_dotenv(override=True)

# ----------------------------------------------------------------------
# 0. GLOBAL CONFIGURATION & WEIGHTS
# ----------------------------------------------------------------------

start = time.time()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# google_api_key = os.getenv('GOOGLE_API_KEY')

# # GOOGLE Model Engine Settings
# MODEL_NAME = "gemini-3.1-flash-lite"
# MODEL_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# MODEL_API_KEY = google_api_key
# MODEL_TEMPERATURE = float(0.0)


# OLLAMA Model Engine Settings
MODEL_NAME = "llama3.2:latest"
MODEL_BASE_URL = "http://localhost:11434/v1"
MODEL_API_KEY = "ollama"
MODEL_TEMPERATURE = float(0.0)

# Three-Pillar Score Weights (Must sum to 1.0)
SKILLS_MATCH_WEIGHT = 0.45
SKILL_TENURE_WEIGHT = 0.35
WORK_EXPERIENCE_WEIGHT = 0.20

# ----------------------------------------------------------------------
# 1. PYDANTIC SCHEMAS (AGENT OUTPUTS)
# ----------------------------------------------------------------------

# --- AGENT 1 SCHEMAS ---
class JobRequirement(BaseModel):
    skill_name: str = Field(description="The exact name of the competency, tool, language, license, or platform (e.g., CPA, React, Class 5 License, Epic EMR).")
    target_years: float = Field(default=1.0, description="Minimum years required. Set to 1.0 if not specified.")

class JobRequirementsOutput(BaseModel):
    job_title: str = Field(description="The title of the role from the job description.")
    required_skills: List[JobRequirement] = Field(description="Exhaustive list of ALL technical skills, tools, and platforms required.")

# --- AGENT 2 SCHEMAS ---
class CandidateSkillMatch(BaseModel):
    skill_name: str = Field(description="The skill name matching one from the required list.")
    start_date: str = Field(description="YYYY-MM when the candidate first used this tool in a project or job.")
    end_date: str = Field(description="YYYY-MM when last used, or 'Present'.")

class CVSkillMatchOutput(BaseModel):
    matched_skills: List[CandidateSkillMatch] = Field(description="List of required skills explicitly found in the candidate's CV.")

# --- AGENT 3 SCHEMAS ---
class WorkRole(BaseModel):
    role_title: str = Field(description="Title of the candidate's position.")
    start_date: str = Field(description="Format strictly as YYYY-MM.")
    end_date: str = Field(description="Format strictly as YYYY-MM or 'Present'.")
    match_rationale: str = Field(description="A 1-sentence explanation comparing this role to the target job description to determine if they are in the same industry.")
    is_relevant: bool = Field(description="Set to True ONLY if the match_rationale confirms this role is directly relevant to the target job. Set to False if unrelated.")

class OverallExperienceOutput(BaseModel):
    target_overall_years: float = Field(default=2.0, description="Minimum overall career years demanded by job. Set to 1.0 if unstated.")
    candidate_roles: List[WorkRole] = Field(description="List ALL professional positions found in the CV.")

# ----------------------------------------------------------------------
# 2. SPECIALIZED AGENT PROMPTS
# ----------------------------------------------------------------------

JOB_PARSER_PROMPT = """
You are an Expert Technical Job Parser. Your ONLY task is to read the provided Job Description and extract EVERY required technical skill.
To perform an exhaustive extraction like a senior recruiter, you must follow these strict rules:
1. Extract specific frameworks and tools (e.g., React, Node.js, AWS, PostgreSQL, Git).
2. Extract languages individually. If the text says "JavaScript or TypeScript", you MUST extract "JavaScript" and "TypeScript" as TWO separate items.
3. Extract broader technical methodologies and concepts (e.g., "Relational Databases", "REST APIs", "AI-assisted development tools", "Software Development").
4. Do NOT stop early. A standard intermediate job listing contains 10 to 15 distinct technical requirements. Find and extract them all.
"""

CV_SKILL_MATCHER_PROMPT = """
You are a Precision CV Skill Auditor.
You will be given a Candidate's CV and a target list of required technical skills.
Your task:
1. Scan the CV line-by-line for EACH skill in the provided target list.
2. If the skill is found, extract the start date (YYYY-MM) and end date (YYYY-MM or 'Present') during which the candidate used it in their projects, education, or employment.
3. Ignore skills on the target list that do NOT appear in the CV.
4. Do NOT include skills that are not on the provided target list.
"""

CV_EXPERIENCE_PROMPT = """
You are an Industry-Agnostic Career Tenure Specialist. Follow these steps strictly:
STEP 1: Scan the Job Description for the minimum years of experience required (e.g., if it asks for 2-5 years, extract 2.0).
STEP 2: Scan the CV to extract ALL professional employment roles held by the candidate.
STEP 3: For EACH role, carefully compare it to the target Job Description.
STEP 4: Write a brief 'match_rationale' explaining if the role's industry aligns with the job listing.
STEP 5: Based on your rationale, output 'is_relevant' as True if they align, or False if they are completely unrelated (e.g., Truck Driver is False for a Software role).
"""

# ----------------------------------------------------------------------
# 3. AGENT PIPELINE RUNTIME
# ----------------------------------------------------------------------

class DocumentParser:
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            return "".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            raise IOError(f"Failed to read PDF at {pdf_path}: {e}")

class MultiAgentPipeline:
    def __init__(self):
        self.client = OpenAI(base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY)

    def _call_agent(self, system_prompt: str, user_content: str, response_model):
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

    def run_pipeline(self, job_text: str, cv_text: str) -> dict:
        # AGENT 1: Extract Job Requirements
        print(" -> [Agent 1] Parsing Job Listing for explicit requirements...")
        job_data: JobRequirementsOutput = self._call_agent(
            JOB_PARSER_PROMPT, 
            f"Job Description:\n{job_text}", 
            JobRequirementsOutput
        )

        # Build clean target list for Agent 2
        target_skills_list = [req.skill_name for req in job_data.required_skills]
        print(f"    Found {len(target_skills_list)} target requirements: {', '.join(target_skills_list)}")

        # AGENT 2: Audit CV against isolated skill list
        print(" -> [Agent 2] Auditing CV against extracted job requirements...")
        matcher_input = f"Target Required Skills List:\n{target_skills_list}\n\nCandidate CV:\n{cv_text}"
        cv_skills_data: CVSkillMatchOutput = self._call_agent(
            CV_SKILL_MATCHER_PROMPT, 
            matcher_input, 
            CVSkillMatchOutput
        )

        # AGENT 3: Extract Career Roles from CV
        print(" -> [Agent 3] Extracting career employment history...")
        experience_data: OverallExperienceOutput = self._call_agent(
            CV_EXPERIENCE_PROMPT, 
            f"Job Text:\n{job_text}\n\nCV Text:\n{cv_text}", 
            OverallExperienceOutput
        )

        return {
            "job_requirements": job_data,
            "matched_skills": cv_skills_data,
            "overall_experience": experience_data
        }

# ----------------------------------------------------------------------
# 4. DETERMINISTIC PYTHON SCORING ENGINE
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
        try:
            # Use flexible parsing instead of strict strptime
            start = parser.parse(start_str.strip())
            
            end_clean = end_str.strip().lower()
            if end_clean in ["present", "current", "now"]:
                end = datetime.now()
            else:
                end = parser.parse(end_clean)
            
            days = (end - start).days
            return max(round(days / 365.25, 2), 0.0)
        except (ValueError, TypeError):
            return 0.0

    def calculate_scorecard(self, pipeline_output: dict) -> dict:
        job_reqs: JobRequirementsOutput = pipeline_output["job_requirements"]
        matched_cv: CVSkillMatchOutput = pipeline_output["matched_skills"]
        overall_exp: OverallExperienceOutput = pipeline_output["overall_experience"]

        # --- PILLAR A: SKILLS MATCH (Binary Existence) ---
        target_dict = {req.skill_name.lower(): req.target_years for req in job_reqs.required_skills}
        total_job_skills = len(target_dict)

        # Python defensive check: filter out hallucinations not in original job list
        valid_matches = [
            skill for skill in matched_cv.matched_skills 
            if skill.skill_name.lower() in target_dict
        ]
        total_found = len(valid_matches)
        skills_match_score = (total_found / total_job_skills * 100) if total_job_skills > 0 else 0.0

        # --- PILLAR B: SKILL TENURE (Duration spent using individual tools) ---
        tenure_scores = []
        for skill in valid_matches:
            target_yrs = max(target_dict.get(skill.skill_name.lower(), 1.0), 0.1)
            candidate_yrs = self.calculate_duration_in_years(skill.start_date, skill.end_date)
            tenure_scores.append(min((candidate_yrs / target_yrs) * 100, 100.0))

        skill_tenure_score = (sum(tenure_scores) / len(tenure_scores)) if tenure_scores else 0.0

        # --- PILLAR C: OVERALL CAREER TENURE ---
        # Python now filters based on the SLM's boolean tag
        relevant_roles = [r for r in overall_exp.candidate_roles if r.is_relevant]
        
        total_career_years = sum([
            self.calculate_duration_in_years(r.start_date, r.end_date) 
            for r in relevant_roles
        ])
        
        target_career_years = max(overall_exp.target_overall_years, 0.1)
        work_exp_score = min((total_career_years / target_career_years) * 100, 100.0)

        # --- FINAL WEIGHTED SCORE ---
        final_relevance = (
            (self.weights["skills_match"] * skills_match_score) +
            (self.weights["skill_tenure"] * skill_tenure_score) +
            (self.weights["work_exp"] * work_exp_score)
        )

        return {
                    "final_relevance": round(final_relevance, 1),
                    "pillar_a": {"score": round(skills_match_score, 1), "raw": f"{total_found}/{total_job_skills} skills"},
                    "pillar_b": {"score": round(skill_tenure_score, 1), "raw": f"Avg tenure fit across {total_found} tools"},
                    "pillar_c": {"score": round(work_exp_score, 1), "raw": f"{round(total_career_years, 1)} yrs vs {target_career_years} yrs required"},
                    "validated_skills": [s.skill_name for s in valid_matches],
                    "counted_roles": [r.role_title for r in relevant_roles] # ADD THIS LINE
                }

# ----------------------------------------------------------------------
# 5. EXECUTION ENTRY POINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    job_pdf = os.path.join(project_root, "dataSet", "tradeMeJobListing", "Job_listing.pdf")
    cv_pdf = os.path.join(project_root, "dataSet", "tradeMeCV", "Sonny H Tapara CV.pdf")

    if not os.path.exists(job_pdf) or not os.path.exists(cv_pdf):
        print("Error: Could not find PDF files. Check path setup.")
        exit(1)

    job_desc = DocumentParser.extract_text_from_pdf(job_pdf)
    cv_text = DocumentParser.extract_text_from_pdf(cv_pdf)

    print(f"Engine: {MODEL_NAME} | Architecture: Multi-Agent Pipeline")
    print("Executing Multi-Agent Execution Flow...\n")

    pipeline = MultiAgentPipeline()
    pipeline_data = pipeline.run_pipeline(job_desc, cv_text)

    scoring_engine = RelevanceScoringEngine()
    report = scoring_engine.calculate_scorecard(pipeline_data)

    schema_dict = JobRequirementsOutput.model_json_schema()

    print("\n" + "=" * 60)
    print("MULTI-AGENT COMPUTED RELEVANCE REPORT")
    print("=" * 60)
    print(f"Overall Match Score: {report['final_relevance']}%")
    print("-" * 60)
    print(f"• Pillar A (Skills Match)   [{int(SKILLS_MATCH_WEIGHT*100)}% weight]: {report['pillar_a']['raw']} ({report['pillar_a']['score']}%)")
    print(f"• Pillar B (Skill Tenure)   [{int(SKILL_TENURE_WEIGHT*100)}% weight]: {report['pillar_b']['raw']} ({report['pillar_b']['score']}%)")
    print(f"• Pillar C (Overall Tenure) [{int(WORK_EXPERIENCE_WEIGHT*100)}% weight]: {report['pillar_c']['raw']} ({report['pillar_c']['score']}%)")

    print("-" * 60)
    print(f"-> Validated Matching Skills: {', '.join(report['validated_skills'])}")
    print(f"-> Roles Counted for Tenure: {', '.join(report['counted_roles'])}") # ADD THIS LINE
    print("=" * 60)

    print("\n[DEBUG] Full Report JSON:")
    print(json.dumps(report, indent=4))

    print("Execution time:", round(time.time() - start, 2), "seconds")

