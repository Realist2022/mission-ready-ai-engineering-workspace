import json
from openai import OpenAI
from pypdf import PdfReader

# 1. Point the client to your local Ollama SLM server
client = OpenAI(
    base_url="http://localhost:11434/v1", 
    api_key="ollama" 
)

# 2. Updated system prompt forcing a structured JSON data extraction
system_prompt = """
You are an expert New Zealand tech recruiter. 
Your task is to analyze a candidate's CV against a Job Description and extract specific alignment metrics.

CRITICAL INSTRUCTIONS:
1. 'missing_skills' MUST ONLY include core technical skills that are explicitly written in the Job Description but are completely absent from the Candidate's CV.
2. Do NOT invent or assume required skills (like C# or .NET) if they are not explicitly typed in the Job Description.
3. If a skill is listed in the Candidate's CV (e.g., OpenAI, CrewAI), it can NEVER be listed as a missing skill.

You must return your response as a valid JSON object with the following structure:
{
  "skills_matched": <int: count of core technical skills from the listing found in the CV>,
  "skills_total": <int: total number of core technical skills required in the job listing>,
  "candidate_exp": <float: years of relevant experience the candidate has>,
  "target_exp": <float: years of experience required by the job listing. If not specified, default to 3.0>,
  "alignment_score": <int: score from 0-100 indicating role title and seniority match>,
  "justification": "<string: a brief, 2-sentence justification for these metrics>",
  "missing_skills": ["<string: skill 1>", "<string: skill 2>"]
}

Output ONLY the raw JSON object. Do not include markdown code block syntax.
"""

# 3. Read data dynamically from your dataSet files
job_reader = PdfReader("dataSet/tradeMeJobListing/job_listing.pdf")
job_description = "".join([page.extract_text() for page in job_reader.pages])

cv_reader = PdfReader("dataSet/trademeCV/Sonny H Tapara CV.pdf")
cv_text = "".join([page.extract_text() for page in cv_reader.pages])

user_prompt = f"""
Job Description:
{job_description}

Candidate CV:
{cv_text}
"""

# 4. Send the request to your local SLM with JSON formatting enabled
print("Thinking...")
response = client.chat.completions.create(
  model="llama3.2:latest", 
  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
  ],
  response_format={"type": "json_object"}, # Forces Ollama to return structured JSON data
  temperature=0.1 
)

# 5. Extract, Parse, and Compute the Formula in Python
raw_output = response.choices[0].message.content

try:
    data = json.loads(raw_output)
    
    # Pillar 1: Technical Skills (TS)
    skills_total = data.get("skills_total", 1) # Prevent division by zero
    skills_matched = data.get("skills_matched", 0)
    ts_percentage = (skills_matched / skills_total) * 100
    
    # Pillar 2: Years of Experience (EXP) - capped at 100%
    target_exp = data.get("target_exp", 1)
    candidate_exp = data.get("candidate_exp", 0)
    exp_percentage = min((candidate_exp / target_exp) * 100, 100)
    
    # Pillar 3: Role Seniority Alignment (M)
    m_percentage = data.get("alignment_score", 0)
    
    # Weighted Linear Combination Math
    # 50% Technical Skills + 30% Experience + 20% Role Alignment
    final_relevance = (0.50 * ts_percentage) + (0.30 * exp_percentage) + (0.20 * m_percentage)
    
    # Print the beautiful, calculated output
    print("\n--- DESIGNATED RELEVANCE REPORT ---")
    print(f"Calculated Relevance Score: {final_relevance:.1f}%")
    print("-" * 35)
    print(f"• Technical Skills Match: {skills_matched}/{skills_total} ({ts_percentage:.1f}%)")
    print(f"• Experience Match: {candidate_exp} yrs vs {target_exp} yrs required ({exp_percentage:.1f}%)")
    print(f"• Role Alignment: {m_percentage}%")
    print(f"\nJustification:\n{data.get('justification')}")
    print(f"\nMissing Skills: {', '.join(data.get('missing_skills', []))}")

except json.JSONDecodeError:
    print("\nFailed to parse JSON directly. Raw output from SLM:")
    print(raw_output)