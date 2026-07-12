from openai import OpenAI
from pypdf import PdfReader

# 1. Point the client to your local SLM server instead of the cloud
# If using LM Studio, the default is usually "http://localhost:1234/v1"
# If using Ollama, the default is usually "http://localhost:11434/v1"
client = OpenAI(
    base_url="http://localhost:11434/v1", 
    api_key="not-needed" # Local SLMs don't require API keys!
)

# 2. Set up the exact prompt we drafted earlier
system_prompt = """
You are an expert New Zealand tech recruiter. 
Your task is to compare a candidate's CV to a Job Description.
Calculate a relevance score out of 100% based on:
- Up to 50 points for matching core technical skills.
- Up to 30 points for matching years of experience and seniority.
- Up to 20 points for overlapping industry tools.

Required Output:
1. The final relevance percentage (e.g., "Relevance Score: 75%").
2. A brief, 2-sentence justification for the score.
3. A bulleted list of any critical skills missing from the CV.
"""

# 3. Read data dynamically from your dataSet files
# Note: Replace 'your_cv_file.txt' and 'your_job_file.txt' with the actual filenames inside those folders!

# Read Job Listing PDF
job_reader = PdfReader("dataSet/tradeMeJobListing/job_listing.pdf")
job_description = "".join([page.extract_text() for page in job_reader.pages])

# Read CV PDF
cv_reader = PdfReader("dataSet/trademeCV/Sonny H Tapara CV.pdf")
cv_text = "".join([page.extract_text() for page in cv_reader.pages])

# with open("dataSet/tradeMeJobListing/job_listing.pdf", "r", encoding="utf-8") as job_file:
#     job_description = job_file.read()

# with open("dataSet/trademeCV/Sonny H Tapara CV.pdf", "r", encoding="utf-8") as cv_file:
#     cv_text = cv_file.read()

# Combine them into the user prompt format
user_prompt = f"""
Job Description:
{job_description}

Candidate CV:
{cv_text}
"""

# 4. Send the request to your local SLM
print("Thinking...")
response = client.chat.completions.create(
  model="llama3.2:latest", # Must match a model shown at http://localhost:11434/v1/models
  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
  ],
  temperature=0.1 # Keep temperature low so the AI is more analytical and less "creative" with the score
)

# 5. Print the output
print("\n--- SLM OUTPUT ---")
print(response.choices[0].message.content)