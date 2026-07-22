import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

start = time.time()
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"user","content":"I need a python code that prints the first 10 Fibonacci numbers."}],
)
print(resp.choices[0].message.content)

print("Latency:", time.time() - start, "seconds")