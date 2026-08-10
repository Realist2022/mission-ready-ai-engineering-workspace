from fastapi import FastAPI, Header, HTTPException

app = FastAPI()

API_KEY = "enterprise-key-123"

@app.post("/summarize")
def summarize(text: str, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    summary = text[:50] + "..."  # mock AI logic
    return {"summary": summary}