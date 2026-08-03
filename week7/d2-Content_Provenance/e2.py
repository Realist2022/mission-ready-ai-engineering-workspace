import json
import hashlib

def fingerprint(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()

def provenance_record(content, metadata):
    record = {
        "content": content,
        "metadata": metadata
    }
    serialized = json.dumps(record, sort_keys=True)
    return fingerprint(serialized)

metadata = {
    "creator": "AI Generator",
    "timestamp": "2026-01-03"
}

print(provenance_record("Hello world", metadata))