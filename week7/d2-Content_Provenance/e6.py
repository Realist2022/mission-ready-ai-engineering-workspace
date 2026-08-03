provenance = {
    "origin": "AI-generated",
    "model": "example-model-v1",
    "created_at": "2026-01-03T10:15:00Z",
    "integrity_verified": False,
    "watermark_confidence": 0.87,
    "modifications_detected": False
}

# for k, v in provenance.items():
#     print(f"{k}: {v}")
    
def render_provenance_badge(p):
    if p["origin"] == "AI-generated":
        return "ðŸŸ¦ AI-generated content"
    return "ðŸŸ© Verified human content"

# print(render_provenance_badge(provenance))

def confidence_label(score):
    if score > 0.9:
        return "High confidence AI-generated"
    if score > 0.7:
        return "Likely AI-generated"
    if score > 0.5:
        return "Possibly AI-generated"
    return "Uncertain origin"

# print(confidence_label(provenance["watermark_confidence"]))

def provenance_status(p):
    if not p:
        return "âšª Origin unknown"
    if p["integrity_verified"]:
        return "ðŸŸ¢ Provenance verified"
    return "ðŸŸ  Provenance broken"

print(provenance_status(provenance))
print(provenance_status(None))


