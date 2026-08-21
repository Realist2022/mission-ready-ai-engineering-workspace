import uuid
import time

def run_ai_request(prompt):

    trace_id = str(uuid.uuid4())
    start_time = time.time()

    # Simulated model response
    response = f"Processed: {prompt}"

    latency = time.time() - start_time

    log_record = {
        "trace_id": trace_id,
        "prompt": prompt,
        "response": response,
        "latency": latency,
        "tokens_used": len(prompt.split()),
        "status": "success"
    }

    print("TRACE LOG:", log_record)

    return response


# Simulate requests
run_ai_request("Explain continual learning")
run_ai_request("Summarize observability in AI")