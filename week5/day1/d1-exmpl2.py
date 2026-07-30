import requests
import json

print("ðŸ¦™ Ollama Local LLM Demo")
print("=" * 40)

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

def call_ollama(prompt, model="phi"):
    """Call Ollama API with error handling"""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        return response.json()["response"]
    
    except (requests.exceptions.ConnectionError, ConnectionRefusedError):
        return "âŒ Error: Ollama server not running. Please start Ollama first with 'ollama serve'"
    
    except requests.exceptions.Timeout:
        return "â° Error: Request timed out. Model might be loading or busy."
    
    except KeyboardInterrupt:
        return "â›” Operation cancelled by user."
    
    except Exception as e:
        return f"âŒ Error: {str(e)}"

# Demo 1: Technical Explanation
print("\nðŸ“š Demo 1: Technical Explanation")
print("-" * 35)
prompt1 = "Explain what python is?"
response1 = call_ollama(prompt1)
print(response1)
