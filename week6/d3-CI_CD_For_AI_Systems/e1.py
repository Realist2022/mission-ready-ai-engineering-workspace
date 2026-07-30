import json

def is_valid_response(text):
    try:
        data = json.loads(text)
        return "answer" in data
    except json.JSONDecodeError:
        return False

# Simulated model outputs
good_output = '{"answer": "Yes"}'
bad_output = 'Sure! The answer is yes.'

print(is_valid_response(good_output))  # True
print(is_valid_response(bad_output))   # False
