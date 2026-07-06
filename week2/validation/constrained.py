from transformers import pipeline

gen = pipeline("text-generation", model="gpt2")

# 1. Unconstrained vs Constrained
prompt = "The color is"

unconstrained = gen(prompt, max_new_tokens=10, do_sample=True, temperature=1.0)
print(f"âŒ Unconstrained: '{unconstrained[0]['generated_text']}'")

constrained = gen(prompt, max_new_tokens=2, do_sample=False)
print(f"âœ… Constrained:   '{constrained[0]['generated_text']}'")

# # 2. Temperature control  
# prompt2 = "Pick red or blue:"

# random_out = gen(prompt2, max_new_tokens=3, temperature=1.5, do_sample=True)
# print(f"High temp: '{random_out[0]['generated_text']}'")

# controlled = gen(prompt2, max_new_tokens=3, temperature=0.1, do_sample=True) 
# print(f"Low temp:  '{controlled[0]['generated_text']}'")

# # Better constraint: Force shorter output
# prompt3 = "Choose: red or blue. Answer:"
# very_short = gen(prompt3, max_new_tokens=1, do_sample=False)
# print(f"Max 1 token: '{very_short[0]['generated_text']}'")
