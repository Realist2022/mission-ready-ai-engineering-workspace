from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

model_name = "meta-llama/Llama-2-7b-hf"  # example only; pick any supported model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # turn on 4â€‘bit quantization
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",      # QLoRA typically uses NF4
)


tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# 1) Load the base model in 4â€‘bit (QLoRA base)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # let HF place layers on available devices
)

# 2) Define a LoRA adapter config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # typical attention projections
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# 3) Attach the LoRA adapter on top of the 4â€‘bit model
model = get_peft_model(model, lora_config)

# Optional: show which params are trainable (only LoRA adapter weights)
model.print_trainable_parameters()