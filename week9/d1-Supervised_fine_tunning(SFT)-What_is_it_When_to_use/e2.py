from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

# Minimal demo setup: small toy model + tiny dataset
model_name = "gpt2"  # any small HF model you have installed

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Ensure we have a pad token for padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Tiny in-memory dataset for a quick demo
texts = ["Hello world", "Fine-tuning with Trainer", "Simple SFT example"]
train_dataset = Dataset.from_dict({"text": texts})

# Tokenize dataset
def tokenize_batch(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=32)

train_dataset = train_dataset.map(tokenize_batch, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Check accelerate BEFORE creating TrainingArguments / Trainer
try:
    import accelerate  # type: ignore  # noqa: F401
    accelerate_ok = True
except ImportError:
    accelerate_ok = False

if not accelerate_ok:
    print(
        "Trainer backend (accelerate>=0.26.0) is not installed in this venv.\n"
        "This script will NOT run training, it only shows the setup pattern.\n"
        "To actually run training later, install:\n"
        "  pip install 'transformers[torch]' 'accelerate>=0.26.0'"
    )
else:
    args = TrainingArguments(
        output_dir="sft_out",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()