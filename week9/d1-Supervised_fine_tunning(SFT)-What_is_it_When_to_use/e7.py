from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import numpy as np
import evaluate

args = TrainingArguments(
    output_dir="sft_out",
    num_train_epochs=3,

    # ðŸ‘‡ These lines control WHEN and HOW OFTEN we EVALUATE:
    evaluation_strategy="steps",  # run eval every N steps (not just at the end)
    eval_steps=100,               # do an evaluation pass every 100 training steps

    # ðŸ‘‡ These lines control WHEN we SAVE CHECKPOINTS:
    save_steps=100,               # save a checkpoint every 100 training steps
    save_total_limit=2,           # keep only the last 2 checkpoints on disk

    logging_steps=50,
)

# --- Minimal demo wiring for checkpoints + eval ----------------------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tiny toy dataset
texts = ["good movie", "bad movie", "great film", "terrible film"]
labels = [1, 0, 1, 0]
dataset = Dataset.from_dict({"text": texts, "label": labels})

def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=16)

dataset = dataset.map(tokenize_fn, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Simple accuracy metric for eval
accuracy = evaluate.load("accuracy")

# Turns the modelâ€™s raw outputs on the eval set into class predictions
# and computes accuracy by comparing them to the true labels.
def compute_metrics(pred):
    logits = pred.predictions
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=pred.label_ids)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    eval_dataset=dataset,   # ðŸ‘ˆ this is the data used during eval
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # ðŸ‘ˆ how eval metrics (e.g. accuracy) are computed
)

# When you call trainer.train():
# - every 100 steps â†’ run eval on eval_dataset and compute metrics
# - every 100 steps â†’ save a checkpoint under sft_out/
# trainer.train()
