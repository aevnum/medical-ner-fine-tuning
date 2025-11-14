# %%
!pip install wandb seqeval --quiet
!pip install protobuf==3.20.*

# %%
from kaggle_secrets import UserSecretsClient
import wandb

# Load your secret
user_secrets = UserSecretsClient()
wandb_key = user_secrets.get_secret("WANDB_API_KEY")

# Login
wandb.login(key=wandb_key)

# %%
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    Trainer, 
    TrainingArguments
)
from datasets import load_dataset
from seqeval.metrics import classification_report, f1_score
import wandb

# Initialize W&B
wandb.init(
    project="biomedical-ner",
    name="pubmedbert-maccrobat-v1",
    tags=["pubmedbert", "maccrobat", "baseline"]
)

print("✅ W&B initialized")

# %%
# Load MACCROBAT from HuggingFace
dataset = load_dataset("ktgiahieu/maccrobat2018_2020")

# Check structure
print(f"Train examples: {len(dataset['train'])}")
print(f"Example structure:")
example = dataset['train'][0]
print(f"  Keys: {example.keys()}")
print(f"  Tokens: {example['tokens'][:10]}")
print(f"  Tags: {example['tags'][:10]}")

# Get unique tags for num_labels
unique_tags = set()
for example in dataset['train']:
    unique_tags.update(example['tags'])

# Ensure every B-XXX tag has a corresponding I-XXX tag
extra_i_tags = set()
for tag in unique_tags:
    if tag.startswith("B-"):
        i_tag = "I-" + tag[2:]
        if i_tag not in unique_tags:
            extra_i_tags.add(i_tag)
unique_tags.update(extra_i_tags)

unique_tags = sorted(list(unique_tags))
num_labels = len(unique_tags)
tag2id = {tag: i for i, tag in enumerate(unique_tags)}
id2tag = {i: tag for tag, i in tag2id.items()}

print(f"\n✅ Found {num_labels} unique entity types")
print(f"Entity types: {unique_tags[:10]}...")  # Show first 10

# %%
# Load tokenizer
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(examples):
    """
    Tokenize text and align labels with tokens.
    
    Key insight: BERT tokenizes words into subwords (e.g., "myocardial" → ["my", "##ocardial"]).
    We need to align BIO tags with these subword tokens.
    
    Strategy: 
    - First subword of a word gets the original tag (B-DISEASE)
    - Continuation subwords get I-tag version (I-DISEASE)
    - Special tokens ([CLS], [SEP]) get ignored (label_id = -100)
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,  # Already pre-tokenized
        max_length=512,
        padding=False  # We'll handle padding in data collator
    )
    
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens → ignore in loss
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word → use original label
                label_ids.append(tag2id[label[word_idx]])
            else:
                # Continuation subwords → convert B-tag to I-tag
                original_tag = label[word_idx]
                if original_tag.startswith("B-"):
                    # B-DISEASE → I-DISEASE
                    label_ids.append(tag2id["I-" + original_tag[2:]])
                else:
                    # Keep I-tags as-is
                    label_ids.append(tag2id[original_tag])
            
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply tokenization to dataset
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names
)

print(f"✅ Tokenized train examples: {len(tokenized_dataset['train'])}")

# %%
from transformers import DataCollatorForTokenClassification

# Data collator pads sequences to same length in each batch
data_collator = DataCollatorForTokenClassification(tokenizer)

print("✅ Data collator created")

# %%
from seqeval.metrics import precision_score, recall_score, f1_score

def compute_metrics(p):
    """
    Compute standard NER metrics: precision, recall, F1.
    Uses seqeval for entity-level evaluation.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Remove -100 labels (special tokens we ignored)
    true_predictions = [
        [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # ✅ Calculate CORRECT metrics using seqeval
    results = {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions)
    }
    
    return results

# %%
from collections import Counter

def calculate_class_weights(dataset, tag2id):
    """Calculate inverse frequency weights for class imbalance"""
    tag_counts = Counter()
    
    for example in dataset['train']:
        for tag in example['tags']:
            tag_counts[tag] += 1
    
    # Calculate weights (inverse frequency)
    total = sum(tag_counts.values())
    weights = {}
    for tag, count in tag_counts.items():
        # Inverse frequency with smoothing
        weights[tag2id[tag]] = total / (count + 100)  # +100 for smoothing
    
    # Convert to tensor
    weight_list = [weights.get(i, 1.0) for i in range(len(tag2id))]
    class_weights = torch.tensor(weight_list, dtype=torch.float32)
    
    return class_weights

class_weights = calculate_class_weights(dataset, tag2id)
print(f"\n✅ Class weights calculated")
print(f"   Rare entity weight example: {class_weights.max():.2f}x")
print(f"   Common entity weight example: {class_weights.min():.2f}x")

# %%
# Load PubMedBERT with token classification head
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2tag,
    label2id=tag2id
)

# Move weights to same device as model
if torch.cuda.is_available():
    class_weights = class_weights.cuda()

# Monkey-patch the model's loss to use weighted cross entropy
original_forward = model.forward

def weighted_forward(*args, **kwargs):
    outputs = original_forward(*args, **kwargs)
    
    # If loss is computed, recompute with weights
    if outputs.loss is not None and 'labels' in kwargs:
        from torch.nn import CrossEntropyLoss
        loss_fct = CrossEntropyLoss(weight=class_weights)
        
        logits = outputs.logits
        labels = kwargs['labels']
        
        # Reshape for loss calculation
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        outputs.loss = loss
    
    return outputs

model.forward = weighted_forward

print(f"✅ Model loaded: {model_name}")
print(f"   Parameters: {model.num_parameters():,}")
print(f"   Entity types: {num_labels}")

print(f"\n✅ Label mappings set:")
print(f"   Sample id2label: {model.config.id2label[0]}, {model.config.id2label[1]}, {model.config.id2label[2]}")
print(f"   Sample label2id: B-AGE={model.config.label2id.get('B-AGE')}, O={model.config.label2id.get('O')}")

# %%
training_args = TrainingArguments(
    output_dir="./pubmedbert-maccrobat",

    # Data
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,

    # Epochs & steps
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,

    # Learning rate
    learning_rate=1e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    
    # Optimization
    optim="adamw_torch",
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,

    # Logging & saving
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",

    # W&B integration
    report_to=["wandb"],
    run_name="pubmedbert-maccrobat-v2",

    # Misc
    seed=42,
    fp16=True,
)

print("✅ Training arguments configured")

# %%
from datasets import DatasetDict

# Split train set into train/validation (e.g., 90% train, 10% val)
split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1, seed=42)
tokenized_dataset = DatasetDict({
    "train": split_dataset["train"],
    "validation": split_dataset["test"]
})

print(f"Train size: {len(tokenized_dataset['train'])}")
print(f"Validation size: {len(tokenized_dataset['validation'])}")

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

print("✅ Trainer initialized")

# %%
# Train!
print("Starting training...")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

train_result = trainer.train()

print("\n✅ Training complete!")
print(f"Final loss: {train_result.training_loss:.4f}")

# %%
# Evaluate on training set (TODO: use actual validation set)
print("Evaluating...")
eval_results = trainer.evaluate()

print("\nEvaluation Results:")
for key, value in eval_results.items():
    print(f"  {key}: {value:.4f}")

# Save final model
trainer.save_model("./pubmedbert-maccrobat-final")

# Save label mappings to config
import json
config_path = "./pubmedbert-maccrobat-final/config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

config['id2label'] = id2tag
config['label2id'] = tag2id

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("\n✅ Model saved to ./pubmedbert-maccrobat-final")
print(f"✅ Label mappings saved to config.json ({len(id2tag)} labels)")

# Log final metrics to W&B
wandb.log(eval_results)
wandb.finish()

# %%
from seqeval.metrics import classification_report

# Get detailed per-entity report
print("\nDetailed Classification Report:")
print("="*80)

# Get predictions on validation set
predictions, labels, _ = trainer.predict(tokenized_dataset["validation"])
predictions = np.argmax(predictions, axis=2)

true_predictions = [
    [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

print(classification_report(true_labels, true_predictions))

# ✅ Find which entity types are failing
from collections import Counter
pred_counts = Counter([tag for seq in true_predictions for tag in seq if tag != 'O'])
gold_counts = Counter([tag for seq in true_labels for tag in seq if tag != 'O'])

print("\nEntity Distribution (Gold vs Predicted):")
print(f"{'Entity Type':<30} {'Gold Count':<15} {'Pred Count':<15} {'Difference'}")
print("-"*80)
for entity in sorted(gold_counts.keys()):
    gold = gold_counts[entity]
    pred = pred_counts.get(entity, 0)
    diff = pred - gold
    print(f"{entity:<30} {gold:<15} {pred:<15} {diff:+d}")

# %%
# Load the trained model
from transformers import pipeline

# Reload best model
best_model = AutoModelForTokenClassification.from_pretrained("./pubmedbert-maccrobat-final")
nlp = pipeline(
    "token-classification",
    model=best_model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

# Test on a sample
test_text = "A 67-year-old female presented with acute myocardial infarction and received aspirin."
results = nlp(test_text)

print("Test Inference:")
print(f"Input: {test_text}\n")
for result in results:
    print(f"  {result['word']:30s} → {result['entity_group']:30s} (conf: {result['score']:.3f})")


