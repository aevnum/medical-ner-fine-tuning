# %%
!pip install wandb seqeval --quiet
!pip install protobuf==3.20.*

# %%
import warnings
warnings.filterwarnings('ignore')

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
print(f"Entity types: {unique_tags[:10]}...")
print(f"Sample tags: B-Age={tag2id.get('B-Age')}, O={tag2id.get('O')}")

# %%
# Debug - Check raw data before tokenization
print("\n🔍 DEBUG: Checking raw dataset structure...")
sample = dataset['train'][0]
print(f"Sample keys: {sample.keys()}")
print(f"Sample tokens type: {type(sample['tokens'])}")
print(f"Sample tokens length: {len(sample['tokens'])}")
print(f"First 10 tokens: {sample['tokens'][:10]}")
print(f"First 10 tags: {sample['tags'][:10]}")

# %%
# Load tokenizer
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(examples):
    """
    Tokenize text and align labels with tokens.
    
    Key changes:
    - Added validation for empty examples  # ← NEW: Data validation
    - Better error handling for missing tags  # ← NEW: Error handling
    - Filter out problematic examples  # ← NEW: Filtering
    """
    # ✅ Validate input - filter out empty examples
    valid_indices = []
    valid_tokens = []
    valid_tags = []
    
    for i, (tokens, tags) in enumerate(zip(examples["tokens"], examples["tags"])):
        # ← NEW: Check if example is valid
        if tokens and tags and len(tokens) == len(tags):
            valid_indices.append(i)
            valid_tokens.append(tokens)
            valid_tags.append(tags)
        else:
            print(f"⚠️  Skipping invalid example {i}: tokens={len(tokens) if tokens else 0}, tags={len(tags) if tags else 0}")
    
    # ← NEW: If no valid examples, return empty dict
    if not valid_tokens:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    
    # Tokenize valid examples only
    tokenized_inputs = tokenizer(
        valid_tokens,                    # ← CHANGED: Use filtered tokens
        truncation=True,
        is_split_into_words=True,
        max_length=512,
        padding=False
    )
    
    labels = []
    # ✅ Iterate over valid examples
    for i, label in enumerate(valid_tags):  # ← CHANGED: Use filtered tags
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # ← NEW: Add bounds checking
                if word_idx < len(label):
                    label_ids.append(tag2id[label[word_idx]])
                else:
                    label_ids.append(-100)  # Safety fallback
            else:
                # ← NEW: Add bounds checking
                if word_idx < len(label):
                    original_tag = label[word_idx]
                    if original_tag.startswith("B-"):
                        i_tag = "I-" + original_tag[2:]
                        if i_tag in tag2id:
                            label_ids.append(tag2id[i_tag])
                        else:
                            label_ids.append(tag2id[original_tag])
                    elif original_tag == "O":
                        label_ids.append(tag2id["O"])
                    else:
                        label_ids.append(tag2id[original_tag])
                else:
                    label_ids.append(-100)  # Safety fallback
            
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# %%
print("\n🧪 Testing tokenization on single example...")
test_example = dataset['train'][0]
print(f"Input tokens (first 5): {test_example['tokens'][:5]}")
print(f"Input tags (first 5): {test_example['tags'][:5]}")

# ✅ Must wrap in lists for batched processing (this is how .map() calls it)
test_result = tokenize_and_align_labels({
    'tokens': [test_example['tokens']],    # ✅ List of token lists
    'tags': [test_example['tags']]         # ✅ List of tag lists
})

print(f"\n✅ Tokenization test results:")
print(f"   Number of examples processed: {len(test_result['input_ids'])}")
print(f"   First example input_ids length: {len(test_result['input_ids'][0])}")
print(f"   First example labels length: {len(test_result['labels'][0])}")
print(f"   First 10 input_ids: {test_result['input_ids'][0][:10]}")
print(f"   First 10 labels: {test_result['labels'][0][:10]}")
print(f"   First 10 decoded tokens: {tokenizer.convert_ids_to_tokens(test_result['input_ids'][0][:10])}")

# %%
# # Apply tokenization to dataset
# print("Tokenizing dataset...")
# tokenized_dataset = dataset.map(
#     tokenize_and_align_labels,
#     batched=True,
#     remove_columns=dataset["train"].column_names
# )

# print(f"✅ Tokenized train examples: {len(tokenized_dataset['train'])}")

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
print(f"   Sample id2label: {id2tag[0]}, {id2tag[1]}, {id2tag[2]}")
print(f"   Sample label2id: B-Age={tag2id.get('B-Age')}, O={tag2id.get('O')}")

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
# Create split BEFORE tokenization
from datasets import DatasetDict

# Split BEFORE tokenization (on raw data)
print("\nSplitting dataset into train/validation...")
split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({
    "train": split_dataset["train"],
    "validation": split_dataset["test"]
})

print(f"Raw train size: {len(dataset['train'])}")
print(f"Raw validation size: {len(dataset['validation'])}")

# Then tokenize BOTH splits
print("\nTokenizing both splits...")
tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    batch_size=100,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing"
)

print(f"✅ Tokenized train size: {len(tokenized_dataset['train'])}")
print(f"✅ Tokenized validation size: {len(tokenized_dataset['validation'])}")

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
# ← NEW: Add this entire diagnostic section
print("\n🔍 DIAGNOSTIC: Checking tokenized dataset...")
print("="*80)

# Check if datasets are empty
print(f"Train dataset size: {len(tokenized_dataset['train'])}")
print(f"Validation dataset size: {len(tokenized_dataset['validation'])}")

if len(tokenized_dataset['train']) == 0:
    print("❌ ERROR: Training dataset is empty after tokenization!")
    print("   This means all examples were filtered out.")
    print("   Check the warnings above for details.")
else:
    # Check first example structure
    first_example = tokenized_dataset['train'][0]
    print(f"\n✅ First training example:")
    print(f"   Keys: {first_example.keys()}")
    print(f"   input_ids length: {len(first_example['input_ids'])}")
    print(f"   labels length: {len(first_example['labels'])}")
    print(f"   First 10 input_ids: {first_example['input_ids'][:10]}")
    print(f"   First 10 labels: {first_example['labels'][:10]}")
    
    # Check for common issues
    if len(first_example['input_ids']) == 0:
        print("   ❌ ERROR: input_ids is empty!")
    if len(first_example['labels']) == 0:
        print("   ❌ ERROR: labels is empty!")
    if len(first_example['input_ids']) != len(first_example['labels']):
        print(f"   ⚠️  WARNING: Mismatch - input_ids({len(first_example['input_ids'])}) != labels({len(first_example['labels'])})")

print("="*80)

# %%
# Train!
print("Starting training...")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

train_result = trainer.train()

print("\n✅ Training complete!")
print(f"Final loss: {train_result.training_loss:.4f}")

# Log training summary to W&B
wandb.log({
    "train/final_loss": train_result.training_loss,
    "train/total_epochs": training_args.num_train_epochs,
})

# %%
# Enhanced evaluation with W&B logging
print("Evaluating on validation set...")
eval_results = trainer.evaluate()

print("\nEvaluation Results:")
for key, value in eval_results.items():
    print(f"  {key}: {value:.4f}")

# Log comprehensive evaluation metrics to W&B
wandb.log({
    "eval/final_loss": eval_results["eval_loss"],
    "eval/final_f1": eval_results["eval_f1"],
    "eval/final_precision": eval_results["eval_precision"],
    "eval/final_recall": eval_results["eval_recall"],
})

# Save final model locally first (needed for W&B upload)
local_model_path = "./pubmedbert-maccrobat-final"
trainer.save_model(local_model_path)

# Save label mappings to config
import json
config_path = f"{local_model_path}/config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

config['id2label'] = id2tag
config['label2id'] = tag2id

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n✅ Model saved locally to {local_model_path}")
print(f"✅ Label mappings saved to config.json ({len(id2tag)} labels)")

# Save model to W&B
print("\n📤 Uploading model to W&B...")
artifact = wandb.Artifact(
    name="pubmedbert-maccrobat-ner",
    type="model",
    description=f"PubMedBERT fine-tuned on MACCROBAT NER with {len(id2tag)} entity types",
    metadata={
        "num_labels": len(id2tag),
        "model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "dataset": "ktgiahieu/maccrobat2018_2020",
        "final_f1": eval_results["eval_f1"],
        "final_precision": eval_results["eval_precision"],
        "final_recall": eval_results["eval_recall"],
        "num_epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
    }
)

# Add model files to artifact
artifact.add_dir(local_model_path)
wandb.log_artifact(artifact)

print("✅ Model uploaded to W&B!")

# Log final summary statistics to W&B
wandb.summary.update({
    # Overall metrics
    "final_f1": eval_results["eval_f1"],
    "final_precision": eval_results["eval_precision"],
    "final_recall": eval_results["eval_recall"],
    "final_loss": eval_results["eval_loss"],
    
    # Training info
    "total_epochs": training_args.num_train_epochs,
    "learning_rate": training_args.learning_rate,
    "batch_size": training_args.per_device_train_batch_size,
    "num_labels": len(id2tag),
    
    # Dataset info
    "train_samples": len(tokenized_dataset["train"]),
    "val_samples": len(tokenized_dataset["validation"]),
    
    # Best entity performance
    "best_entity_f1": max(entity_f1_scores.values()) if entity_f1_scores else 0.0,
    "worst_entity_f1": min(entity_f1_scores.values()) if entity_f1_scores else 0.0,
})

print("\n✅ All metrics logged to W&B")
print(f"📊 View run at: {wandb.run.url}")

# %%
# Add W&B logging for per-entity metrics
from seqeval.metrics import classification_report
import re

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

# Get classification report as string and print
report_str = classification_report(true_labels, true_predictions)
print(report_str)

# Parse classification report and log to W&B
# Extract per-entity metrics from classification report
report_dict = classification_report(true_labels, true_predictions, output_dict=True)

# Log per-entity F1 scores to W&B
entity_f1_scores = {}
for entity_type, metrics in report_dict.items():
    if entity_type not in ['micro avg', 'macro avg', 'weighted avg'] and isinstance(metrics, dict):
        entity_f1_scores[f"entity_f1/{entity_type}"] = metrics.get('f1-score', 0.0)

wandb.log(entity_f1_scores)

# ✅ Find which entity types are failing
from collections import Counter
pred_counts = Counter([tag for seq in true_predictions for tag in seq if tag != 'O'])
gold_counts = Counter([tag for seq in true_labels for tag in seq if tag != 'O'])

print("\nEntity Distribution (Gold vs Predicted):")
print(f"{'Entity Type':<30} {'Gold Count':<15} {'Pred Count':<15} {'Difference'}")
print("-"*80)

# Prepare entity distribution data for W&B
entity_distribution = {}
for entity in sorted(gold_counts.keys()):
    gold = gold_counts[entity]
    pred = pred_counts.get(entity, 0)
    diff = pred - gold
    print(f"{entity:<30} {gold:<15} {pred:<15} {diff:+d}")
    
    # ✅ Log to W&B
    entity_distribution[f"entity_dist/{entity}/gold"] = gold
    entity_distribution[f"entity_dist/{entity}/predicted"] = pred
    entity_distribution[f"entity_dist/{entity}/difference"] = diff

wandb.log(entity_distribution)

# Create W&B Table for entity performance
import pandas as pd

entity_performance_data = []
for entity_type, metrics in report_dict.items():
    if entity_type not in ['micro avg', 'macro avg', 'weighted avg'] and isinstance(metrics, dict):
        entity_performance_data.append({
            'Entity Type': entity_type,
            'Precision': metrics.get('precision', 0.0),
            'Recall': metrics.get('recall', 0.0),
            'F1-Score': metrics.get('f1-score', 0.0),
            'Support': metrics.get('support', 0),
            'Gold Count': gold_counts.get(entity_type, 0),
            'Pred Count': pred_counts.get(entity_type, 0),
        })

entity_perf_df = pd.DataFrame(entity_performance_data)
entity_perf_df = entity_perf_df.sort_values('F1-Score', ascending=False)

# Log as W&B Table
wandb.log({"entity_performance_table": wandb.Table(dataframe=entity_perf_df)})

print("\n✅ Per-entity metrics logged to W&B")

wandb.finish()

# %%
# Instructions for downloading model from W&B
print("\n" + "="*80)
print("MODEL ARTIFACT INFO")
print("="*80)
print(f"✅ Model saved to W&B as artifact: 'pubmedbert-maccrobat-ner'")
print(f"📦 Artifact version: v{wandb.run.summary.get('_wandb', {}).get('artifact_version', 0)}")
print(f"\n📥 To download and use this model later:")
print(f"""
# Download from W&B
import wandb
api = wandb.Api()
artifact = api.artifact('YOUR_USERNAME/biomedical-ner/pubmedbert-maccrobat-ner:latest')
artifact_dir = artifact.download()

# Load model
from transformers import AutoModelForTokenClassification, AutoTokenizer
model = AutoModelForTokenClassification.from_pretrained(artifact_dir)
tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
""")
print("="*80)

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


