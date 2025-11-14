# %% [markdown]
# # 0. Project Configuration & Setup

# %% [markdown]
# ## 0.1 Import Libraries

# %%
# ============================================================================
# 0.1 IMPORT LIBRARIES
# ============================================================================

# Standard Libraries
import os
import sys
import json
import random
import warnings
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Progress bars
from tqdm.auto import tqdm

# Machine Learning & NLP
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Transformers & Datasets
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from datasets import load_dataset, Dataset, DatasetDict

# Evaluation
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from seqeval.metrics import (
    classification_report as seqeval_classification_report,
    f1_score as seqeval_f1_score,
    precision_score as seqeval_precision_score,
    recall_score as seqeval_recall_score
)

# Weights & Biases for experiment tracking
import wandb

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Suppress warnings
warnings.filterwarnings('ignore')

# Check GPU availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

print("\n✓ All libraries imported successfully!")

# %% [markdown]
# ## 0.2 Configuration Constants

# %%
# ============================================================================
# 0.2 CONFIGURATION CONSTANTS
# ============================================================================

# Disable parallelism for tokenizers to avoid warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Project Information
PROJECT_NAME = "Medical_Entity_Recognition_Linking"

# Timestamp for experiment tracking
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_NAME = f"MERL_{TIMESTAMP}"

# Dataset Configuration
DATASET_NAME = "ktgiahieu/maccrobat2018_2020"
DATASET_SPLIT_SIZES = {
    'train': 340,
    'validation': 30,
    'test': 30
}

# Model Configuration
MODEL_CONFIGS = {
    'pubmedbert': {
        'name': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
        'display_name': 'PubMedBERT'
    },
    'biobert': {
        'name': 'dmis-lab/biobert-v1.1',
        'display_name': 'BioBERT'
    },
    'bioformer': {
        'name': 'bioformers/bioformer-16L',
        'display_name': 'Bioformer-16L'
    }
}

# Primary model for baseline
PRIMARY_MODEL = 'pubmedbert'
PRIMARY_MODEL_NAME = MODEL_CONFIGS[PRIMARY_MODEL]['name']

# Training Hyperparameters
TRAINING_CONFIG = {
    'learning_rate': 2e-5,
    'batch_size': 16,
    'num_epochs': 5,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'gradient_accumulation_steps': 1,
    'fp16': torch.cuda.is_available(),  # Use mixed precision if GPU available
    'logging_steps': 50,
    'eval_steps': 100,
    'save_steps': 100,
    'save_total_limit': 2,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'f1',
    'greater_is_better': True
}

# Entity Type Tiers (from EDA)
ENTITY_TIERS = {
    'frequent': [
        'Age', 'Diagnostic_procedure', 'Sign_symptom', 'Lab_value',
        'Biological_structure', 'Detailed_description', 'Date',
        'Disease_disorder', 'History', 'Therapeutic_procedure',
        'Medication', 'Dosage', 'Duration', 'Clinical_event',
        'Nonbiological_location'
    ],
    'medium': [
        'Family_history', 'Coreference', 'Sex', 'Distance', 'Other_entity',
        'Area', 'Volume', 'Time', 'Frequency', 'Activity', 'Other_event',
        'Personal_background', 'Administration'
    ],
    'rare': [
        'Subject', 'Occupation', 'Outcome', 'Shape', 'Severity',
        'Qualitative_concept', 'Quantitative_concept', 'Texture',
        'Color', 'Height', 'Weight', 'Biological_attribute', 'Mass'
    ]
}

# High-value entities for medical coding (production focus)
HIGH_VALUE_ENTITIES = [
    'Disease_disorder',
    'Therapeutic_procedure',
    'Medication',
    'Sign_symptom',
    'Diagnostic_procedure'
]

# Confidence Thresholds to Test
CONFIDENCE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

# Output Directories
OUTPUT_DIR = f"./outputs/{EXPERIMENT_NAME}"
MODEL_SAVE_DIR = f"{OUTPUT_DIR}/models"
RESULTS_DIR = f"{OUTPUT_DIR}/results"
PLOTS_DIR = f"{OUTPUT_DIR}/plots"

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Random Seeds for Reproducibility
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

# Display Configuration
print("="*80)
print(f"PROJECT: {PROJECT_NAME}")
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print(f"TIMESTAMP: {TIMESTAMP}")
print("="*80)
print(f"\nPrimary Model: {MODEL_CONFIGS[PRIMARY_MODEL]['display_name']}")
print(f"Model Path: {PRIMARY_MODEL_NAME}")
print(f"\nDataset: {DATASET_NAME}")
print(f"Train: {DATASET_SPLIT_SIZES['train']} | "
      f"Val: {DATASET_SPLIT_SIZES['validation']} | "
      f"Test: {DATASET_SPLIT_SIZES['test']}")
print(f"\nTraining Config:")
for key, value in TRAINING_CONFIG.items():
    print(f"  {key}: {value}")
print(f"\nOutput Directory: {OUTPUT_DIR}")
print("="*80)
print("\n✓ Configuration loaded successfully!")

# %% [markdown]
# ## 0.3 WandB Initialization

# %%
# ============================================================================
# 0.3 WANDB INITIALIZATION
# ============================================================================

# Method 1: Using Kaggle Secrets (if available)
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")
    os.environ['WANDB_API_KEY'] = wandb_api_key
    print("✓ WandB API key loaded from Kaggle Secrets")
except:
    print("⚠ Kaggle Secrets not available. WandB will prompt for login.")
    print("  You can manually set: os.environ['WANDB_API_KEY'] = 'your_key_here'")

# Initialize WandB
try:
    # Login to WandB
    wandb.login()
    
    # Initialize WandB project
    wandb.init(
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        config={
            "model_name": PRIMARY_MODEL_NAME,
            "dataset": DATASET_NAME,
            "timestamp": TIMESTAMP,
            **TRAINING_CONFIG,
            "entity_tiers": ENTITY_TIERS,
            "high_value_entities": HIGH_VALUE_ENTITIES,
            "random_seed": RANDOM_SEED
        },
        tags=[PRIMARY_MODEL],
        notes=f"Baseline NER model training on MACCROBAT dataset. "
              f"Focus: production-readiness metrics and error analysis."
    )
    
    print("="*80)
    print(f"✓ WandB initialized successfully!")
    print(f"  Project: {PROJECT_NAME}")
    print(f"  Run Name: {EXPERIMENT_NAME}")
    print(f"  Dashboard: {wandb.run.get_url()}")
    print("="*80)
    
    WANDB_ENABLED = True
    
except Exception as e:
    print(f"⚠ WandB initialization failed: {e}")
    print("  Continuing without WandB logging...")
    WANDB_ENABLED = False

# %% [markdown]
# ## 0.4 Helper Functions

# %%
# ============================================================================
# 0.4 HELPER FUNCTIONS
# ============================================================================

def log_to_wandb(data: Dict, step: Optional[int] = None, commit: bool = True):
    """
    Safely log data to WandB if enabled
    
    Args:
        data: Dictionary of metrics to log
        step: Optional step number
        commit: Whether to commit the log immediately
    """
    if WANDB_ENABLED:
        try:
            wandb.log(data, step=step, commit=commit)
        except Exception as e:
            print(f"Warning: Failed to log to WandB: {e}")


def save_results(data: Dict, filename: str, directory: str = RESULTS_DIR):
    """
    Save results to JSON file
    
    Args:
        data: Dictionary to save
        filename: Name of the file (with .json extension)
        directory: Directory to save to
    """
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Results saved to: {filepath}")
    
    # Also log to WandB as artifact
    if WANDB_ENABLED:
        try:
            artifact = wandb.Artifact(
                name=filename.replace('.json', ''),
                type='results'
            )
            artifact.add_file(filepath)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"Warning: Failed to log artifact to WandB: {e}")


def save_plot(fig, filename: str, directory: str = PLOTS_DIR, dpi: int = 300):
    """
    Save matplotlib figure and log to WandB
    
    Args:
        fig: Matplotlib figure object
        filename: Name of the file (with extension)
        directory: Directory to save to
        dpi: Resolution for saved figure
    """
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"✓ Plot saved to: {filepath}")
    
    # Log to WandB
    if WANDB_ENABLED:
        try:
            wandb.log({filename.replace('.png', ''): wandb.Image(filepath)})
        except Exception as e:
            print(f"Warning: Failed to log plot to WandB: {e}")
    
    plt.close(fig)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")


def create_summary_table(results_dict: Dict) -> pd.DataFrame:
    """
    Create a formatted summary table from results dictionary
    
    Args:
        results_dict: Dictionary with model names as keys and metrics as values
    
    Returns:
        Pandas DataFrame with formatted results
    """
    df = pd.DataFrame(results_dict).T
    df = df.round(4)
    return df


print("="*80)
print("✓ Helper functions defined successfully!")
print("\nAvailable functions:")
print("  - log_to_wandb(): Log metrics to WandB")
print("  - save_results(): Save results to JSON")
print("  - save_plot(): Save and log matplotlib figures")
print("  - format_time(): Convert seconds to readable format")
print("  - print_gpu_memory(): Check GPU usage")
print("  - create_summary_table(): Format results as DataFrame")
print("="*80)

# %% [markdown]
# ---

# %% [markdown]
# # 1. Environment Setup & Data Preparation

# %% [markdown]
# ## 1.1 Load MACCROBAT Dataset

# %%
# ============================================================================
# 1.1 LOAD MACCROBAT DATASET
# ============================================================================

print("="*80)
print("Loading MACCROBAT Dataset...")
print("="*80)

# Load the dataset from HuggingFace
medical_ner_data = load_dataset(DATASET_NAME)

print(f"\n✓ Dataset loaded successfully!")
print(f"\nDataset structure:")
print(medical_ner_data)

print(f"\nDataset splits:")
for split in medical_ner_data.keys():
    print(f"  {split}: {len(medical_ner_data[split])} examples")

# Examine first example
print(f"\n" + "="*80)
print("Sample Example:")
print("="*80)
example = medical_ner_data['train'][0]
print(f"Keys: {list(example.keys())}")
print(f"\nFirst 10 tokens: {example['tokens'][:10]}")
print(f"First 10 tags: {example['tags'][:10]}")
print(f"\nTotal tokens in example: {len(example['tokens'])}")
print(f"Total tags in example: {len(example['tags'])}")

# Log to WandB
log_to_wandb({
    'dataset/total_examples': len(medical_ner_data['train']),
    'dataset/name': DATASET_NAME
})

# %% [markdown]
# ## 1.2 Dataset Structure Analysis

# %%
# ============================================================================
# 1.2 DATASET STRUCTURE ANALYSIS
# ============================================================================

print("="*80)
print("Dataset Structure Analysis")
print("="*80)

# Collect all unique tags
all_tags = set()
for example in medical_ner_data['train']:
    all_tags.update(example['tags'])

print(f"\nUnique tags found: {len(all_tags)}")
print(f"\nSample tags:")
for tag in sorted(list(all_tags))[:20]:
    print(f"  - {tag}")

# Count tag frequencies
tag_counter = Counter()
for example in medical_ner_data['train']:
    tag_counter.update(example['tags'])

print(f"\n" + "="*80)
print("Top 10 Most Frequent Tags:")
print("="*80)
for tag, count in tag_counter.most_common(10):
    print(f"  {tag:<30} : {count:>6} occurrences")

# Document length statistics
doc_lengths = [len(example['tokens']) for example in medical_ner_data['train']]

print(f"\n" + "="*80)
print("Document Length Statistics:")
print("="*80)
print(f"  Mean length: {np.mean(doc_lengths):.2f} tokens")
print(f"  Median length: {np.median(doc_lengths):.2f} tokens")
print(f"  Std deviation: {np.std(doc_lengths):.2f}")
print(f"  Min length: {np.min(doc_lengths)} tokens")
print(f"  Max length: {np.max(doc_lengths)} tokens")

# Entity count per document
entity_counts = []
for example in medical_ner_data['train']:
    # Count B- tags (each represents one entity)
    entity_count = sum(1 for tag in example['tags'] if tag.startswith('B-'))
    entity_counts.append(entity_count)

print(f"\n" + "="*80)
print("Entities per Document:")
print("="*80)
print(f"  Mean entities: {np.mean(entity_counts):.2f}")
print(f"  Median entities: {np.median(entity_counts):.2f}")
print(f"  Min entities: {np.min(entity_counts)}")
print(f"  Max entities: {np.max(entity_counts)}")

# Log to WandB
log_to_wandb({
    'dataset/unique_tags': len(all_tags),
    'dataset/mean_doc_length': np.mean(doc_lengths),
    'dataset/mean_entities_per_doc': np.mean(entity_counts)
})

# %% [markdown]
# ## 1.3 Tag Mapping Creation (tag2id, id2tag)

# %%
# ============================================================================
# 1.3 TAG MAPPING CREATION (tag2id, id2tag)
# ============================================================================

print("="*80)
print("Creating Tag Mappings")
print("="*80)

# Collect all unique tags from dataset
unique_tags = set()
for example in medical_ner_data['train']:
    unique_tags.update(example['tags'])

# Ensure every B-XXX tag has a corresponding I-XXX tag
# This is important for proper BIO tagging
extra_i_tags = set()
for tag in unique_tags:
    if tag.startswith("B-"):
        entity_type = tag[2:]  # Remove 'B-' prefix
        i_tag = f"I-{entity_type}"
        if i_tag not in unique_tags:
            extra_i_tags.add(i_tag)
            print(f"  Adding missing I-tag: {i_tag}")

unique_tags.update(extra_i_tags)

# Sort tags for consistent ordering
unique_tags = sorted(list(unique_tags))

print(f"\nTotal unique tags (including added I-tags): {len(unique_tags)}")

# Create mappings
tag2id = {tag: idx for idx, tag in enumerate(unique_tags)}
id2tag = {idx: tag for tag, idx in tag2id.items()}

# Display sample mappings
print(f"\n" + "="*80)
print("Sample Tag Mappings:")
print("="*80)
for i, (tag, idx) in enumerate(list(tag2id.items())[:15]):
    print(f"  {tag:<30} -> {idx:>3}")

print(f"\n..." )
print(f"  (Total: {len(tag2id)} tags)")

# Save mappings
mappings = {
    'tag2id': tag2id,
    'id2tag': id2tag,
    'num_labels': len(unique_tags)
}

save_results(mappings, 'tag_mappings.json')

# Store in global variables for later use
NUM_LABELS = len(unique_tags)

print(f"\n✓ Tag mappings created successfully!")
print(f"  Number of labels: {NUM_LABELS}")

# Log to WandB
log_to_wandb({
    'model/num_labels': NUM_LABELS
})

# %% [markdown]
# ## 1.4 Entity Type Tier Classification

# %%
# ============================================================================
# 1.4 ENTITY TYPE TIER CLASSIFICATION
# ============================================================================

print("="*80)
print("Entity Type Tier Classification")
print("="*80)

# CONFIGURATION: Toggle to remove rare entity types
REMOVE_RARE_ENTITIES = False  # Set to True to convert rare entities to 'O'
RARE_THRESHOLD = 100  # Entities with count < threshold are considered rare

print(f"\nConfiguration:")
print(f"  Remove rare entities: {REMOVE_RARE_ENTITIES}")
print(f"  Rare threshold: < {RARE_THRESHOLD} occurrences")

# Extract entity types from tags (B-XXX and I-XXX)
entity_type_counts = Counter()

for example in medical_ner_data['train']:
    for tag in example['tags']:
        if tag != 'O' and '-' in tag:
            entity_type = tag.split('-')[1]
            entity_type_counts[entity_type] += 1

# Create DataFrame for analysis
entity_df = pd.DataFrame(
    entity_type_counts.items(),
    columns=['Entity_Type', 'Count']
).sort_values('Count', ascending=False)

# Classify into tiers based on frequency
def classify_tier(count):
    if count > 1000:
        return 'frequent'
    elif count >= RARE_THRESHOLD:
        return 'medium'
    else:
        return 'rare'

entity_df['Tier'] = entity_df['Count'].apply(classify_tier)

print(f"\nEntity Type Distribution:")
print(entity_df.to_string(index=False))

print(f"\n" + "="*80)
print("Tier Summary:")
print("="*80)
tier_summary = entity_df['Tier'].value_counts()
for tier, count in tier_summary.items():
    print(f"  {tier.capitalize():<10} : {count:>2} entity types")

# Identify rare entity types
rare_entity_types = entity_df[entity_df['Tier'] == 'rare']['Entity_Type'].tolist()
print(f"\n" + "="*80)
print(f"Rare Entity Types ({len(rare_entity_types)} total):")
print("="*80)
for entity_type in rare_entity_types:
    count = entity_df[entity_df['Entity_Type'] == entity_type]['Count'].values[0]
    print(f"  - {entity_type:<30} : {count:>3} occurrences")

# Function to remove rare entities from dataset
def remove_rare_entities_from_dataset(dataset, rare_entities):
    """
    Convert rare entity tags to 'O' in the dataset
    
    Args:
        dataset: HuggingFace dataset
        rare_entities: List of entity types to remove
    
    Returns:
        Modified dataset with rare entities converted to 'O'
    """
    def convert_tags(example):
        new_tags = []
        for tag in example['tags']:
            if tag != 'O' and '-' in tag:
                entity_type = tag.split('-')[1]
                if entity_type in rare_entities:
                    # Convert rare entity to 'O'
                    new_tags.append('O')
                else:
                    new_tags.append(tag)
            else:
                new_tags.append(tag)
        
        example['tags'] = new_tags
        return example
    
    # Apply conversion
    modified_dataset = dataset.map(convert_tags, desc="Removing rare entities")
    
    return modified_dataset


# Apply rare entity removal if enabled
if REMOVE_RARE_ENTITIES:
    print(f"\n" + "="*80)
    print(f"REMOVING RARE ENTITIES")
    print("="*80)
    print(f"Converting {len(rare_entity_types)} rare entity types to 'O'...")
    
    # Store original dataset for reference
    medical_ner_data_original = medical_ner_data
    
    # Remove rare entities from train split
    medical_ner_data['train'] = remove_rare_entities_from_dataset(
        medical_ner_data['train'],
        rare_entity_types
    )
    
    # Count entities before and after
    original_entity_count = sum(
        1 for ex in medical_ner_data_original['train'] 
        for tag in ex['tags'] if tag.startswith('B-')
    )
    new_entity_count = sum(
        1 for ex in medical_ner_data['train'] 
        for tag in ex['tags'] if tag.startswith('B-')
    )
    
    removed_count = original_entity_count - new_entity_count
    
    print(f"\n✓ Rare entities removed!")
    print(f"  Original entities: {original_entity_count}")
    print(f"  Remaining entities: {new_entity_count}")
    print(f"  Removed: {removed_count} ({removed_count/original_entity_count*100:.2f}%)")
    
    # Update entity_df to reflect removal
    entity_df = entity_df[entity_df['Tier'] != 'rare'].copy()
    
    print(f"\n  Remaining entity types: {len(entity_df)}")
    print(f"    Frequent: {len(entity_df[entity_df['Tier'] == 'frequent'])}")
    print(f"    Medium: {len(entity_df[entity_df['Tier'] == 'medium'])}")

else:
    print(f"\n✓ Keeping all entity types (including rare)")

# Verify predefined tiers match data
print(f"\n" + "="*80)
print("Validating Predefined Tiers:")
print("="*80)

for tier_name, tier_entities in ENTITY_TIERS.items():
    # Skip rare tier validation if rare entities were removed
    if REMOVE_RARE_ENTITIES and tier_name == 'rare':
        print(f"\nRare Tier: [SKIPPED - rare entities removed]")
        continue
    
    actual_tier_entities = entity_df[entity_df['Tier'] == tier_name]['Entity_Type'].tolist()
    
    # Check if predefined entities are in data
    missing = set(tier_entities) - set(actual_tier_entities)
    extra = set(actual_tier_entities) - set(tier_entities)
    
    print(f"\n{tier_name.capitalize()} Tier:")
    print(f"  Predefined: {len(tier_entities)} entity types")
    print(f"  In dataset: {len(actual_tier_entities)} entity types")
    
    if missing:
        print(f"  ⚠ Missing from data: {missing}")
    if extra:
        print(f"  ⚠ Extra in data: {extra}")
    if not missing and not extra:
        print(f"  ✓ Perfect match!")

# Save entity distribution
save_results(
    entity_df.to_dict('records'),
    'entity_type_distribution.json'
)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart of entity frequencies (log scale)
colors = entity_df['Tier'].map({'frequent': 'steelblue', 'medium': 'coral', 'rare': 'lightgreen'})
ax1.barh(entity_df['Entity_Type'], entity_df['Count'], color=colors)
ax1.set_xscale('log')
ax1.set_xlabel('Count (log scale)')
ax1.set_ylabel('Entity Type')
ax1.set_title(f"Entity Type Frequency Distribution{' (Rare Removed)' if REMOVE_RARE_ENTITIES else ''}")
ax1.grid(True, alpha=0.3)

# Pie chart of tier distribution
tier_counts = entity_df.groupby('Tier')['Count'].sum()
ax2.pie(
    tier_counts.values,
    labels=[f"{t.capitalize()}\n({c:,} instances)" for t, c in zip(tier_counts.index, tier_counts.values)],
    autopct='%1.1f%%',
    colors=['steelblue', 'coral', 'lightgreen']
)
ax2.set_title(f"Entity Distribution by Tier{' (Rare Removed)' if REMOVE_RARE_ENTITIES else ''}")

plt.tight_layout()
save_plot(fig, 'entity_tier_distribution.png')

# Log to WandB
log_to_wandb({
    'dataset/entity_types_total': len(entity_df),
    'dataset/frequent_entities': len(entity_df[entity_df['Tier'] == 'frequent']),
    'dataset/medium_entities': len(entity_df[entity_df['Tier'] == 'medium']),
    'dataset/rare_entities': len(entity_df[entity_df['Tier'] == 'rare']),
    'config/remove_rare_entities': REMOVE_RARE_ENTITIES,
    'config/rare_threshold': RARE_THRESHOLD
})

print(f"\n✓ Entity tier classification complete!")

# %% [markdown]
# ## 1.5 Dataset Splitting (Train/Val/Test)

# %%
# ============================================================================
# 1.5 DATASET SPLITTING (Train/Val/Test)
# ============================================================================

print("="*80)
print("Dataset Splitting")
print("="*80)

# Use the predefined split sizes
train_size = DATASET_SPLIT_SIZES['train']
val_size = DATASET_SPLIT_SIZES['validation']
test_size = DATASET_SPLIT_SIZES['test']

# Get the full dataset
full_dataset = medical_ner_data['train']
total_size = len(full_dataset)

print(f"\nTotal dataset size: {total_size}")
print(f"Target split: Train={train_size}, Val={val_size}, Test={test_size}")

# Shuffle and split
indices = list(range(total_size))
random.shuffle(indices)

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:train_size + val_size + test_size]

# Create split datasets
train_dataset = full_dataset.select(train_indices)
val_dataset = full_dataset.select(val_indices)
test_dataset = full_dataset.select(test_indices)

# Create DatasetDict
dataset_split = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

print(f"\n" + "="*80)
print("Split Summary:")
print("="*80)
print(f"  Train: {len(train_dataset)} examples ({len(train_dataset)/total_size*100:.1f}%)")
print(f"  Validation: {len(val_dataset)} examples ({len(val_dataset)/total_size*100:.1f}%)")
print(f"  Test: {len(test_dataset)} examples ({len(test_dataset)/total_size*100:.1f}%)")

# Verify entity distribution across splits
print(f"\n" + "="*80)
print("Entity Distribution Across Splits:")
print("="*80)

for split_name, split_data in [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]:
    entity_count = 0
    for example in split_data:
        entity_count += sum(1 for tag in example['tags'] if tag.startswith('B-'))
    
    avg_entities = entity_count / len(split_data)
    print(f"  {split_name:<12} : {entity_count:>5} entities, {avg_entities:.2f} per doc")

# Log to WandB
log_to_wandb({
    'dataset/train_size': len(train_dataset),
    'dataset/val_size': len(val_dataset),
    'dataset/test_size': len(test_dataset)
})

print(f"\n✓ Dataset split complete!")

# %% [markdown]
# ## 1.6 Tokenization & Data Collator

# %%
# ============================================================================
# 1.6 TOKENIZATION & DATA COLLATOR
# ============================================================================

print("="*80)
print("Setting up Tokenization & Data Collator")
print("="*80)

# Load tokenizer for primary model
print(f"\nLoading tokenizer: {PRIMARY_MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(PRIMARY_MODEL_NAME)

print(f"✓ Tokenizer loaded successfully!")
print(f"  Vocab size: {len(tokenizer)}")
print(f"  Model max length: {tokenizer.model_max_length}")

# ADDED: Set explicit max length for BERT models
MAX_SEQUENCE_LENGTH = 512  # BERT's positional embedding limit

print(f"  Using max sequence length: {MAX_SEQUENCE_LENGTH}")

# Helper function to align labels with tokens
def align_labels_with_tokens(labels, word_ids):
    """
    Align BIO labels with tokenized inputs (handling subword tokenization)
    
    Args:
        labels: Original labels (one per word)
        word_ids: Word IDs from tokenizer (mapping tokens to words)
    
    Returns:
        List of aligned labels
    """
    aligned_labels = []
    previous_word_idx = None
    
    for word_idx in word_ids:
        if word_idx is None:
            # Special tokens get -100 (ignored in loss)
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            # First token of a word gets the label
            aligned_labels.append(labels[word_idx])
        else:
            # Subsequent tokens of same word get -100
            # Alternative: could use I- tag for continuation
            aligned_labels.append(-100)
        
        previous_word_idx = word_idx
    
    return aligned_labels


# Tokenization function
def tokenize_and_align_labels(examples):
    """
    Tokenize inputs and align labels with subword tokens
    
    Args:
        examples: Batch of examples from dataset
    
    Returns:
        Tokenized inputs with aligned labels
    """
    # BEFORE:
    # tokenized_inputs = tokenizer(
    #     examples['tokens'],
    #     truncation=True,
    #     is_split_into_words=True,
    #     padding=False
    # )
    
    # AFTER: Added explicit max_length parameter
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,  # ADDED: Explicit max length
        is_split_into_words=True,
        padding=False  # Will be handled by data collator
    )
    
    all_labels = []
    for i, labels in enumerate(examples['tags']):
        word_ids = tokenized_inputs.word_ids(i)
        
        # Convert string labels to IDs
        label_ids = [tag2id[label] for label in labels]
        
        # Align labels with tokens
        aligned_labels = align_labels_with_tokens(label_ids, word_ids)
        all_labels.append(aligned_labels)
    
    tokenized_inputs['labels'] = all_labels
    return tokenized_inputs


print(f"\n" + "="*80)
print("Tokenizing datasets...")
print("="*80)

# Tokenize all splits
tokenized_datasets = dataset_split.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset_split['train'].column_names,
    desc="Tokenizing datasets"
)

print(f"\n✓ Tokenization complete!")
print(f"\nTokenized dataset structure:")
print(tokenized_datasets)

# Test tokenization with one example
print(f"\n" + "="*80)
print("Sample Tokenized Example:")
print("="*80)
sample = tokenized_datasets['train'][0]
print(f"Input IDs length: {len(sample['input_ids'])}")
print(f"Attention mask length: {len(sample['attention_mask'])}")
print(f"Labels length: {len(sample['labels'])}")
print(f"\nFirst 10 input IDs: {sample['input_ids'][:10]}")
print(f"First 10 labels: {sample['labels'][:10]}")
print(f"First 10 tokens: {tokenizer.convert_ids_to_tokens(sample['input_ids'][:10])}")

# ADDED: Check for truncated sequences
print(f"\n" + "="*80)
print("Checking for truncated sequences:")
print(f"="*80)
truncated_count = 0
max_length_found = 0

for split_name in ['train', 'validation', 'test']:
    for example in tokenized_datasets[split_name]:
        length = len(example['input_ids'])
        max_length_found = max(max_length_found, length)
        if length >= MAX_SEQUENCE_LENGTH:
            truncated_count += 1

print(f"  Max sequence length found: {max_length_found}")
print(f"  Sequences at max length: {truncated_count}")
if truncated_count > 0:
    print(f"  ⚠ {truncated_count} sequences were truncated to {MAX_SEQUENCE_LENGTH} tokens")
else:
    print(f"  ✓ All sequences fit within {MAX_SEQUENCE_LENGTH} tokens")

# Create Data Collator
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True
)

print(f"\n✓ Data collator created successfully!")

# Log to WandB
log_to_wandb({
    'tokenizer/vocab_size': len(tokenizer),
    'tokenizer/model_max_length': tokenizer.model_max_length,
    'tokenizer/used_max_length': MAX_SEQUENCE_LENGTH,  # ADDED
    'tokenizer/truncated_sequences': truncated_count  # ADDED
})

# %% [markdown]
# ## 1.7 Evaluation Metrics Setup

# %%
# ============================================================================
# 1.7 EVALUATION METRICS SETUP
# ============================================================================

print("="*80)
print("Setting up Evaluation Metrics")
print("="*80)

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for NER task
    
    Args:
        eval_pred: Tuple of (predictions, labels)
    
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    
    # Get predicted class (argmax of logits)
    predictions = np.argmax(predictions, axis=2)
    
    # Convert IDs to tags, removing special tokens (label = -100)
    true_labels = []
    true_predictions = []
    
    for prediction, label in zip(predictions, labels):
        true_label = []
        true_prediction = []
        
        for pred_id, label_id in zip(prediction, label):
            if label_id != -100:
                true_label.append(id2tag[label_id])
                true_prediction.append(id2tag[pred_id])
        
        if true_label:  # Only add if there are valid labels
            true_labels.append(true_label)
            true_predictions.append(true_prediction)
    
    # Compute metrics using seqeval
    precision = seqeval_precision_score(true_labels, true_predictions)
    recall = seqeval_recall_score(true_labels, true_predictions)
    f1 = seqeval_f1_score(true_labels, true_predictions)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# Helper function for tier-specific metrics
def compute_metrics_for_tier(predictions, labels, tier_entities, id2tag):
    """
    Compute metrics for a specific entity tier
    
    Args:
        predictions: Predicted labels
        labels: True labels
        tier_entities: List of entity types in this tier
        id2tag: ID to tag mapping
    
    Returns:
        Dictionary with precision, recall, f1
    """
    # Filter predictions and labels for this tier
    tier_pred = []
    tier_true = []
    
    for pred_seq, true_seq in zip(predictions, labels):
        tier_pred_seq = []
        tier_true_seq = []
        
        for p, t in zip(pred_seq, true_seq):
            if t == -100:  # Skip special tokens
                continue
                
            pred_tag = id2tag.get(p, 'O')
            true_tag = id2tag.get(t, 'O')
            
            # Check if entity type is in this tier
            pred_entity_type = pred_tag.split('-')[1] if '-' in pred_tag else None
            true_entity_type = true_tag.split('-')[1] if '-' in true_tag else None
            
            if true_entity_type in tier_entities or pred_entity_type in tier_entities:
                tier_pred_seq.append(pred_tag)
                tier_true_seq.append(true_tag)
        
        if tier_pred_seq:  # Only add if sequence has relevant entities
            tier_pred.append(tier_pred_seq)
            tier_true.append(tier_true_seq)
    
    if not tier_pred:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Compute metrics using seqeval
    precision = seqeval_precision_score(tier_true, tier_pred)
    recall = seqeval_recall_score(tier_true, tier_pred)
    f1 = seqeval_f1_score(tier_true, tier_pred)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


print("✓ Evaluation metrics functions defined!")
print("\nAvailable metric functions:")
print("  - compute_metrics(): Standard NER metrics (P/R/F1)")
print("  - compute_metrics_for_tier(): Tier-specific metrics")
print("\nMetrics use seqeval library for proper BIO tag handling")

# Test metrics on a small sample
print(f"\n" + "="*80)
print("Testing metrics with dummy data:")
print("="*80)

# Create dummy predictions and labels
dummy_labels = [[tag2id['O'], tag2id['B-Disease_disorder'], tag2id['I-Disease_disorder'], tag2id['O']]]
dummy_predictions = [[tag2id['O'], tag2id['B-Disease_disorder'], tag2id['I-Disease_disorder'], tag2id['O']]]

# Convert to format expected by compute_metrics
dummy_logits = np.zeros((1, 4, NUM_LABELS))
for i, pred in enumerate(dummy_predictions[0]):
    dummy_logits[0, i, pred] = 10.0  # High logit for predicted class

test_metrics = compute_metrics((dummy_logits, np.array(dummy_labels)))
print(f"\nTest metrics (perfect match):")
for key, value in test_metrics.items():
    print(f"  {key}: {value:.4f}")

print(f"\n✓ Metrics setup complete!")

# %% [markdown]
# ---

# %% [markdown]
# # 2. Baseline Model: PubMedBERT Fine-tuning

# %% [markdown]
# ## 2.1 Model & Tokenizer Initialization

# %%
# ============================================================================
# 2.1 MODEL & TOKENIZER INITIALIZATION
# ============================================================================

print("="*80)
print("Initializing PubMedBERT Model")
print("="*80)

# Model is already loaded from Section 1.6, but let's reinitialize for clarity
print(f"\nModel: {PRIMARY_MODEL_NAME}")
print(f"Number of labels: {NUM_LABELS}")

# Initialize model for token classification
model = AutoModelForTokenClassification.from_pretrained(
    PRIMARY_MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2tag,
    label2id=tag2id,
    ignore_mismatched_sizes=True  # In case of label mismatch
)

print(f"\n✓ Model loaded successfully!")

# Model summary
print(f"\n" + "="*80)
print("Model Architecture Summary:")
print("="*80)
print(f"  Model type: {model.config.model_type}")
print(f"  Hidden size: {model.config.hidden_size}")
print(f"  Number of hidden layers: {model.config.num_hidden_layers}")
print(f"  Number of attention heads: {model.config.num_attention_heads}")
print(f"  Vocabulary size: {model.config.vocab_size}")
print(f"  Max position embeddings: {model.config.max_position_embeddings}")
print(f"  Number of labels: {model.config.num_labels}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n" + "="*80)
print("Model Parameters:")
print("="*80)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size: ~{total_params * 4 / 1024**2:.2f} MB (FP32)")

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"\n✓ Model moved to: {device}")

if torch.cuda.is_available():
    print_gpu_memory()

# Log to WandB
log_to_wandb({
    'model/total_parameters': total_params,
    'model/trainable_parameters': trainable_params,
    'model/hidden_size': model.config.hidden_size,
    'model/num_layers': model.config.num_hidden_layers,
    'model/device': str(device)
})

# %% [markdown]
# ## 2.2 Training Configuration

# %%
# ============================================================================
# 2.2 TRAINING CONFIGURATION
# ============================================================================

print("="*80)
print("Setting up Training Configuration")
print("="*80)

# Define training arguments
training_args = TrainingArguments(
    # Output directory
    output_dir=MODEL_SAVE_DIR,

    # Explicit run name for WandB
    run_name=EXPERIMENT_NAME,  # Use our experiment name instead of output_dir
    
    # Training hyperparameters
    learning_rate=TRAINING_CONFIG['learning_rate'],
    per_device_train_batch_size=TRAINING_CONFIG['batch_size'],
    per_device_eval_batch_size=TRAINING_CONFIG['batch_size'],
    num_train_epochs=TRAINING_CONFIG['num_epochs'],
    weight_decay=TRAINING_CONFIG['weight_decay'],
    warmup_steps=TRAINING_CONFIG['warmup_steps'],
    gradient_accumulation_steps=TRAINING_CONFIG['gradient_accumulation_steps'],
    
    # Mixed precision training
    fp16=TRAINING_CONFIG['fp16'],
    
    # Evaluation strategy
    eval_strategy="steps",
    eval_steps=TRAINING_CONFIG['eval_steps'],
    save_strategy="steps",
    save_steps=TRAINING_CONFIG['save_steps'],
    save_total_limit=TRAINING_CONFIG['save_total_limit'],
    
    # Logging
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=TRAINING_CONFIG['logging_steps'],
    report_to="wandb" if WANDB_ENABLED else "none",
    
    # Best model tracking
    load_best_model_at_end=TRAINING_CONFIG['load_best_model_at_end'],
    metric_for_best_model=TRAINING_CONFIG['metric_for_best_model'],
    greater_is_better=TRAINING_CONFIG['greater_is_better'],
    
    # Other settings
    push_to_hub=False,
    seed=RANDOM_SEED,
    dataloader_num_workers=0,
    remove_unused_columns=True,
)

print("\n✓ Training arguments configured!")

# Display configuration
print(f"\n" + "="*80)
print("Training Configuration:")
print("="*80)
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Number of epochs: {training_args.num_train_epochs}")
print(f"  Warmup steps: {training_args.warmup_steps}")
print(f"  Weight decay: {training_args.weight_decay}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  FP16 training: {training_args.fp16}")
print(f"  Evaluation steps: {training_args.eval_steps}")
print(f"  Logging steps: {training_args.logging_steps}")
print(f"  Save steps: {training_args.save_steps}")
print(f"  Metric for best model: {training_args.metric_for_best_model}")

# Calculate training steps
num_train_examples = len(tokenized_datasets['train'])
steps_per_epoch = num_train_examples // (
    training_args.per_device_train_batch_size * 
    training_args.gradient_accumulation_steps
)
total_steps = steps_per_epoch * training_args.num_train_epochs

print(f"\n" + "="*80)
print("Training Schedule:")
print("="*80)
print(f"  Training examples: {num_train_examples}")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Total training steps: {total_steps}")
print(f"  Evaluations: ~{total_steps // training_args.eval_steps}")

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print(f"\n✓ Trainer initialized successfully!")

# %% [markdown]
# ## 2.3 Training Loop with WandB Logging

# %%
# ============================================================================
# 2.3 TRAINING LOOP WITH WANDB LOGGING
# ============================================================================

print("="*80)
print("Starting Model Training")
print("="*80)

# Record start time
import time
training_start_time = time.time()

print(f"\nTraining started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {device}")

if torch.cuda.is_available():
    print_gpu_memory()

print(f"\n{'='*80}")
print("Training in progress...")
print(f"{'='*80}\n")

try:
    # Train the model
    train_result = trainer.train()
    
    # Record end time
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    
    print(f"\n{'='*80}")
    print("✓ Training completed successfully!")
    print(f"{'='*80}")
    
    # Display training results
    print(f"\nTraining Summary:")
    print(f"  Total training time: {format_time(training_duration)}")
    print(f"  Training loss: {train_result.training_loss:.4f}")
    print(f"  Training steps: {train_result.global_step}")
    
    # Get best metrics
    best_metrics = trainer.state.best_metric
    if best_metrics:
        print(f"  Best {training_args.metric_for_best_model}: {best_metrics:.4f}")
    
    # GPU memory summary
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print_gpu_memory()
        
        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak memory: {peak_memory:.2f}GB")
    
    # Log training summary to WandB
    log_to_wandb({
        'training/duration_seconds': training_duration,
        'training/final_loss': train_result.training_loss,
        'training/total_steps': train_result.global_step,
        'training/best_metric': best_metrics if best_metrics else 0.0
    })
    
    # Save training metrics
    training_metrics = {
        'training_loss': float(train_result.training_loss),
        'training_duration_seconds': training_duration,
        'training_duration_formatted': format_time(training_duration),
        'total_steps': train_result.global_step,
        'best_metric': float(best_metrics) if best_metrics else None,
        'timestamp': datetime.now().isoformat()
    }
    
    save_results(training_metrics, 'training_summary.json')
    
except Exception as e:
    print(f"\n{'='*80}")
    print(f"⚠ Training failed with error:")
    print(f"{'='*80}")
    print(f"{str(e)}")
    raise e

print(f"\n✓ Training phase complete!")

# %% [markdown]
# ## 2.4 Model Evaluation on Test Set

# %%
# ============================================================================
# 2.4 MODEL EVALUATION ON TEST SET
# ============================================================================

print("="*80)
print("Evaluating Model on Test Set")
print("="*80)

# Evaluate on validation set first
print("\n[1/2] Validation Set Evaluation:")
print("-" * 80)
val_results = trainer.evaluate(eval_dataset=tokenized_datasets['validation'])

print(f"\nValidation Results:")
for key, value in val_results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Evaluate on test set
print(f"\n[2/2] Test Set Evaluation:")
print("-" * 80)
test_results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])

print(f"\nTest Results:")
for key, value in test_results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Create comparison table
results_comparison = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1 Score'],
    'Validation': [
        val_results['eval_precision'],
        val_results['eval_recall'],
        val_results['eval_f1']
    ],
    'Test': [
        test_results['eval_precision'],
        test_results['eval_recall'],
        test_results['eval_f1']
    ]
})

print(f"\n{'='*80}")
print("Results Comparison:")
print(f"{'='*80}")
print(results_comparison.to_string(index=False))

# Calculate performance difference
print(f"\n{'='*80}")
print("Performance Analysis:")
print(f"{'='*80}")
f1_diff = test_results['eval_f1'] - val_results['eval_f1']
print(f"  F1 difference (Test - Val): {f1_diff:+.4f}")

if abs(f1_diff) < 0.02:
    print(f"  Status: ✓ Good generalization (difference < 2%)")
elif abs(f1_diff) < 0.05:
    print(f"  Status: ⚠ Moderate difference (2-5%)")
else:
    print(f"  Status: ⚠ Significant difference (>5%) - possible overfitting/underfitting")

# Log to WandB
log_to_wandb({
    'val/precision': val_results['eval_precision'],
    'val/recall': val_results['eval_recall'],
    'val/f1': val_results['eval_f1'],
    'test/precision': test_results['eval_precision'],
    'test/recall': test_results['eval_recall'],
    'test/f1': test_results['eval_f1'],
    'test/f1_diff_from_val': f1_diff
})

# Save evaluation results
evaluation_results = {
    'validation': {
        'precision': float(val_results['eval_precision']),
        'recall': float(val_results['eval_recall']),
        'f1': float(val_results['eval_f1']),
        'loss': float(val_results['eval_loss'])
    },
    'test': {
        'precision': float(test_results['eval_precision']),
        'recall': float(test_results['eval_recall']),
        'f1': float(test_results['eval_f1']),
        'loss': float(test_results['eval_loss'])
    },
    'comparison': {
        'f1_difference': float(f1_diff),
        'generalization_quality': 'good' if abs(f1_diff) < 0.02 else 'moderate' if abs(f1_diff) < 0.05 else 'poor'
    }
}

save_results(evaluation_results, 'evaluation_results.json')

# Generate predictions for detailed analysis
print(f"\n{'='*80}")
print("Generating predictions for analysis...")
print(f"{'='*80}")

# Get predictions on test set
predictions_output = trainer.predict(tokenized_datasets['test'])
predictions = predictions_output.predictions
labels = predictions_output.label_ids

# Convert to predicted class
predictions = np.argmax(predictions, axis=2)

print(f"\n✓ Predictions generated!")
print(f"  Shape: {predictions.shape}")
print(f"  Labels shape: {labels.shape}")

# Store predictions for later analysis
test_predictions = {
    'predictions': predictions,
    'labels': labels,
    'metrics': predictions_output.metrics
}

# Save predictions (as numpy arrays)
np.save(f"{RESULTS_DIR}/test_predictions.npy", predictions)
np.save(f"{RESULTS_DIR}/test_labels.npy", labels)

print(f"\n✓ Predictions saved to {RESULTS_DIR}/")
print(f"\n✓ Model evaluation complete!")

# %% [markdown]
# ## 2.5 Save Model Artifacts to WandB

# %%
# ============================================================================
# 2.5 SAVE MODEL ARTIFACTS TO WANDB
# ============================================================================

print("="*80)
print("Saving Model Artifacts")
print("="*80)

# Save model and tokenizer locally
print("\n[1/4] Saving model and tokenizer locally...")
model_save_path = f"{MODEL_SAVE_DIR}/best_model"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"✓ Model saved to: {model_save_path}")

# Save configuration files
print("\n[2/4] Saving configuration files...")

# Save tag mappings
mappings_save_path = f"{model_save_path}/tag_mappings.json"
with open(mappings_save_path, 'w') as f:
    json.dump({
        'tag2id': tag2id,
        'id2tag': id2tag,
        'num_labels': NUM_LABELS
    }, f, indent=2)

print(f"✓ Tag mappings saved to: {mappings_save_path}")

# Save training configuration
config_save_path = f"{model_save_path}/training_config.json"
with open(config_save_path, 'w') as f:
    json.dump({
        'model_name': PRIMARY_MODEL_NAME,
        'dataset_name': DATASET_NAME,
        'training_config': TRAINING_CONFIG,
        'entity_tiers': ENTITY_TIERS,
        'high_value_entities': HIGH_VALUE_ENTITIES,
        'remove_rare_entities': REMOVE_RARE_ENTITIES,
        'random_seed': RANDOM_SEED,
        'timestamp': TIMESTAMP
    }, f, indent=2)

print(f"✓ Training config saved to: {config_save_path}")

# Create model card
print("\n[3/4] Creating model card...")
model_card = f"""
# {MODEL_CONFIGS[PRIMARY_MODEL]['display_name']} for Medical NER

## Model Information
- **Base Model**: {PRIMARY_MODEL_NAME}
- **Task**: Named Entity Recognition (Token Classification)
- **Dataset**: {DATASET_NAME}
- **Number of Labels**: {NUM_LABELS}
- **Training Date**: {datetime.now().strftime('%Y-%m-%d')}

## Training Configuration
- Learning Rate: {TRAINING_CONFIG['learning_rate']}
- Batch Size: {TRAINING_CONFIG['batch_size']}
- Epochs: {TRAINING_CONFIG['num_epochs']}
- Warmup Steps: {TRAINING_CONFIG['warmup_steps']}
- Weight Decay: {TRAINING_CONFIG['weight_decay']}

## Performance Metrics

# Validation Set
- Precision: {val_results['eval_precision']:.4f}
- Recall: {val_results['eval_recall']:.4f}
- F1 Score: {val_results['eval_f1']:.4f}

# Test Set
- Precision: {test_results['eval_precision']:.4f}
- Recall: {test_results['eval_recall']:.4f}
- F1 Score: {test_results['eval_f1']:.4f}

## Entity Types
- Total Entity Types: {len(entity_df)}
- Frequent Entities: {len(entity_df[entity_df['Tier'] == 'frequent'])}
- Medium Entities: {len(entity_df[entity_df['Tier'] == 'medium'])}
- Rare Entities: {len(entity_df[entity_df['Tier'] == 'rare'])}

## Usage
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("{model_save_path}")
tokenizer = AutoTokenizer.from_pretrained("{model_save_path}")

# Example inference
text = "Patient is a 45-year-old male with diabetes."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=2)
```

## Notes

- Rare entities {'removed' if REMOVE_RARE_ENTITIES else 'included'}
- Random seed: {RANDOM_SEED} """

model_card_path = f"{model_save_path}/MODEL_CARD.md" 
with open(model_card_path, 'w') as f: f.write(model_card)

print(f"✓ Model card saved to: {model_card_path}")

# Upload to WandB

if WANDB_ENABLED: print("\n[4/4] Uploading artifacts to WandB...")
try:
    # Create model artifact
    model_artifact = wandb.Artifact(
        name=f"pubmedbert-ner-{TIMESTAMP}",
        type="model",
        description=f"Fine-tuned {MODEL_CONFIGS[PRIMARY_MODEL]['display_name']} for Medical NER",
        metadata={
            'model_name': PRIMARY_MODEL_NAME,
            'test_f1': float(test_results['eval_f1']),
            'test_precision': float(test_results['eval_precision']),
            'test_recall': float(test_results['eval_recall']),
            'num_labels': NUM_LABELS
        }
    )
    
    # Add model directory
    model_artifact.add_dir(model_save_path)
    
    # Log artifact
    wandb.log_artifact(model_artifact)
    
    print(f"✓ Model artifact uploaded to WandB")
    
    # Create results artifact
    results_artifact = wandb.Artifact(
        name=f"results-{TIMESTAMP}",
        type="results",
        description="Training and evaluation results"
    )
        
    # Add results directory
    results_artifact.add_dir(RESULTS_DIR)
    
    # Log artifact
    wandb.log_artifact(results_artifact)
    
    print(f"✓ Results artifact uploaded to WandB")
    
    # Create plots artifact
    plots_artifact = wandb.Artifact(
        name=f"plots-{TIMESTAMP}",
        type="plots",
        description="Visualizations and plots"
    )
    
    # Add plots directory
    plots_artifact.add_dir(PLOTS_DIR)
    
    # Log artifact
    wandb.log_artifact(plots_artifact)
    
    print(f"✓ Plots artifact uploaded to WandB")
    
except Exception as e:
    print(f"⚠ Failed to upload to WandB: {e}")

else: print("\n[4/4] WandB not enabled - skipping upload")

print(f"\n{'='*80}") 
print("✓ All artifacts saved successfully!") 
print(f"{'='*80}") 
print(f"\nLocal paths:") 
print(f" Model: {model_save_path}") 
print(f" Results: {RESULTS_DIR}") 
print(f" Plots: {PLOTS_DIR}")

if WANDB_ENABLED: print(f"\nWandB Dashboard: {wandb.run.get_url()}")

# %% [markdown]
# ---

# %% [markdown]
# # 3. Additional Model Training

# %% [markdown]
# ## 3.1 BioBERT Fine-tuning

# %% [markdown]
# ## 3.2 Alternative Model Training (Bioformer/ClinicalBERT)

# %% [markdown]
# ## 3.3 Model Comparison Matrix

# %% [markdown]
# ## 3.4 Best Model Selection Criteria

# %% [markdown]
# ---

# %% [markdown]
# # 4. Per-Entity-Type Performance Analysis

# %% [markdown]
# ## 4.1 Extract Predictions by Entity Type

# %% [markdown]
# ## 4.2 Calculate Metrics per Entity Type

# %% [markdown]
# ## 4.3 Tier-Based Analysis (Frequent/Medium/Rare)

# %% [markdown]
# ## 4.4 Identify Systematic Weaknesses

# %% [markdown]
# ## 4.5 Visualizations

# %% [markdown]
# ### 4.5.1 F1 Score by Entity Type (Bar Chart)

# %% [markdown]
# ### 4.5.2 Entity Frequency vs F1 Score (Scatter Plot)

# %% [markdown]
# ### 4.5.3 Confusion Matrix Heatmap

# %% [markdown]
# ---

# %% [markdown]
# # 5. Error Analysis Framework

# %% [markdown]
# ## 5.1 Boundary Error Detection

# %% [markdown]
# ### 5.1.1 Partial Match Analysis

# %% [markdown]
# ### 5.1.2 Exact Match vs Partial Match Rate

# %% [markdown]
# ## 5.2 Entity Type Confusion Analysis

# %% [markdown]
# ### 5.2.1 Build Confusion Matrix

# %% [markdown]
# ### 5.2.2 Common Confusion Patterns

# %% [markdown]
# ## 5.3 Multi-token Entity Analysis

# %% [markdown]
# ### 5.3.1 Single-token vs Multi-token Performance

# %% [markdown]
# ### 5.3.2 Entity Length Impact on F1

# %% [markdown]
# ## 5.4 Error Examples Collection

# %% [markdown]
# ### 5.4.1 Sample Mispredictions

# %% [markdown]
# ### 5.4.2 Error Categorization

# %% [markdown]
# ### 5.4.3 Pattern Documentation

# %% [markdown]
# ---

# %% [markdown]
# # 6. Confidence Score Analysis

# %% [markdown]
# ## 6.1 Extract Prediction Confidence Scores

# %% [markdown]
# ## 6.2 Confidence Distribution Analysis

# %% [markdown]
# ### 6.2.1 Correct vs Incorrect Predictions

# %% [markdown]
# ### 6.2.2 Average Confidence by Prediction Type (TP/FP/FN)

# %% [markdown]
# ## 6.3 Threshold Analysis for Production

# %% [markdown]
# ### 6.3.1 Test Multiple Thresholds [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

# %% [markdown]
# ### 6.3.2 Precision/Coverage/Recall at Each Threshold

# %% [markdown]
# ## 6.4 Precision-Coverage Tradeoff Visualization

# %% [markdown]
# ## 6.5 Production Threshold Recommendation

# %% [markdown]
# ---

# %% [markdown]
# # 7. Inference Speed & Scalability Testing

# %% [markdown]
# ## 7.1 Measure Inference Latency

# %% [markdown]
# ### 7.1.1 Per Document (Batch Size = 1)

# %% [markdown]
# ### 7.1.2 Per Batch (Batch Size = 8, 16, 32)

# %% [markdown]
# ## 7.2 Memory Profiling

# %% [markdown]
# ### 7.2.1 GPU Memory During Training

# %% [markdown]
# ### 7.2.2 GPU Memory During Inference

# %% [markdown]
# ### 7.2.3 CPU Inference Feasibility

# %% [markdown]
# ## 7.3 Throughput Estimation

# %% [markdown]
# ### 7.3.1 Documents per Second

# %% [markdown]
# ### 7.3.2 Large-scale Processing Estimates

# %% [markdown]
# ## 7.4 Scalability Considerations

# %% [markdown]
# ---

# %% [markdown]
# # 8. Production-Readiness Metrics

# %% [markdown]
# ## 8.1 Clinical Coding Simulation

# %% [markdown]
# ### 8.1.1 Auto-code vs Manual Review Decision

# %% [markdown]
# ### 8.1.2 Automation Rate at Different Thresholds

# %% [markdown]
# ## 8.2 High-Value Entity Analysis

# %% [markdown]
# ### 8.2.1 F1 on Critical Entities (Disease_disorder, Therapeutic_procedure, Medication)

# %% [markdown]
# ### 8.2.2 Precision on High-Value Entities at 0.85 Confidence

# %% [markdown]
# ## 8.3 False Positive Impact Analysis

# %% [markdown]
# ### 8.3.1 FP Rate by Entity Type

# %% [markdown]
# ### 8.3.2 High-Risk Entity Identification

# %% [markdown]
# ## 8.4 Production Readiness Report Generation

# %% [markdown]
# ---

# %% [markdown]
# # 9. Model Selection & Final Documentation

# %% [markdown]
# ## 9.1 Best Model Selection

# %% [markdown]
# ### 9.1.1 Selection Criteria & Tradeoffs

# %% [markdown]
# ### 9.1.2 Justification Documentation

# %% [markdown]
# ## 9.2 Phase 1 Summary

# %% [markdown]
# ### 9.2.1 Training Results Comparison Table

# %% [markdown]
# ### 9.2.2 Key Visualizations Compilation

# %% [markdown]
# ### 9.2.3 Error Analysis Summary

# %% [markdown]
# ### 9.2.4 Production Metrics Summary

# %% [markdown]
# ## 9.3 Model Artifacts Export

# %% [markdown]
# ### 9.3.1 Save Best Checkpoint

# %% [markdown]
# ### 9.3.2 Save Tokenizer & Mappings

# %% [markdown]
# ### 9.3.3 Save Evaluation Predictions

# %% [markdown]
# ## 9.4 Phase 1 Insights Document

# %% [markdown]
# ### 9.4.1 Key Findings

# %% [markdown]
# ### 9.4.2 Surprising Results

# %% [markdown]
# ### 9.4.3 Recommendations for Phase 2

# %% [markdown]
# ---

# %% [markdown]
# # 10. Appendix

# %% [markdown]
# ## 10.1 Utility Functions Reference

# %% [markdown]
# ## 10.2 Hyperparameter Log

# %% [markdown]
# ## 10.3 Raw Results Tables

# %% [markdown]
# ## 10.4 Additional Visualizations


